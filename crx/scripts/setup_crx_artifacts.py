#!/usr/bin/env python3
"""
Thin CRX artifact setup orchestrator.

Runs:
  (1) generate_concept_bank.py
  (2) compile_concept_meta.py
  (3) cache_clip_concepts.py for tr/va/te (always saves fp32)
  (4) cache_stage1_feats.py for va/te
  (5) fit_residuals_pca.py (ridge on va; residuals for va/te; PCA -> k)

Required inputs:
  --dataset, --data-dir, --stage1-folder, --artifact-dir, --resid-dim

Notes:
  - This script does NOT train stage-1. It expects stage-1 model.pkl already exists.
  - For cache_stage1_feats.py we must pass --stage1_ckpt explicitly. We resolve it from:
        --stage1-folder + dataset name (Waterbirds/CelebA) + optional overrides.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def run_cmd(cmd: List[str]) -> None:
    print("\n$ " + " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(r.returncode)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def find_stage1_ckpt(stage1_folder: Path, dataset: str, stage1_algo: str, stage1_hp: int, stage1_seed: int) -> Path:
    """
    Resolves the canonical CRX output path:
      <stage1_folder>/<Dataset>_<Algo>_hparams{hp}_seed{seed}/model.pkl
    """
    run_dir = stage1_folder / f"{dataset}_{stage1_algo}_hparams{stage1_hp}_seed{stage1_seed}"
    ckpt = run_dir / "model.pkl"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Stage-1 checkpoint not found:\n  {ckpt}\n"
            f"Expected run dir:\n  {run_dir}\n"
            f"Tip: set --stage1-hp/--stage1-seed/--stage1-algo to match an existing ERM run."
        )
    return ckpt


def dataset_defaults(dataset: str) -> Dict[str, object]:
    """
    Minimal dataset->concept-bank defaults.
    Extend as needed.
    """
    if dataset == "Waterbirds":
        return {
            "task_id": "waterbirds",
            "modality": "natural_image",
            "labels": [{"name": "landbird"}, {"name": "waterbird"}],
        }
    if dataset == "CelebA":
        # You may want to adjust depending on which CelebA target label you're using in the paper.
        # This is a safe placeholder for concept-bank generation.
        return {
            "task_id": "celeba",
            "modality": "natural_image",
            "labels": [{"name": "smiling"}, {"name": "not_smiling"}],
        }
    raise ValueError(f"No dataset defaults defined for: {dataset}")


def main() -> None:
    p = argparse.ArgumentParser("Setup CRX artifacts (concept bank/meta, CLIP cache, stage-1 feats, residual PCA).")

    # Minimum required set
    p.add_argument("--dataset", type=str, required=True, choices=["Waterbirds", "CelebA"])
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--stage1-folder", type=str, required=True,
                   help="Folder containing stage-1 CRX runs (each has model.pkl).")
    p.add_argument("--artifact-dir", type=str, required=True,
                   help="Output directory for all CRX artifacts (recommend on /scratch).")
    p.add_argument("--resid-dim", type=int, required=True, help="Residual PCA dimension k.")

    # Small, useful overrides (optional; does not change the minimum conceptually)
    p.add_argument("--overwrite", action="store_true", help="Re-generate outputs even if they already exist.")
    p.add_argument("--python", type=str, default=sys.executable, help="Python executable (default: current).")

    # Stage-1 resolution knobs
    p.add_argument("--stage1-algo", type=str, default="ERM")
    p.add_argument("--stage1-hp", type=int, default=0)
    p.add_argument("--stage1-seed", type=int, default=0)

    # cache_stage1_feats args
    p.add_argument("--train-attr", type=str, default="no", choices=["yes", "no"],
                   help="Must match how stage-1 was trained (attrNo vs attrYes). Default: no")
    p.add_argument("--image-arch", type=str, default="resnet50",
                   help="Featurizer arch used for stage-1 (default: resnet50).")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=8)

    # cache_clip_concepts args (optional overrides)
    p.add_argument("--clip-model", type=str, default="RN50")
    p.add_argument("--clip-pretrained", type=str, default="openai")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--text-batch-size", type=int, default=64)

    # compile_concept_meta toggle
    p.add_argument("--include-artifacts", action="store_true",
                   help="Pass --include-artifacts to compile_concept_meta (include imaging_artifacts in training).")

    args = p.parse_args()

    dataset = args.dataset
    data_dir = Path(args.data_dir).expanduser().resolve()
    stage1_folder = Path(args.stage1_folder).expanduser().resolve()
    artifact_dir = Path(args.artifact_dir).expanduser().resolve()
    ensure_dir(artifact_dir)

    bank_path = artifact_dir / "concept_bank.json"
    meta_path = artifact_dir / "meta.json"

    concept_cache = {s: artifact_dir / f"concepts_{s}.pt" for s in ["tr", "va", "te"]}
    feat_cache = {s: artifact_dir / f"stage1_feats_{s}.pt" for s in ["va", "te"]}
    resid_cache = {s: artifact_dir / f"resid_{s}.pt" for s in ["va", "te"]}
    resid_meta = artifact_dir / "resid_meta.json"

    # Resolve stage-1 checkpoint
    stage1_ckpt = find_stage1_ckpt(stage1_folder, dataset, args.stage1_algo, args.stage1_hp, args.stage1_seed)

    # ----------------------------
    # (1) Concept bank
    # ----------------------------
    info = dataset_defaults(dataset)
    labels_json = json.dumps(info["labels"])  # avoids shell quoting issues

    if args.overwrite or not bank_path.exists():
        run_cmd([
            args.python, "-m", "crx.scripts.generate_concept_bank",
            "--task-id", str(info["task_id"]),
            "--modality", str(info["modality"]),
            "--labels", labels_json,
            "--out", str(bank_path),
        ])
    else:
        print(f"[skip] concept bank exists: {bank_path}")

    # ----------------------------
    # (2) Compile meta
    # ----------------------------
    if args.overwrite or not meta_path.exists():
        cmd = [
            args.python, "-m", "crx.scripts.compile_concept_meta",
            "--bank", str(bank_path),
            "--out", str(meta_path),
        ]
        if args.include_artifacts:
            cmd.append("--include-artifacts")
        run_cmd(cmd)
    else:
        print(f"[skip] concept meta exists: {meta_path}")

    # ----------------------------
    # (3) Cache CLIP concepts (tr/va/te)
    # ----------------------------
    for split in ["tr", "va", "te"]:
        outp = concept_cache[split]
        if (not args.overwrite) and outp.exists():
            print(f"[skip] CLIP concept cache exists: {outp}")
            continue

        run_cmd([
            args.python, "-m", "crx.scripts.cache_clip_concepts",
            "--dataset", dataset,
            "--data-dir", str(data_dir),
            "--split", split,
            "--meta", str(meta_path),
            "--out", str(outp),
            "--batch-size", str(args.batch_size),
            "--num-workers", str(args.num_workers),
            "--device", str(args.device),
            "--clip-model", str(args.clip_model),
            "--clip-pretrained", str(args.clip_pretrained),
            "--text-batch-size", str(args.text_batch_size),
        ])

    # ----------------------------
    # (4) Cache stage-1 features (va/te)
    # ----------------------------
    for split in ["va", "te"]:
        outp = feat_cache[split]
        if (not args.overwrite) and outp.exists():
            print(f"[skip] stage-1 feature cache exists: {outp}")
            continue

        run_cmd([
            args.python, "-m", "crx.scripts.cache_stage1_feats",
            "--dataset", dataset,
            "--data_dir", str(data_dir),
            "--split", split,
            "--train_attr", str(args.train_attr),
            "--image_arch", str(args.image_arch),
            "--stage1_ckpt", str(stage1_ckpt),
            "--out", str(outp),
            "--batch_size", str(args.batch_size),
            "--num_workers", str(args.num_workers),
        ])

    # ----------------------------
    # (5) Fit residuals PCA
    # ----------------------------
    need_run = args.overwrite or (not resid_cache["va"].exists()) or (not resid_cache["te"].exists()) or (not resid_meta.exists())
    if need_run:
        run_cmd([
            args.python, "-m", "crx.scripts.fit_residuals_pca",
            "--concept_meta", str(meta_path),
            "--concept_va", str(concept_cache["va"]),
            "--concept_te", str(concept_cache["te"]),
            "--feat_va", str(feat_cache["va"]),
            "--feat_te", str(feat_cache["te"]),
            "--pca_k", str(int(args.resid_dim)),
            "--out_va", str(resid_cache["va"]),
            "--out_te", str(resid_cache["te"]),
            "--out_meta", str(resid_meta),
        ])
    else:
        print(f"[skip] residual caches exist: {resid_cache['va']}, {resid_cache['te']} and {resid_meta}")

    print("\nCRX setup complete.")
    print(f"Artifact dir: {artifact_dir}")
    print(f"  stage1_ckpt: {stage1_ckpt}")
    print(f"  meta:        {meta_path}")
    print(f"  concepts:    {concept_cache['tr']}, {concept_cache['va']}, {concept_cache['te']}")
    print(f"  stage1 feats:{feat_cache['va']}, {feat_cache['te']}")
    print(f"  residuals:   {resid_cache['va']}, {resid_cache['te']}")
    print(f"  resid meta:  {resid_meta}")


if __name__ == "__main__":
    main()
