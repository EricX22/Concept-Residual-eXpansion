#!/usr/bin/env python3
import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path

from crx import hparams_registry


def main():
    ap = argparse.ArgumentParser("Run a CRX training job with an artifact directory (no manual JSON).")

    # core train.py args
    ap.add_argument("--dataset", default="Waterbirds", choices=["Waterbirds", "CelebA", "CheXpertNoFinding"])
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--output-dir", default="./output")
    ap.add_argument("--output-folder-name", default="crx_run")

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hparams-seed", type=int, default=0)

    # stage-1 location (train.py expects output_dir/<stage1_folder>/<dataset>_<stage1_algo>_hparamsX_seedY/model.pkl)
    ap.add_argument("--stage1-folder", required=True, help="Folder name under output_dir for stage-1 runs (e.g., vanilla_attrNo)")
    ap.add_argument("--stage1-algo", default="ERM")

    # artifact dir
    ap.add_argument("--artifact-dir", required=True,
                    help="Directory created by setup_crx_artifacts (contains meta.json, concepts_*.pt, resid_*.pt, resid_meta.json)")

    # optional overrides (nice for quick runs)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=None)

    args = ap.parse_args()

    art = Path(args.artifact_dir)
    # canonical filenames produced by setup_crx_artifacts
    concept_meta = art / "meta.json"
    resid_meta = art / "resid_meta.json"
    concepts_tr = art / "concepts_tr.pt"
    concepts_va = art / "concepts_va.pt"
    concepts_te = art / "concepts_te.pt"
    resid_va = art / "resid_va.pt"
    resid_te = art / "resid_te.pt"

    required = [concept_meta, concepts_tr, concepts_va, concepts_te, resid_va, resid_te, resid_meta]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit("Missing required artifact files:\n  " + "\n  ".join(missing))

    # start from default hparams (so you keep swept knobs there)
    hp = hparams_registry.default_hparams("CRX", args.dataset)

    # inject artifact paths (NOT in registry)
    hp.update({
        "cr_concept_meta_path": str(concept_meta),
        "cr_concept_path_tr": str(concepts_tr),
        "cr_concept_path_va": str(concepts_va),
        "cr_concept_path_te": str(concepts_te),

        "cr_resid_path_va": str(resid_va),
        "cr_resid_path_te": str(resid_te),
        "cr_resid_meta_path": str(resid_meta),
    })

    # stage-1 invariants (never sweep these)
    hp.update({
        "stage1_hparams_seed": 0,
        "stage1_seed": 0,
        "stage1_model": "model.pkl",
    })

    # optional overrides
    if args.batch_size is not None:
        hp["batch_size"] = int(args.batch_size)
    if args.lr is not None:
        hp["lr"] = float(args.lr)
    if args.weight_decay is not None:
        hp["weight_decay"] = float(args.weight_decay)

    # build the train.py command
    cmd = [
        "python", "-m", "crx.train",
        "--dataset", args.dataset,
        "--algorithm", "CRX",
        "--train_attr", "no",
        "--data_dir", args.data_dir,
        "--output_dir", args.output_dir,
        "--output_folder_name", args.output_folder_name,
        "--seed", str(args.seed),
        "--hparams_seed", str(args.hparams_seed),
        "--stage1_folder", args.stage1_folder,
        "--stage1_algo", args.stage1_algo,
        "--hparams", json.dumps(hp),
    ]

    print("==> Running:")
    print(" ".join(shlex.quote(x) for x in cmd))
    print("")

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
