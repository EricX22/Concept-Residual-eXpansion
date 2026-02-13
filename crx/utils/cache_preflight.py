import os
import time
import subprocess
from pathlib import Path
from typing import Optional
from contextlib import contextmanager
import json

@contextmanager
def file_lock(lock_path: Path, poll_s: float = 2.0):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as f:
        try:
            import fcntl
            while True:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(poll_s)
        except Exception:
            # best-effort fallback
            pass
        yield


def _run(cmd, cwd: str):
    print("[preflight] running:", " ".join(map(str, cmd)), flush=True)
    subprocess.check_call(list(map(str, cmd)), cwd=cwd)


def ensure_cr_caches(
    *,
    dataset: str,
    data_dir: str,
    repo_dir: str,
    hparams: dict,
    stage1_ckpt: Optional[str] = None,
):
    """
    Auto-create concept caches and residual caches needed by CR if missing.

    Uses crx/scripts CLIs:
      - generate_concept_bank.py
      - compile_concept_meta.py
      - cache_clip_concepts.py
      - cache_stage1_feats.py
      - fit_residuals_pca.py

    NOTE: residuals require feat_va/feat_te + concept_va/concept_te + concept_meta.
    """
    use_concepts = bool(hparams.get("cr_use_concepts", False))
    use_resid = bool(hparams.get("cr_use_resid", False))
    if not (use_concepts or use_resid):
        return

    # NEW: if user provided explicit cache paths, don't rebuild
    if use_concepts:
        has_concept_paths = all([
            hparams.get("cr_concept_meta_path"),
            hparams.get("cr_concept_path_tr"),
            hparams.get("cr_concept_path_va"),
            hparams.get("cr_concept_path_te"),
        ])
    else:
        has_concept_paths = True

    if use_resid:
        has_resid_paths = all([
            hparams.get("cr_resid_path_va"),
            hparams.get("cr_resid_path_te"),
        ])
    else:
        has_resid_paths = True

    # If everything needed is explicitly provided, trust it and exit.
    if has_concept_paths and has_resid_paths:
        return


    repo_dir = str(repo_dir)
    artifacts = Path(repo_dir) / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Decide a stable task_id
    # -----------------------------
    # Prefer explicit config; else map from dataset.
    task_id = hparams.get("cr_task_id", None)
    if task_id is None:
        # You can extend this mapping later.
        if dataset.lower() == "waterbirds":
            task_id = "waterbirds_binary"
        else:
            # fallback: dataset name (user should set cr_task_id for CelebA etc.)
            task_id = dataset.lower()

    lock = artifacts / ".locks" / f"cr_cache_{task_id}.lock"

    # -----------------------------
    # Paths (match your current naming conventions)
    # -----------------------------
    concepts_dir = artifacts / "concepts"
    scores_dir = artifacts / "concept_scores"
    feat_dir = artifacts / "feat_cache"
    resid_dir = artifacts / "resid_cache"
    for d in [concepts_dir, scores_dir, feat_dir, resid_dir]:
        d.mkdir(parents=True, exist_ok=True)

    bank_path = concepts_dir / f"{task_id}_bank.json"
    meta_path = scores_dir / f"{task_id}_meta.json"
    tr_concept = scores_dir / f"{task_id}_tr.pt"
    va_concept = scores_dir / f"{task_id}_va.pt"
    te_concept = scores_dir / f"{task_id}_te.pt"

    # Residual cache naming (keep consistent with your Waterbirds ones)
    pca_k = int(hparams.get("cr_resid_dim", 64) or 64)
    va_resid = resid_dir / f"{task_id}_resid_va_pca{pca_k}.pt"
    te_resid = resid_dir / f"{task_id}_resid_te_pca{pca_k}.pt"
    resid_meta = resid_dir / f"{task_id}_resid_meta_pca{pca_k}.pt"

    # Stage1 feature caches required by fit_residuals_pca
    # (You can choose names; this is just a consistent scheme)
    va_feat = feat_dir / f"{task_id}_feat_va.pt"
    te_feat = feat_dir / f"{task_id}_feat_te.pt"

    with file_lock(lock):
        # ------------------------------------------------------------
        # (1) Concept bank JSON (if missing)
        # ------------------------------------------------------------
        if use_concepts or use_resid and not bank_path.exists():
            modality = hparams.get("cr_modality", None)
            labels_json = hparams.get("cr_labels_json", None)
            if modality is None or labels_json is None:
                raise ValueError(
                    f"[preflight] Missing cr_modality / cr_labels_json needed to generate concept bank for task_id={task_id}. "
                    f"(For waterbirds_binary you can set these once in hparams_registry.)"
                )

            labels_arg = labels_json if isinstance(labels_json, str) else json.dumps(labels_json)

            _run(
                [
                    "python", "-m", "crx.scripts.generate_concept_bank",
                    "--task-id", task_id,
                    "--modality", modality,
                    "--labels", labels_arg,
                    "--out", str(bank_path),
                ],
                cwd=repo_dir,
            )

        # ------------------------------------------------------------
        # (2) Compile meta JSON (if missing)
        # ------------------------------------------------------------
        if use_concepts or use_resid and not meta_path.exists():
            _run(
                [
                    "python", "-m", "crx.scripts.compile_concept_meta",
                    "--bank", str(bank_path),
                    "--out", str(meta_path),
                ],
                cwd=repo_dir,
            )

        # ------------------------------------------------------------
        # (3) Cache CLIP concept probs for splits (if missing)
        # ------------------------------------------------------------
        if use_concepts or use_resid:
            for split, outp in [("tr", tr_concept), ("va", va_concept), ("te", te_concept)]:
                if not outp.exists():
                    _run(
                        [
                            "python", "-m", "crx.scripts.cache_clip_concepts",
                            "--dataset", dataset,
                            "--data-dir", data_dir,
                            "--split", split,
                            "--meta", str(meta_path),
                            "--out", str(outp),
                            "--fp16",
                        ],
                        cwd=repo_dir,
                    )

        # Wire hparams to the files we just ensured
        if use_concepts or use_resid:
            hparams["cr_concept_meta_path"] = str(meta_path)
            hparams["cr_concept_path_tr"] = str(tr_concept)
            hparams["cr_concept_path_va"] = str(va_concept)
            hparams["cr_concept_path_te"] = str(te_concept)


        # ------------------------------------------------------------
        # (4) If residuals requested, ensure stage1 feature caches exist
        # ------------------------------------------------------------
        if use_resid:
            # If user already provided feat paths, respect them
            feat_va_hp = hparams.get("cr_feat_path_va", None)
            feat_te_hp = hparams.get("cr_feat_path_te", None)
            if feat_va_hp is not None:
                va_feat = Path(feat_va_hp)
            if feat_te_hp is not None:
                te_feat = Path(feat_te_hp)

            # Need stage1 ckpt + image_arch to produce feats
            if stage1_ckpt is None or not Path(stage1_ckpt).exists():
                raise ValueError(
                    f"[preflight] Residuals requested but stage1_ckpt missing/not found: {stage1_ckpt}"
                )

            image_arch = hparams.get("image_arch", None)
            if image_arch is None:
                raise ValueError("[preflight] Residuals requested but hparams['image_arch'] missing.")

            train_attr = str(hparams.get("train_attr", "no"))  # safe default

            # Cache va feats
            if not va_feat.exists():
                _run(
                    [
                        "python", "-m", "crx.scripts.cache_stage1_feats",
                        "--dataset", dataset,
                        "--data_dir", data_dir,
                        "--split", "va",
                        "--train_attr", train_attr,
                        "--image_arch", image_arch,
                        "--stage1_ckpt", str(stage1_ckpt),
                        "--out", str(va_feat),
                    ],
                    cwd=repo_dir,
                )

            # Cache te feats
            if not te_feat.exists():
                _run(
                    [
                        "python", "-m", "crx.scripts.cache_stage1_feats",
                        "--dataset", dataset,
                        "--data_dir", data_dir,
                        "--split", "te",
                        "--train_attr", train_attr,
                        "--image_arch", image_arch,
                        "--stage1_ckpt", str(stage1_ckpt),
                        "--out", str(te_feat),
                    ],
                    cwd=repo_dir,
                )

            # Tell residual fitter where feats are (optional, but useful)
            hparams["cr_feat_path_va"] = str(va_feat)
            hparams["cr_feat_path_te"] = str(te_feat)

            # --------------------------------------------------------
            # (5) Fit residuals PCA (if missing)
            # --------------------------------------------------------
            if (not va_resid.exists()) or (not te_resid.exists()) or (not resid_meta.exists()):
                block_channels = hparams.get("cr_block_channels_csv", None)  # optional "a,b,c"
                ridge_lam = float(hparams.get("cr_ridge_lam", 1e-3))
                cmd = [
                    "python", "-m", "crx.scripts.fit_residuals_pca",
                    "--concept_meta", str(meta_path),
                    "--concept_va", str(va_concept),
                    "--concept_te", str(te_concept),
                    "--feat_va", str(va_feat),
                    "--feat_te", str(te_feat),
                    "--ridge_lam", str(ridge_lam),
                    "--pca_k", str(pca_k),
                    "--out_va", str(va_resid),
                    "--out_te", str(te_resid),
                    "--out_meta", str(resid_meta),
                ]
                if block_channels:
                    cmd += ["--block_channels", str(block_channels)]
                _run(cmd, cwd=repo_dir)

            hparams["cr_resid_path_va"] = str(va_resid)
            hparams["cr_resid_path_te"] = str(te_resid)
