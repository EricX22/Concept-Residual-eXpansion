#!/usr/bin/env python3
import argparse
from pathlib import Path
import json
import torch


def ridge_fit(C: torch.Tensor, F: torch.Tensor, lam: float) -> torch.Tensor:
    """
    Solve W = argmin ||F - C W||^2 + lam ||W||^2.

    Numerically stable:
      - compute in float64 on CPU
      - solve SPD system with Cholesky
    """
    if lam <= 0:
        raise ValueError(f"ridge_lam must be > 0 for a well-posed ridge system, got {lam}")

    C = C.detach().to(device="cpu", dtype=torch.float64)
    F = F.detach().to(device="cpu", dtype=torch.float64)

    K = C.shape[1]
    A = C.T @ C + float(lam) * torch.eye(K, dtype=torch.float64)
    B = C.T @ F

    L = torch.linalg.cholesky(A)
    W = torch.cholesky_solve(B, L)

    return W.to(dtype=torch.float32)


def pca_fit_transform(X: torch.Tensor, k: int):
    """
    PCA via SVD on centered data.
    Returns:
      Z = (X - mu) @ V_k   [N, k]
      mu: [d]
      V_k: [d, k]
    """
    X = X.float()
    mu = X.mean(dim=0, keepdim=True)
    Xc = X - mu
    U, S, Vt = torch.linalg.svd(Xc, full_matrices=False)
    V = Vt.T
    Vk = V[:, :k].contiguous()
    Z = (Xc @ Vk).contiguous()
    return Z, mu.squeeze(0), Vk


def apply_pca(X: torch.Tensor, mu: torch.Tensor, Vk: torch.Tensor) -> torch.Tensor:
    Xc = X.float() - mu.view(1, -1)
    return (Xc @ Vk).contiguous()


def _unwrap_tensor(obj):
    """Accept raw tensor or dict-wrapped tensor."""
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, dict):
        # Support both canonical and legacy keys
        if "concepts" in obj:
            return obj["concepts"]
        if "concept_logits" in obj:
            return obj["concept_logits"]
        if "feats" in obj:
            return obj["feats"]
        if "resid" in obj:
            return obj["resid"]
    return obj


def load_concepts(cache_path: str, use_mask: torch.Tensor):
    cache = torch.load(cache_path, map_location="cpu")
    C = _unwrap_tensor(cache)
    if not isinstance(C, torch.Tensor):
        raise TypeError(f"Concept cache at {cache_path} did not contain a tensor (got {type(C)})")
    C = C.float()

    if use_mask is not None:
        if use_mask.dtype != torch.bool:
            use_mask = use_mask.to(dtype=torch.bool)
        if use_mask.numel() != C.shape[1]:
            raise ValueError(
                f"use_mask length mismatch for {cache_path}: "
                f"use_mask has {use_mask.numel()} but concepts have K={C.shape[1]}"
            )
        C = C[:, use_mask]

    return C.contiguous()


def load_feats(feat_path: str) -> torch.Tensor:
    obj = torch.load(feat_path, map_location="cpu")
    F = _unwrap_tensor(obj)
    if not isinstance(F, torch.Tensor):
        raise TypeError(f"Feature cache at {feat_path} did not contain a tensor (got {type(F)})")
    return F.float().contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept_meta", required=True)
    ap.add_argument("--concept_va", required=True)
    ap.add_argument("--concept_te", required=True)
    ap.add_argument("--feat_va", required=True)
    ap.add_argument("--feat_te", required=True)
    ap.add_argument("--block_channels", default="", help="comma-separated channels to block (optional)")
    ap.add_argument("--ridge_lam", type=float, default=1e-2)
    ap.add_argument("--pca_k", type=int, default=64)
    ap.add_argument("--out_va", required=True)
    ap.add_argument("--out_te", required=True)
    ap.add_argument("--out_meta", required=True)
    args = ap.parse_args()

    with open(args.concept_meta, "r") as f:
        meta = json.load(f)

    channels = meta.get("channel", None)
    use_in_training = meta.get("use_in_training", None)

    block = set([c.strip() for c in args.block_channels.split(",") if c.strip()])

    # Build use_mask exactly like CRX would (use_in_training AND not blocked channel)
    if channels is None:
        use_mask = None
    else:
        channel_mask = torch.tensor([c not in block for c in channels], dtype=torch.bool)
        if use_in_training is None:
            use_mask = channel_mask
        else:
            use_mask = torch.as_tensor(use_in_training, dtype=torch.bool) & channel_mask

    # Load concepts (va/te) with same mask as CRX would use
    C_va = load_concepts(args.concept_va, use_mask)
    C_te = load_concepts(args.concept_te, use_mask)

    # Load features (va/te)
    F_va = load_feats(args.feat_va)
    F_te = load_feats(args.feat_te)

    # IMPORTANT: this assumes both caches are indexed in the same first-dimension space
    # (often max_i+1 “dataset_i” index space).
    assert C_va.shape[0] == F_va.shape[0], f"va N mismatch: C {C_va.shape} vs F {F_va.shape}"
    assert C_te.shape[0] == F_te.shape[0], f"te N mismatch: C {C_te.shape} vs F {F_te.shape}"
    print(f"C_va {tuple(C_va.shape)}  F_va {tuple(F_va.shape)}")

    # Fit ridge on VA (consistent with stage-2 training on VA; avoids leakage)
    W = ridge_fit(C_va, F_va, lam=args.ridge_lam)  # [K_used, d]

    # Residuals
    R_va = (F_va - C_va @ W).contiguous()
    R_te = (F_te - C_te @ W).contiguous()

    # PCA compress on residuals (fit on VA)
    if args.pca_k <= 0:
        raise ValueError(f"pca_k must be > 0, got {args.pca_k}")
    if args.pca_k > R_va.shape[1]:
        raise ValueError(f"pca_k={args.pca_k} > resid dim d={R_va.shape[1]}")

    Z_va, mu, Vk = pca_fit_transform(R_va, k=args.pca_k)
    Z_te = apply_pca(R_te, mu, Vk)

    out_va = Path(args.out_va); out_va.parent.mkdir(parents=True, exist_ok=True)
    out_te = Path(args.out_te); out_te.parent.mkdir(parents=True, exist_ok=True)

    torch.save({"resid": Z_va}, out_va)
    torch.save({"resid": Z_te}, out_te)

    out_meta = Path(args.out_meta); out_meta.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "ridge_lam": float(args.ridge_lam),
        "pca_k": int(args.pca_k),
        "blocked_channels": sorted(list(block)),
        "use_mask_sum": int(use_mask.sum().item()) if use_mask is not None else None,
        "W": W,      # optional: keep for reproducibility
        "mu": mu,
        "Vk": Vk
    }, out_meta)

    print(f"Saved resid va: {out_va}  shape={tuple(Z_va.shape)}")
    print(f"Saved resid te: {out_te}  shape={tuple(Z_te.shape)}")
    print(f"Saved resid meta: {out_meta}")


if __name__ == "__main__":
    main()
