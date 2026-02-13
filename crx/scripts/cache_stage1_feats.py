import argparse
import os
from pathlib import Path
import torch
import numpy as np

from crx.dataset import datasets
from crx.models import networks
from crx import hparams_registry

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Waterbirds")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", choices=["va", "te"], required=True)
    ap.add_argument("--train_attr", default="no", choices=["yes", "no"])
    ap.add_argument("--image_arch", default="resnet50")
    ap.add_argument("--stage1_ckpt", required=True, help="Path to stage1 model.pkl")
    ap.add_argument("--out", required=True, help="Output .pt path")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=1)
    args = ap.parse_args()

    # hparams only needed to build dataset + featurizer consistently
    hparams = hparams_registry.default_hparams("ERM", args.dataset)
    hparams.update({"image_arch": args.image_arch})
    hparams.setdefault("pretrained", True)
    hparams.setdefault("last_layer_dropout", 0.0)


    dset_cls = vars(datasets)[args.dataset]
    dset = dset_cls(args.data_dir, args.split, hparams, train_attr=args.train_attr)
    loader = torch.utils.data.DataLoader(
        dset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build featurizer
    featurizer = networks.Featurizer(dset.data_type, dset.INPUT_SHAPE, hparams).to(device)
    featurizer.eval()

    # Load stage1 checkpoint, but only featurizer weights (strip classifier)
    ckpt = torch.load(args.stage1_ckpt, map_location="cpu", weights_only=False)
    sd = ckpt["model_dict"] if isinstance(ckpt, dict) and "model_dict" in ckpt else ckpt
    from collections import OrderedDict
    new_sd = OrderedDict()
    for k, v in sd.items():
        if "classifier" in k or "network.1." in k:
            continue
        new_sd[k] = v
    featurizer.load_state_dict(new_sd, strict=False)

    # Cache features in dataset index order (by i)
    N = len(dset)
    d = featurizer.n_outputs
    feats = torch.empty((N, d), dtype=torch.float32)

    for i, x, y, a in loader:
        x = x.to(device, non_blocking=True)
        f = featurizer(x).detach().cpu().float()  # [B, d]
        feats.index_copy_(0, i.long(), f)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"feats": feats}, out_path)
    print(f"Saved feats: {out_path}  shape={tuple(feats.shape)}")

if __name__ == "__main__":
    main()