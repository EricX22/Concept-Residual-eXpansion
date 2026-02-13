import numpy as np
import torch


def predict_on_set(algorithm, loader, device, split_name=None):
    # Set split for cache-based methods (CRX)
    if hasattr(algorithm, "set_active_split"):
        if split_name is None:
            split_name = getattr(loader.dataset, "split", None)
        if split_name is not None:
            algorithm.set_active_split(split_name)

    ys, atts, preds = [], [], []

    algorithm.eval()
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 4:
                i, x, y, a = batch
                i_dev = i.to(device)
            elif len(batch) == 3:
                x, y, a = batch
                i_dev = None
            else:
                raise ValueError(f"Unexpected batch length {len(batch)} (expected 3 or 4)")

            x_dev = x.to(device)

            # Get logits
            try:
                logits = algorithm.predict(x_dev, i=i_dev) if i_dev is not None else algorithm.predict(x_dev)
            except TypeError:
                logits = algorithm.predict(x_dev)

            # Convert logits -> predicted class
            if logits.ndim == 1 or logits.shape[-1] == 1:
                # binary logit
                yhat = (torch.sigmoid(logits.view(-1)) >= 0.5).long()
            else:
                yhat = logits.argmax(dim=-1)

            ys.append(y.detach().cpu().numpy())
            atts.append(a.detach().cpu().numpy())
            preds.append(yhat.detach().cpu().numpy())

    y = np.concatenate(ys, axis=0)
    a = np.concatenate(atts, axis=0)
    yhat = np.concatenate(preds, axis=0)
    return y, a, yhat


def eval_metrics(algorithm, loader, device, split_name=None):
    y, a, yhat = predict_on_set(algorithm, loader, device, split_name=split_name)

    # group id + per-group accuracy
    num_attributes = int(getattr(loader.dataset, "num_attributes", int(a.max()) + 1))
    g = y * num_attributes + a

    per_group = {}
    for gid in np.unique(g):
        mask = (g == gid)
        acc = float((yhat[mask] == y[mask]).mean()) if mask.any() else float("nan")
        per_group[int(gid)] = {"accuracy": acc, "n_samples": int(mask.sum())}

    overall_acc = float((yhat == y).mean()) if len(y) else float("nan")
    worst_acc = float(np.min([per_group[k]["accuracy"] for k in per_group])) if per_group else float("nan")

    return {
        "overall": {"accuracy": overall_acc, "n_samples": int(len(y))},
        "per_group": per_group,
        "min_group": {"accuracy": worst_acc},
    }
