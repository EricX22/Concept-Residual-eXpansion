import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
from transformers import get_scheduler
import json


from crx.models import networks
from crx.learning.optimizers import get_optimizers
from crx.utils.misc import mixup_data


ALGORITHMS = [
    'ERM',
    # subgroup methods
    'GroupDRO',
    'JTT',
    'DFR',
    'CRX',
    # imbalanced learning methods
    'CRT',
]


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a subgroup robustness algorithm.
    Subclasses should implement the following:
    - _init_model()
    - _compute_loss()
    - update()
    - return_feats()
    - predict()
    """
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super(Algorithm, self).__init__()
        self.hparams = hparams
        self.data_type = data_type
        self.num_classes = num_classes
        self.num_attributes = num_attributes
        self.num_examples = num_examples

    def _init_model(self):
        raise NotImplementedError

    def _compute_loss(self, i, x, y, a, step):
        raise NotImplementedError

    def update(self, minibatch, step):
        """Perform one update step."""
        raise NotImplementedError

    def return_feats(self, x):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

    def return_groups(self, y, a):
        """Given a list of (y, a) tuples, return indexes of samples belonging to each subgroup"""
        idx_g, idx_samples = [], []
        all_g = y * self.num_attributes + a

        for g in all_g.unique():
            idx_g.append(g)
            idx_samples.append(all_g == g)

        return zip(idx_g, idx_samples)

    @staticmethod
    def return_attributes(all_a):
        """Given a list of attributes, return indexes of samples belonging to each attribute"""
        idx_a, idx_samples = [], []

        for a in all_a.unique():
            idx_a.append(a)
            idx_samples.append(all_a == a)

        return zip(idx_a, idx_samples)


class ERM(Algorithm):
    """Empirical Risk Minimization (ERM)"""
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super(ERM, self).__init__(
            data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes)

        self.featurizer = networks.Featurizer(data_type, input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier']
        )
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self._init_model()

    def _init_model(self):
        self.clip_grad = (self.data_type == "text" and self.hparams["optimizer"] == "adamw")

        if self.data_type in ["images", "tabular"]:
            self.optimizer = get_optimizers['sgd'](
                self.network,
                self.hparams['lr'],
                self.hparams['weight_decay']
            )
            self.lr_scheduler = None
            self.loss = torch.nn.CrossEntropyLoss(reduction="none")
        elif self.data_type == "text":
            self.network.zero_grad()
            self.optimizer = get_optimizers[self.hparams["optimizer"]](
                self.network,
                self.hparams['lr'],
                self.hparams['weight_decay']
            )
            self.lr_scheduler = get_scheduler(
                "linear",
                optimizer=self.optimizer,
                num_warmup_steps=0,
                num_training_steps=self.hparams["steps"]
            )
            self.loss = torch.nn.CrossEntropyLoss(reduction="none")
        else:
            raise NotImplementedError(f"{self.data_type} not supported.")

    def _compute_loss(self, i, x, y, a, step):
        return self.loss(self.predict(x), y).mean()

    def update(self, minibatch, step):
        all_i, all_x, all_y, all_a = minibatch
        loss = self._compute_loss(all_i, all_x, all_y, all_a, step)

        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        if self.data_type == "text":
            self.network.zero_grad()

        return {'loss': loss.item()}

    def return_feats(self, x):
        return self.featurizer(x)

    def predict(self, x):
        return self.network(x)


class GroupDRO(ERM):
    """
    Group DRO minimizes the error at the worst group [https://arxiv.org/pdf/1911.08731.pdf]
    """
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super(GroupDRO, self).__init__(
            data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes)
        self.register_buffer(
            "q", torch.ones(self.num_classes * self.num_attributes).cuda())

    def _compute_loss(self, i, x, y, a, step):
        losses = self.loss(self.predict(x), y)

        for idx_g, idx_samples in self.return_groups(y, a):
            self.q[idx_g] *= (self.hparams["groupdro_eta"] * losses[idx_samples].mean()).exp().item()

        self.q /= self.q.sum()

        loss_value = 0
        for idx_g, idx_samples in self.return_groups(y, a):
            loss_value += self.q[idx_g] * losses[idx_samples].mean()

        return loss_value


class CRT(ERM):
    """Classifier re-training with balanced sampling during the second earning stage"""
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super(CRT, self).__init__(data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes)
        # fix stage 1 trained featurizer
        for name, param in self.featurizer.named_parameters():
            param.requires_grad = False
        # only optimize the classifier
        if self.data_type in ["images", "tabular"]:
            self.optimizer = get_optimizers['sgd'](
                self.classifier,
                self.hparams['lr'],
                self.hparams['weight_decay']
            )
            self.lr_scheduler = None
        elif self.data_type == "text":
            self.network.zero_grad()
            self.optimizer = get_optimizers[self.hparams["optimizer"]](
                self.classifier,
                self.hparams['lr'],
                self.hparams['weight_decay']
            )
            self.lr_scheduler = get_scheduler(
                "linear",
                optimizer=self.optimizer,
                num_warmup_steps=0,
                num_training_steps=self.hparams["steps"]
            )
        else:
            raise NotImplementedError(f"{self.data_type} not supported.")

class DFR(ERM):
    """
    Classifier re-training with sub-sampled, group-balanced, held-out(validation) data and l1 regularization.
    Note that when attribute is unavailable in validation data, group-balanced reduces to class-balanced.
    https://openreview.net/pdf?id=Zb6c8A-Fghk
    """
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super(DFR, self).__init__(data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes)
        # fix stage 1 trained featurizer
        for name, param in self.featurizer.named_parameters():
            param.requires_grad = False
        # only optimize the classifier
        if self.data_type in ["images", "tabular"]:
            self.optimizer = get_optimizers['sgd'](
                self.classifier,
                self.hparams['lr'],
                0.
            )
            self.lr_scheduler = None
        elif self.data_type == "text":
            self.network.zero_grad()
            self.optimizer = get_optimizers[self.hparams["optimizer"]](
                self.classifier,
                self.hparams['lr'],
                0.
            )
            self.lr_scheduler = get_scheduler(
                "linear",
                optimizer=self.optimizer,
                num_warmup_steps=0,
                num_training_steps=self.hparams["steps"]
            )
        else:
            raise NotImplementedError(f"{self.data_type} not supported.")

    def _compute_loss(self, i, x, y, a, step):
        return self.loss(self.predict(x), y).mean() + self.hparams['dfr_reg'] * torch.norm(self.classifier.weight, 1)


class CRX(ERM):
    def __init__(self, data_type, input_shape, num_classes, num_attributes,
                 num_examples, hparams, grp_sizes=None):
        super().__init__(data_type, input_shape, num_classes, num_attributes,
                         num_examples, hparams, grp_sizes)

        # ---- Freeze stage-1 featurizer
        for p in self.featurizer.parameters():
            p.requires_grad = False

        # ---- Split control (used for caches);
        self._active_split = self.hparams.get("cr_active_split", "va")
        self._index_map_by_split = {}  # optional tensor maps: map[global_i] -> local_i

        # ---- Flags + dims
        self.use_concepts = bool(self.hparams.get("cr_use_concepts", True))
        self.use_resid    = bool(self.hparams.get("cr_use_resid", True))
        self.concept_dim  = int(self.hparams.get("cr_concept_dim", 0))
        self.resid_dim    = int(self.hparams.get("cr_resid_dim", 64))

        # ---- Dropout knobs
        self.cr_feat_dropout    = float(self.hparams.get("cr_feat_dropout", 0.0))
        self.cr_concept_dropout = float(self.hparams.get("cr_concept_dropout", 0.0))
        self.cr_resid_dropout   = float(self.hparams.get("cr_resid_dropout", 0.0))

        # ---- Block-level gates
        self.cr_block_gates = bool(self.hparams.get("cr_block_gates", False))
        if self.cr_block_gates:
            init = float(self.hparams.get("cr_gate_init", 0.0))  # sigmoid(init)
            self.gate_alpha_logit = torch.nn.Parameter(torch.tensor(init))  # concepts
            self.gate_beta_logit  = torch.nn.Parameter(torch.tensor(init))  # residuals

        # ---- Load caches + meta
        self._concept_cache_by_split = {}
        self._resid_cache_by_split   = {}
        self._concept_use_mask       = None
        self.concept_dim_used        = 0

        if self.use_concepts:
            self._load_concept_caches()
            self._load_concept_meta_and_mask()

        if self.use_resid:
            self._load_resid_caches()

        # ---- Feature mode (validate against flags)
        self.mode = self.hparams.get("cr_feature_mode", "concat_plus_resid")
        self._validate_feature_mode()

        # ---- Build classifier input dim
        in_dim = self._infer_in_dim(num_classes)

        self.classifier = networks.Classifier(
            in_dim, num_classes, self.hparams["nonlinear_classifier"]
        )

        # ---- Optimizer (classifier + optional gates only)
        params = list(self.classifier.parameters())
        if self.cr_block_gates:
            params += [self.gate_alpha_logit, self.gate_beta_logit]

        if self.data_type in ["images", "tabular"]:
            self.optimizer = torch.optim.SGD(
                params,
                lr=self.hparams["lr"],
                weight_decay=0.0,
                momentum=self.hparams.get("momentum", 0.9),
                nesterov=self.hparams.get("nesterov", False),
            )
            self.lr_scheduler = None
        elif self.data_type == "text":
            self.classifier.zero_grad()
            self.optimizer = get_optimizers[self.hparams["optimizer"]](
                self.classifier, self.hparams["lr"], 0.0
            )
            self.lr_scheduler = get_scheduler(
                "linear",
                optimizer=self.optimizer,
                num_warmup_steps=0,
                num_training_steps=self.hparams["steps"],
            )
        else:
            raise NotImplementedError(f"{self.data_type} not supported.")

    # -------------------------
    # Loading helpers
    # -------------------------
    def _load_concept_caches(self):
        for split_key in ["va", "te", "tr"]:
            p = self.hparams.get(f"cr_concept_path_{split_key}", None)
            if p is None:
                continue
            cache = torch.load(p, map_location="cpu")
            if isinstance(cache, dict) and "concepts" in cache:
                cache = cache["concepts"]
            assert torch.is_tensor(cache) and cache.dim() == 2
            self._concept_cache_by_split[split_key] = cache.contiguous()

        if not self._concept_cache_by_split:
            p = self.hparams.get("cr_concept_path", None)
            assert p is not None, "cr_use_concepts=True requires cr_concept_path or cr_concept_path_<split>"
            cache = torch.load(p, map_location="cpu")
            if isinstance(cache, dict) and "concepts" in cache:
                cache = cache["concepts"]
            assert torch.is_tensor(cache) and cache.dim() == 2
            self._concept_cache_by_split["va"] = cache.contiguous()

        any_cache = next(iter(self._concept_cache_by_split.values()))
        cache_K = int(any_cache.shape[1])
        if self.concept_dim <= 0:
            self.concept_dim = cache_K
        else:
            assert self.concept_dim == cache_K, f"cr_concept_dim={self.concept_dim} but cache has K={cache_K}"

    def _load_concept_meta_and_mask(self):
        meta_path = self.hparams.get("cr_concept_meta_path", None)
        assert meta_path is not None, "cr_use_concepts=True requires cr_concept_meta_path"
        with open(meta_path, "r") as f:
            meta = json.load(f)

        block = set(self.hparams.get("cr_concept_block_channels", []))

        channel = meta.get("channel", None)
        if channel is None:
            channel_mask = torch.ones(self.concept_dim, dtype=torch.bool)
        else:
            channel_mask = torch.tensor([c not in block for c in channel], dtype=torch.bool)

        use_mask = meta.get("use_in_training", None)
        if use_mask is None:
            use_mask = torch.ones(self.concept_dim, dtype=torch.bool)
        else:
            use_mask = torch.as_tensor(use_mask, dtype=torch.bool)

        self._concept_use_mask = use_mask & channel_mask
        self.concept_dim_used = int(self._concept_use_mask.sum().item())

    def _load_resid_caches(self):
        for split_key in ["va", "te", "tr"]:
            p = self.hparams.get(f"cr_resid_path_{split_key}", None)
            if p is None:
                continue
            cache = torch.load(p, map_location="cpu")
            if isinstance(cache, dict) and "resid" in cache:
                cache = cache["resid"]
            assert torch.is_tensor(cache) and cache.dim() == 2
            self._resid_cache_by_split[split_key] = cache.contiguous()

        if not self._resid_cache_by_split:
            p = self.hparams.get("cr_resid_path", None)
            assert p is not None, "cr_use_resid=True requires cr_resid_path or cr_resid_path_<split>"
            cache = torch.load(p, map_location="cpu")
            if isinstance(cache, dict) and "resid" in cache:
                cache = cache["resid"]
            assert torch.is_tensor(cache) and cache.dim() == 2
            self._resid_cache_by_split["va"] = cache.contiguous()

        any_r = next(iter(self._resid_cache_by_split.values()))
        cache_R = int(any_r.shape[1])
        if self.resid_dim <= 0:
            self.resid_dim = cache_R
        else:
            assert self.resid_dim == cache_R, f"cr_resid_dim={self.resid_dim} but cache has R={cache_R}"

    # -------------------------
    # Mode + dim helpers
    # -------------------------
    def _validate_feature_mode(self):
        valid = {"concat", "concat_plus_resid", "concept_only", "resid_only", "resid_plus_concept"}
        if self.mode not in valid:
            raise ValueError(f"Unknown cr_feature_mode={self.mode}; valid={sorted(valid)}")

        if self.mode in {"concat", "concept_only", "resid_plus_concept"} and not self.use_concepts:
            raise ValueError(f"cr_feature_mode={self.mode} requires cr_use_concepts=True")
        if self.mode in {"concat_plus_resid", "resid_only", "resid_plus_concept"} and not self.use_resid:
            # concat_plus_resid can still be used with concepts only, but name suggests resid is intended;
            # make it strict to avoid silent config mistakes
            raise ValueError(f"cr_feature_mode={self.mode} requires cr_use_resid=True")

    def _infer_in_dim(self, num_classes):
        fdim = self.featurizer.n_outputs
        if self.mode == "concat":
            return fdim + self.concept_dim_used
        if self.mode == "concat_plus_resid":
            return fdim + (self.concept_dim_used if self.use_concepts else 0) + self.resid_dim
        if self.mode == "concept_only":
            return self.concept_dim_used
        if self.mode == "resid_only":
            return self.resid_dim
        if self.mode == "resid_plus_concept":
            return self.resid_dim + (self.concept_dim_used if self.use_concepts else 0)
        raise RuntimeError("unreachable")

    # -------------------------
    # Split/index helpers
    # -------------------------
    def set_active_split(self, split: str):
        self._active_split = split

    def _maybe_remap_idx(self, idx_cpu: torch.Tensor) -> torch.Tensor:
        m = self._index_map_by_split.get(self._active_split, None)
        if m is None:
            return idx_cpu
        if not torch.is_tensor(m):
            raise TypeError(f"[CRX] index map for split={self._active_split} must be a tensor")
        local = m.index_select(0, idx_cpu)
        if (local < 0).any():
            bad = idx_cpu[local < 0][:10].tolist()
            raise IndexError(f"[CRX] indices not present in split={self._active_split}: examples={bad}")
        return local.long()

    def _raise_cache_oob(self, kind: str, split: str, idx: torch.Tensor, cache_rows: int):
        idx_min = int(idx.min()) if idx.numel() else -1
        idx_max = int(idx.max()) if idx.numel() else -1
        raise IndexError(
            f"{kind} cache too small for split={split}: idx range=[{idx_min},{idx_max}], "
            f"cache rows={cache_rows}. Likely indexing mismatch (global vs split-local)."
        )

    def _get_concepts(self, i, device, dtype):
        idx = self._maybe_remap_idx(i.detach().to("cpu").long())
        cache = self._concept_cache_by_split.get(self._active_split, None)
        if cache is None:
            raise KeyError(f"No concept cache for active split={self._active_split}. Have={list(self._concept_cache_by_split)}")
        if int(idx.max()) >= cache.shape[0]:
            self._raise_cache_oob("Concept", self._active_split, idx, cache.shape[0])
        c = cache.index_select(0, idx)
        if self._concept_use_mask is not None:
            c = c[:, self._concept_use_mask]
        return c.to(device=device, dtype=dtype)

    def _get_resid(self, i, device, dtype):
        idx = self._maybe_remap_idx(i.detach().to("cpu").long())
        cache = self._resid_cache_by_split.get(self._active_split, None)
        if cache is None:
            raise KeyError(f"No resid cache for active split={self._active_split}. Have={list(self._resid_cache_by_split)}")
        if int(idx.max()) >= cache.shape[0]:
            self._raise_cache_oob("Resid", self._active_split, idx, cache.shape[0])
        r = cache.index_select(0, idx)
        return r.to(device=device, dtype=dtype)

    # -------------------------
    # API
    # -------------------------
    def return_feats(self, x):
        return self.featurizer(x)

    def predict(self, x, i=None):
        f = self.featurizer(x)
        if self.cr_feat_dropout > 0:
            f = F.dropout(f, p=self.cr_feat_dropout, training=self.training)

        dtype = f.dtype
        device = f.device

        c = r = None
        if self.use_concepts:
            assert i is not None, "CRX requires dataset indices i when using concepts"
            c = self._get_concepts(i, device, dtype)
            scale = float(self.hparams.get("cr_concept_scale", 1.0))
            if scale != 1.0:
                c = c * scale
            if self.cr_concept_dropout > 0:
                c = F.dropout(c, p=self.cr_concept_dropout, training=self.training)

        if self.use_resid:
            assert i is not None, "CRX requires dataset indices i when using residuals"
            r = self._get_resid(i, device, dtype)
            if self.cr_resid_dropout > 0:
                r = F.dropout(r, p=self.cr_resid_dropout, training=self.training)

        if self.cr_block_gates:
            if c is not None:
                c = c * torch.sigmoid(self.gate_alpha_logit).to(device=device, dtype=dtype)
            if r is not None:
                r = r * torch.sigmoid(self.gate_beta_logit).to(device=device, dtype=dtype)

        if self.mode == "concat":
            z = torch.cat([f, c], dim=1)
        elif self.mode == "concat_plus_resid":
            parts = [f]
            if c is not None:
                parts.append(c)
            parts.append(r)  # required by validation
            z = torch.cat(parts, dim=1)
        elif self.mode == "concept_only":
            z = c
        elif self.mode == "resid_only":
            z = r
        elif self.mode == "resid_plus_concept":
            parts = [r]
            if c is not None:
                parts.append(c)
            z = torch.cat(parts, dim=1)
        else:
            raise RuntimeError("unreachable")

        return self.classifier(z)

    def _compute_loss(self, i, x, y, a, step):
        logits = self.predict(x, i=i)
        ce = self.loss(logits, y).mean()
        reg = float(self.hparams.get("cr_reg", 0.0)) * torch.norm(self.classifier.weight, 1)

        gate_reg = 0.0
        if self.cr_block_gates:
            lam = float(self.hparams.get("cr_gate_reg", 0.0))
            terms = []
            if self.use_concepts:
                terms.append(torch.sigmoid(self.gate_alpha_logit).abs())
            if self.use_resid:
                terms.append(torch.sigmoid(self.gate_beta_logit).abs())
            if terms:
                gate_reg = lam * sum(terms)

        return ce + reg + gate_reg

    def update(self, minibatch, step):
        all_i, all_x, all_y, all_a = minibatch
        loss = self._compute_loss(all_i, all_x, all_y, all_a, step)

        self.optimizer.zero_grad()
        loss.backward()
        if getattr(self, "clip_grad", False):
            torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), 1.0)
        self.optimizer.step()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return {"loss": loss.item()}


class AbstractTwoStage(Algorithm):
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super().__init__(
            data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes)

        self.stage1_model = ERM(data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes)
        self.first_stage_step_frac = hparams['first_stage_step_frac']
        self.switch_step = int(self.first_stage_step_frac * hparams['steps'])
        self.cur_model = self.stage1_model

        self.stage2_model = None    # implement in child classes

    def update(self, minibatch, step):
        all_i, all_x, all_y, all_a = minibatch

        if step < self.switch_step:
            self.cur_model = self.stage1_model
            self.cur_model.train()
            loss = self.stage1_model._compute_loss(all_i, all_x, all_y, all_a, step)
        else:
            self.cur_model = self.stage2_model
            self.cur_model.train()
            self.stage1_model.eval()
            loss = self.stage2_model._compute_loss(all_i, all_x, all_y, all_a, step, self.stage1_model)
        
        self.cur_model.optimizer.zero_grad()
        loss.backward()
        if self.cur_model.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.cur_model.network.parameters(), 1.0)
        self.cur_model.optimizer.step()

        if self.cur_model.lr_scheduler is not None:
            self.cur_model.lr_scheduler.step()

        if self.data_type == "text":
            self.cur_model.network.zero_grad()

        return {'loss': loss.item()}

    def return_feats(self, x):
        return self.cur_model.featurizer(x)
    
    def predict(self, x):
        return self.cur_model.network(x)


class JTT_Stage2(ERM): 
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super().__init__(
            data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes)

    def _compute_loss(self, i, x, y, a, step, stage1_model):
        with torch.no_grad():
            predictions = stage1_model.predict(x)

        if predictions.squeeze().ndim == 1:
            wrong_predictions = (predictions > 0).detach().ne(y).float()
        else:
            wrong_predictions = predictions.argmax(1).detach().ne(y).float()

        weights = torch.ones(wrong_predictions.shape).to(x.device).float()
        weights[wrong_predictions == 1] = self.hparams["jtt_lambda"]

        return (self.loss(self.predict(x), y) * weights).mean()


class JTT(AbstractTwoStage):
    """
    Just-train-twice (JTT) [https://arxiv.org/pdf/2107.09044.pdf]
    """
    def __init__(self, data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes=None):
        super().__init__(data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes)
        self.stage2_model = JTT_Stage2(
            data_type, input_shape, num_classes, num_attributes, num_examples, hparams, grp_sizes)