import numpy as np
from crx.utils import misc


def _hparams(algorithm, dataset, random_seed):
    """
    Global registry of hyperparams. Each entry is a (default, random) tuple.
    """
    hparams = {}

    def _hparam(name, default_val, random_val_fn):
        assert name not in hparams
        r = np.random.RandomState(misc.seed_hash(random_seed, name))
        hparams[name] = (default_val, random_val_fn(r))

    # Shared
    _hparam("resnet18", False, lambda r: False)
    _hparam("nonlinear_classifier", False, lambda r: False)


    # sampling
    if algorithm == "CRT":
        _hparam("group_balanced", True, lambda r: True)
    else:
        _hparam("group_balanced", False, lambda r: False)

    _hparam("lr", 1e-3, lambda r: 10 ** r.uniform(-4, -2))
    _hparam("weight_decay", 1e-4, lambda r: 10 ** r.uniform(-6, -3))
    _hparam('batch_size', 108, lambda r: int(2**r.uniform(6, 6.75)))

    # stage-1 checkpoint filename (used by 2-stage methods)
    if algorithm in ["CRT", "DFR", "CRX"]:
        _hparam("stage1_model", "model.pkl", lambda r: "model.pkl")
        # If you truly need these as knobs, keep; otherwise drop.
        _hparam("stage1_hparams_seed", 0, lambda r: 0)
        _hparam("stage1_seed", 0, lambda r: 0)

    # -------------------------
    # Algorithm-specific knobs
    # -------------------------
    if algorithm == "GroupDRO":
        _hparam("groupdro_eta", 1e-2, lambda r: 10 ** r.uniform(-3, -1))

    elif algorithm == "DFR":
        _hparam("dfr_reg", 0.1, lambda r: 10 ** r.uniform(-2, 0.5))

    elif algorithm == 'JTT':
        _hparam('first_stage_step_frac', 0.5, lambda r: r.uniform(0.2, 0.8))
        _hparam('jtt_lambda', 10, lambda r: 10**r.uniform(0, 2.5))

    elif algorithm == "CRX":
        _hparam("cr_feat_dropout", 0.0, lambda r: float(r.choice([0.0, 0.2, 0.5])))
        _hparam("cr_concept_dropout", 0.0, lambda r: float(r.choice([0.0, 0.1, 0.2])))
        _hparam("cr_resid_dropout", 0.0, lambda r: float(r.choice([0.0, 0.1, 0.2])))

        _hparam("cr_reg", 0.0, lambda r: float(10**r.uniform(-8, -3)))

        _hparam("cr_concept_block_channels",
                [],
                lambda r: r.choice([[], ["environment_context", "imaging_artifacts"]]))

        _hparam("cr_block_gates", False, lambda r: bool(r.choice([False, True])))
        _hparam("cr_gate_reg", 0.0, lambda r: float(10**r.uniform(-6, -2)))




    return hparams


def default_hparams(algorithm, dataset):
    return {k: v[0] for k, v in _hparams(algorithm, dataset, 0).items()}


def random_hparams(algorithm, dataset, seed):
    return {k: v[1] for k, v in _hparams(algorithm, dataset, seed).items()}
