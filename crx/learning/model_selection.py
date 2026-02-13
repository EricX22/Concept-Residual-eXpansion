class SelectionMethod:
    """Abstract base class."""
    name = "abstract"

    def __init__(self):
        raise TypeError

    @classmethod
    def run_acc(cls, run_records):
        raise NotImplementedError

    @classmethod
    def hparams_accs(cls, records):
        # group by hparams_seed, compute best step per run, then sort by val_acc desc
        return (
            records.group("args.hparams_seed")
            .map(lambda _, run_records: (cls.run_acc(run_records), run_records))
            .filter(lambda x: x[0] is not None)
            .sorted(key=lambda x: x[0]["val_acc"])[::-1]
        )

    @classmethod
    def sweep_acc(cls, records):
        accs = cls.hparams_accs(records)
        return accs[0][0]["test_acc"] if len(accs) else None

    @classmethod
    def sweep_acc_worst(cls, records):
        accs = cls.hparams_accs(records)
        return accs[0][0]["test_acc_worst"] if len(accs) else None


class ValMeanAcc(SelectionMethod):
    """
    Select checkpoint by validation overall accuracy.
    (No group-label-based selection.)
    """
    name = "validation set mean accuracy"

    @classmethod
    def _step_acc(cls, record):
        # Assumes records contain 'va' and 'te' entries like train.py writes
        return {
            "val_acc": record["va"]["overall"]["accuracy"],
            "test_acc": record["te"]["overall"]["accuracy"],
            "test_acc_worst": record["te"]["min_group"]["accuracy"],
        }

    @classmethod
    def run_acc(cls, run_records):
        if not len(run_records):
            return None
        # pick the checkpoint with best val_acc
        return run_records.map(cls._step_acc).argmax("val_acc")
