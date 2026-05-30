# CRX: Concept-Residual Representation Expansion for Robustness to Spurious Correlations

[![DOI](https://zenodo.org/badge/1153242215.svg)](https://doi.org/10.5281/zenodo.20467988)

This repository provides a reproducible implementation of **CRX**, a two-stage method for improving worst-group accuracy under spurious correlations using:

* Frozen stage-1 ERM features
* CLIP concept logits
* Residual PCA features (stage-1 features with the concept-predictable component removed)

Supported datasets:

* Waterbirds
* CelebA
* CheXpert (No Finding)

---

## Installation

### Option A: Conda

```bash

conda env create -f environment.yml
conda activate crx

```

### Option B: pip

```bash

pip install -r requirements.txt

```

---

## Data

Place datasets under a single root directory (referred to below as `DATA_DIR`).

### Waterbirds

Download the dataset and generate the metadata CSV:

```bash
python -m crx.scripts.download waterbirds --data_path "$DATA_DIR" --download
```

This fetches `waterbird_complete95_forest2water2.tar.gz` from the [Stanford NLP group](https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz) and writes `metadata_waterbirds.csv`.

### CelebA

```bash
python -m crx.scripts.download celeba --data_path "$DATA_DIR" --download
```

This downloads the aligned images and attribute files from Google Drive and writes `metadata_celeba.csv`.

### CheXpert (No Finding)

CheXpert requires a free registration at [stanfordmlgroup.github.io/competitions/chexpert](https://stanfordmlgroup.github.io/competitions/chexpert). Download the dataset and place it under `DATA_DIR/chexpert/`.

The split/group metadata file (`metadata_no_finding.csv`) follows the [SubpopBench](https://github.com/YyzHarry/SubpopBench) format. Place it at:

```
DATA_DIR/chexpert/subpop_bench_meta/metadata_no_finding.csv
```

---

## Quickstart (Waterbirds)

Set paths first:

```bash

DATA_DIR=/path/to/data
OUT_DIR=/path/to/output
STAGE1_FOLDER=vanilla_attrNo_crx
ART_DIR=/path/to/artifacts/wb_crx_v1

```

### 1) Train stage-1 ERM checkpoint

```bash

python -m crx.train 
--dataset Waterbirds 
--algorithm ERM 
--train_attr no 
--data_dir "$DATA_DIR" 
--output_dir "$OUT_DIR" 
--output_folder_name "$STAGE1_FOLDER" 
--seed 0 
--hparams_seed 0

```

This produces a stage-1 checkpoint at:

* `OUT_DIR/STAGE1_FOLDER/Waterbirds_ERM_hparams0_seed0/model.pkl`

---

### 2) Build CRX artifacts

(Concept caches + stage-1 feature caches + residual PCA)

```bash

python -m crx.scripts.setup_crx_artifacts 
--dataset Waterbirds 
--data-dir "$DATA_DIR" 
--stage1-folder "$OUT_DIR/$STAGE1_FOLDER" 
--artifact-dir "$ART_DIR" 
--resid-dim 64

```

This generates the following in `ART_DIR`:

* `meta.json`
* `concepts_tr.pt`, `concepts_va.pt`, `concepts_te.pt`
* `stage1_feats_va.pt`, `stage1_feats_te.pt`
* `resid_va.pt`, `resid_te.pt`
* `resid_meta.json`

---

### 3) Train CRX (stage-2 on validation split)

```bash

python -m crx.scripts.run_crx 
--dataset Waterbirds 
--data-dir "$DATA_DIR" 
--output-dir "$OUT_DIR" 
--output-folder-name wb_crx_run 
--stage1-folder "$STAGE1_FOLDER" 
--artifact-dir "$ART_DIR"

```

Training logs and metrics are written to:

* `OUT_DIR/<output-folder-name>/Waterbirds_CRX_hparams<seed>_seed<seed>/`

Key outputs:

* `results.json` — checkpoint-level metrics
* `final_results.pkl` — final evaluation metrics
* `model.pkl` — trained CRX model

---

## Quickstart (CelebA)

Repeat the same three steps, replacing:

* `--dataset Waterbirds` → `--dataset CelebA`
* Use a new artifact directory (e.g., `celeba_crx_v1`)

---

## Quickstart (CheXpert)

Repeat the same three steps, replacing:

* `--dataset Waterbirds` → `--dataset CheXpertNoFinding`
* Use a new artifact directory (e.g., `chexpert_crx_v1`)

The concept bank for CheXpert uses `--modality medical_xray` automatically when invoked via `setup_crx_artifacts`.

---

## Outputs and Metrics

During training, the system reports:

* Average accuracy (`*_avg_acc`)
* Worst-group accuracy (`*_worst_acc`)

These are written per checkpoint to `results.json` and summarized in `final_results.pkl`.

---

## Reproducing paper runs (best hyperparameters)

We provide the hyperparameters used for the best runs reported in the paper:

- `configs/best_hparams_waterbirds.json`
- `configs/best_hparams_celeba.json`
- `configs/best_hparams_chexpert.json`

These JSON files include only *algorithm hyperparameters* (e.g., lr, weight decay, dropouts, gate regularization).  
They do **not** include dataset/artifact paths; those are supplied via the artifact setup step or the sweep script.

### Example: run best CRX

Assuming you already have:
- a stage-1 ERM checkpoint under `OUT_DIR/STAGE1_FOLDER/Waterbirds_ERM_hparams0_seed0/model.pkl`
- CRX artifacts built under `ART_DIR` (see Quickstart)

```bash
python -m crx.train \
  --dataset Waterbirds \
  --algorithm CRX \
  --train_attr no \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUT_DIR" \
  --output_folder_name wb_crx_best \
  --stage1_folder "$STAGE1_FOLDER" \
  --stage1_algo ERM \
  --hparams_seed 0 \
  --seed 0 \
  --hparams "$(cat configs/best_hparams_waterbirds.json)"
```

```bash
python -m crx.train \
  --dataset CelebA \
  --algorithm CRX \
  --train_attr no \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUT_DIR" \
  --output_folder_name celeba_crx_best \
  --stage1_folder "$STAGE1_FOLDER" \
  --stage1_algo ERM \
  --hparams_seed 0 \
  --seed 0 \
  --hparams "$(cat configs/best_hparams_celeba.json)"
```

```bash
python -m crx.train \
  --dataset CheXpertNoFinding \
  --algorithm CRX \
  --train_attr no \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUT_DIR" \
  --output_folder_name chexpert_crx_best \
  --stage1_folder "$STAGE1_FOLDER" \
  --stage1_algo ERM \
  --hparams_seed 0 \
  --seed 0 \
  --hparams "$(cat configs/best_hparams_chexpert.json)"
```

## Hyperparameter Sweeps

We provide a Slurm sweep launcher similar to the one used to produce paper results:
```bash
bash scripts/run_sweep_slurm.sh
```

## Monitor and Collect Results

The following command provides a summary of an ongoing/completed sweep:

```bash
python -m crx.scripts.summarize_progress --root /path/to/output/<TAG>
```


## Acknowledgements

This repo is adapted from the structure of SubpopBench.
Concept scoring is performed using OpenCLIP.
