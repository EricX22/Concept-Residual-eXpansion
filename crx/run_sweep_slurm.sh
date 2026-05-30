#!/bin/bash
set -euo pipefail

# ============================================================
# CRX minimal repo: Slurm sweep launcher
#
# What it does:
#  1) (optional) train stage-1 ERM if missing
#  2) (optional) build CRX artifacts once
#  3) submit sweeps for a list of algorithms / hparams / seeds
#  4) throttle max in-flight jobs
#
# You MUST edit the USER CONFIG section below.
# ============================================================


# --------------------------- USER CONFIG ---------------------------
# Repo + data/output locations
REPO_DIR="/path/to/this/repo"              # repo root (contains crx/ and scripts/)
DATA_DIR="/path/to/data"                   # dataset root
OUT_DIR="/path/to/output"                  # where training runs are written
ART_DIR="/path/to/artifacts/wb_crx_v1"      # where CRX artifacts are written (recommended)

# Experiment selection
DATASET="Waterbirds"                       # Waterbirds, CelebA, or CheXpertNoFinding
TRAIN_ATTR="no"                            # yes/no (must match crx.train choices)

# Sweep tag (used for output folder name + Slurm job name)
TAG="crx_sweep_$(date +%Y%m%d_%H%M%S)"

# Algorithms to run (must match this repo’s supported algorithms)
# Typical paper set:
ALGS=(ERM GroupDRO JTT DFR CRT CRX)

# Hyperparameter seeds and random seeds
HP_SEEDS=$(seq 0 15)
SEEDS=(0)

# Stage-1 settings used by 2-stage methods (DFR/CRX, and CRT if applicable)
STAGE1_FOLDER="vanilla_attrNo"             # folder under OUT_DIR where ERM stage-1 lives
STAGE1_ALGO="ERM"
STAGE1_HPSEED=0
STAGE1_SEED=0
STAGE1_STEPS=5000                          # only used if stage-1 is missing and we auto-train it

# CRX artifact settings
RESID_DIM=64

# Throttle
MAX_IN_FLIGHT=8                            # max Slurm jobs with name TAG in queue/running

# Slurm resources (EDIT THESE for your cluster)
SBATCH_PART=""                             # e.g. "-p gpu" or "" if not needed
SBATCH_GRES=""                             # e.g. "--gres=gpu:1" or "" if not needed
SBATCH_CPU="-c 4"
SBATCH_MEM="--mem=24G"
SBATCH_TIME="-t 0-12:00:00"
SBATCH_ACCOUNT=""                          # e.g. "-A <acct>" or "" if not needed

# Environment activation (EDIT ONE of these)
# Option A: conda
USE_CONDA=1
CONDA_ENV_NAME="subpop_bench"

# Option B: module + venv (set USE_CONDA=0 and fill these)
MODULE_LOAD_CMD=""                         # e.g. "module load cuda/12.1"
VENV_ACTIVATE=""                           # e.g. "source /path/to/venv/bin/activate"
# ------------------------------------------------------------------


# ------------------------- derived paths --------------------------
STAGE1_RUN="${DATASET}_${STAGE1_ALGO}_hparams${STAGE1_HPSEED}_seed${STAGE1_SEED}"
STAGE1_CKPT="${OUT_DIR}/${STAGE1_FOLDER}/${STAGE1_RUN}/model.pkl"

SLURM_DIR="${REPO_DIR}/slurm"
OUT_TAG_DIR="${OUT_DIR}/${TAG}"
mkdir -p "${SLURM_DIR}" "${OUT_TAG_DIR}"
cd "${REPO_DIR}"
# ------------------------------------------------------------------


in_flight () { squeue -u "$USER" -h -n "${TAG}" | wc -l; }

wait_for_slot () {
  while [ "$(in_flight)" -ge "${MAX_IN_FLIGHT}" ]; do
    echo "[throttle] in-flight=$(in_flight) >= ${MAX_IN_FLIGHT}, sleeping 30s..."
    sleep 30
  done
}

run_dir () {
  local algo="$1" hp="$2" seed="$3"
  echo "${OUT_TAG_DIR}/${DATASET}_${algo}_hparams${hp}_seed${seed}"
}

is_done () {
  local d="$1"
  [ -f "${d}/done" ] || [ -f "${d}/final_results.pkl" ]
}

env_setup_snippet () {
  if [ "${USE_CONDA}" -eq 1 ]; then
    cat <<EOF
source "\$HOME/anaconda3/etc/profile.d/conda.sh" || true
conda activate "${CONDA_ENV_NAME}"
EOF
  else
    cat <<EOF
${MODULE_LOAD_CMD}
${VENV_ACTIVATE}
EOF
  fi
}

# ------------------------------------------------------------
# (1) Stage-1 bootstrap: train ERM if missing
# ------------------------------------------------------------
if [ ! -f "${STAGE1_CKPT}" ]; then
  echo "[stage1] Missing checkpoint:"
  echo "  ${STAGE1_CKPT}"
  echo "[stage1] Submitting stage-1 ERM (${STAGE1_STEPS} steps) and exiting."
  echo "         Re-run this script after stage-1 finishes."

  sbatch \
    -J "${TAG}" \
    ${SBATCH_ACCOUNT} ${SBATCH_PART} ${SBATCH_GRES} ${SBATCH_CPU} ${SBATCH_MEM} ${SBATCH_TIME} \
    -o "${SLURM_DIR}/${TAG}_${DATASET}_stage1_ERM_%j.out" \
    -e "${SLURM_DIR}/${TAG}_${DATASET}_stage1_ERM_%j.err" \
    --wrap "
      set -euo pipefail
      cd '${REPO_DIR}'
      $(env_setup_snippet)

      python -m crx.train \
        --dataset '${DATASET}' \
        --algorithm ERM \
        --train_attr '${TRAIN_ATTR}' \
        --data_dir '${DATA_DIR}' \
        --output_dir '${OUT_DIR}' \
        --output_folder_name '${STAGE1_FOLDER}' \
        --seed ${STAGE1_SEED} \
        --hparams_seed ${STAGE1_HPSEED} \
        --steps ${STAGE1_STEPS}
    "
  exit 0
fi

echo "[stage1] Found checkpoint: ${STAGE1_CKPT}"
echo "[sweep] Output folder: ${OUT_TAG_DIR}"


# ------------------------------------------------------------
# (2) One-time CRX artifact setup (recommended)
# ------------------------------------------------------------
if [[ " ${ALGS[*]} " == *" CRX "* ]]; then
  mkdir -p "${ART_DIR}"
  if [ ! -f "${ART_DIR}/done" ]; then
    echo "[crx setup] Building artifacts in: ${ART_DIR}"
    echo "           (This may take time; consider running on a GPU node if needed.)"

    # NOTE: setup_crx_artifacts may call CLIP caching; if your cluster requires GPU for this,
    # you can submit this block as an sbatch job instead of running on the login node.
    python -m crx.scripts.setup_crx_artifacts \
      --dataset "${DATASET}" \
      --data-dir "${DATA_DIR}" \
      --stage1-folder "${OUT_DIR}/${STAGE1_FOLDER}" \
      --artifact-dir "${ART_DIR}" \
      --resid-dim "${RESID_DIM}"

    touch "${ART_DIR}/done"
  else
    echo "[crx setup] Found ${ART_DIR}/done (skipping)"
  fi
fi


# ------------------------------------------------------------
# Helper: write CRX path hparams JSON for a given run dir
# ------------------------------------------------------------
write_crx_paths_json () {
  local out_json="$1"
  cat > "${out_json}" <<EOF
{
  "cr_task_id": "${DATASET,,}",
  "cr_concept_meta_path": "${ART_DIR}/meta.json",
  "cr_concept_path_tr": "${ART_DIR}/concepts_tr.pt",
  "cr_concept_path_va": "${ART_DIR}/concepts_va.pt",
  "cr_concept_path_te": "${ART_DIR}/concepts_te.pt",
  "cr_resid_dim": ${RESID_DIM},
  "cr_resid_path_va": "${ART_DIR}/resid_va.pt",
  "cr_resid_path_te": "${ART_DIR}/resid_te.pt"
}
EOF
}

submit_one () {
  local algo="$1" hp="$2" seed="$3"
  local d; d="$(run_dir "$algo" "$hp" "$seed")"

  if is_done "$d"; then
    echo "[skip done] $d"
    return
  fi

  mkdir -p "$d"
  wait_for_slot

  local stage_args=""
  if [[ "$algo" == "DFR" || "$algo" == "CRX" || "$algo" == "CRT" ]]; then
    stage_args="--stage1_folder '${STAGE1_FOLDER}' --stage1_algo '${STAGE1_ALGO}'"
  fi

  local extra_hparams_arg=""
  if [[ "$algo" == "CRX" ]]; then
    local hp_json="${d}/crx_paths.json"
    write_crx_paths_json "${hp_json}"
    # merge path hparams via --hparams (train.py will json.loads and update)
    extra_hparams_arg="--hparams \"\$(cat '${hp_json}')\""
  fi

  sbatch \
    -J "${TAG}" \
    ${SBATCH_ACCOUNT} ${SBATCH_PART} ${SBATCH_GRES} ${SBATCH_CPU} ${SBATCH_MEM} ${SBATCH_TIME} \
    -o "${SLURM_DIR}/${TAG}_${DATASET}_${algo}_hp${hp}_seed${seed}_%j.out" \
    -e "${SLURM_DIR}/${TAG}_${DATASET}_${algo}_hp${hp}_seed${seed}_%j.err" \
    --wrap "
      set -euo pipefail
      cd '${REPO_DIR}'
      $(env_setup_snippet)

      python -m crx.train \
        --dataset '${DATASET}' \
        --algorithm '${algo}' \
        --train_attr '${TRAIN_ATTR}' \
        --data_dir '${DATA_DIR}' \
        --output_dir '${OUT_DIR}' \
        --output_folder_name '${TAG}' \
        --hparams_seed '${hp}' \
        --seed '${seed}' \
        ${stage_args} \
        ${extra_hparams_arg}
    "
}

for algo in "${ALGS[@]}"; do
  for hp in ${HP_SEEDS}; do
    for seed in "${SEEDS[@]}"; do
      submit_one "$algo" "$hp" "$seed"
    done
  done
done

echo "All jobs submitted (throttled to ${MAX_IN_FLIGHT})."
echo "Monitor progress:"
echo "  python -m crx.scripts.summarize_progress --root ${OUT_TAG_DIR}"

