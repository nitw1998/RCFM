#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 || ! "$1" =~ ^[0-9]+$ || ! "$2" =~ ^(ptbxl|cpsc2018|mimic_afib|wesad|mmecg|all)$ || ! "$3" =~ ^(31|32|33|all)$ ]]; then
  echo "Usage: $0 GPU_INDEX {ptbxl|cpsc2018|mimic_afib|wesad|mmecg|all} {31|32|33|all}" >&2
  exit 2
fi

GPU_INDEX="$1"
DATASET_REQUEST="$2"
SEED_REQUEST="$3"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="${RCFM_WORKSPACE_ROOT:-$(cd "$REPO_ROOT/.." && pwd)}"
RUN_ROOT="${RCFM_RUNS_ROOT:-$WORKSPACE_ROOT/runs/training/rcfm_onestep_cfm50_v2}"
WORKER="$REPO_ROOT/scripts/run_rcfm_onestep_five_dataset_worker.sh"

if [[ "$DATASET_REQUEST" == "all" ]]; then DATASETS=(ptbxl cpsc2018 mimic_afib wesad mmecg); else DATASETS=("$DATASET_REQUEST"); fi
if [[ "$SEED_REQUEST" == "all" ]]; then SEEDS=(31 32 33); else SEEDS=("$SEED_REQUEST"); fi

if [[ ! -x "$WORKER" ]]; then
  echo "Worker is missing or not executable: $WORKER" >&2
  exit 1
fi
if ! nvidia-smi -i "$GPU_INDEX" >/dev/null 2>&1; then
  echo "GPU $GPU_INDEX is not visible to nvidia-smi." >&2
  exit 1
fi
ACTIVE_PIDS="$(nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')"
if [[ -n "$ACTIVE_PIDS" ]]; then
  echo "GPU $GPU_INDEX already has compute process(es): $ACTIVE_PIDS" >&2
  exit 1
fi

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/pids" "$RUN_ROOT/wandb" "$RUN_ROOT/matplotlib"
QUEUE_TAG="${DATASET_REQUEST}_s${SEED_REQUEST}"
LOG_PATH="$RUN_ROOT/logs/${QUEUE_TAG}_gpu${GPU_INDEX}.log"
PID_PATH="$RUN_ROOT/pids/${QUEUE_TAG}_gpu${GPU_INDEX}.pid"

nohup env \
  CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
  PYTHONUNBUFFERED=1 \
  MPLCONFIGDIR="$RUN_ROOT/matplotlib" \
  WANDB_DIR="$RUN_ROOT/wandb" \
  RCFM_RUNS_ROOT="$RUN_ROOT" \
  RCFM_WORKSPACE_ROOT="$WORKSPACE_ROOT" \
  RCFM_PYTHON="${RCFM_PYTHON:-python}" \
  WANDB_MODE="${WANDB_MODE:-online}" \
  bash -c 'set -euo pipefail; worker="$1"; shift; datasets_csv="$1"; shift; IFS=, read -r -a datasets <<< "$datasets_csv"; for dataset in "${datasets[@]}"; do "$worker" "$dataset" "$@"; done' \
  _ "$WORKER" "$(IFS=,; echo "${DATASETS[*]}")" "${SEEDS[@]}" >"$LOG_PATH" 2>&1 &

PID=$!
echo "$PID" >"$PID_PATH"
echo "Started CFM-NFE50 -> RCFM-OneStep queue on GPU $GPU_INDEX (PID $PID)."
echo "Datasets: ${DATASETS[*]}; seeds: ${SEEDS[*]}"
echo "Log: $LOG_PATH"
echo "Run root: $RUN_ROOT/one_step"
