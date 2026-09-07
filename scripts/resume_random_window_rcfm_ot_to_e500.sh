#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 || ! "$1" =~ ^[0-9]+$ || ! "$2" =~ ^(wesad|mmecg)$ ]]; then
  echo "Usage: $0 GPU_INDEX {wesad|mmecg}" >&2
  exit 2
fi

GPU_INDEX="$1"
DATASET_KEY="$2"
MIN_FREE_MEMORY_MIB=20480
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${RCFM_PYTHON:-python}"
DATA_ROOT="${RCFM_DATA_ROOT:?Set RCFM_DATA_ROOT to the preprocessing root containing the dataset directory}"
RUN_ROOT="${RCFM_RUNS_ROOT:?Set RCFM_RUNS_ROOT to the training output root}"
WANDB_MODE="${WANDB_MODE:-online}"

case "$DATASET_KEY" in
  wesad)
    DATASET_NAME="WESAD"; TASK="ppg2ecg"
    CONFIG="configs/wesad/rcfm_ot_random_window80_20_record_minmax_seed31.yaml"
    EXPECTED_VERSION="wesad-all-windows-random80-20-subject-overlap-linear-resample-source-record-minmax-v2"
    EXPECTED_HASH="ef5687b00e5cc3809a8ac3d6b95d05671ee18b37182e04fd7635fe6657a3906c"
    EXPECTED_TRAIN=17365; EXPECTED_HELDOUT=4342; EXPECTED_GLOBAL_STEP=27200
    SOURCE_RUN_ID="wesad_rcfm_ot_random_window80_20_record_minmax_s31_e200_v1"
    TARGET_RUN_ID="wesad_rcfm_ot_random_window80_20_record_minmax_s31_e500_resume_e200_v1"
    EXPECTED_CHECKPOINT_SHA256="29b74f64c8880365813f450bd7bf15bced2bd38611367a14ab91fcdbe22f7809"
    WANDB_GROUP_VALUE="wesad-rcfm-ot-random-window80-20-record-minmax-e500-resume"
    ;;
  mmecg)
    DATASET_NAME="mmECG"; TASK="rcg2ecg"
    CONFIG="configs/mmecg/rcfm_ot_random_window80_20_seed31.yaml"
    EXPECTED_VERSION="mmecg-all-windows-random80-20-subject-record-overlap-window-minmax-v1"
    EXPECTED_HASH="6e5365be9b71c3815907eeabab2ee6b83a11a280521243a1f79c4f90da570dc2"
    EXPECTED_TRAIN=9973; EXPECTED_HELDOUT=2494; EXPECTED_GLOBAL_STEP=15600
    SOURCE_RUN_ID="mmecg_rcfm_ot_random_window80_20_s31_e200_v1"
    TARGET_RUN_ID="mmecg_rcfm_ot_random_window80_20_s31_e500_resume_e200_v1"
    EXPECTED_CHECKPOINT_SHA256="023e33132dadd8bbb8e346f7c848fdf9b4900d836e2c78b381880e1806f4ce54"
    WANDB_GROUP_VALUE="mmecg-rcfm-ot-random-window80-20-e500-resume"
    ;;
esac

MANIFEST="$DATA_ROOT/$DATASET_NAME/dataset_manifest.json"
SOURCE_CHECKPOINT="$RUN_ROOT/$TASK/$DATASET_NAME/$SOURCE_RUN_ID/checkpoint_latest.pt"
TARGET_RUN_DIR="$RUN_ROOT/$TASK/$DATASET_NAME/$TARGET_RUN_ID"
LOG_ROOT="$RUN_ROOT/logs"
PID_ROOT="$RUN_ROOT/pids"
MPL_ROOT="${MPLCONFIGDIR:-$RUN_ROOT/matplotlib}"
LAUNCH_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_PATH="$LOG_ROOT/${TARGET_RUN_ID}_gpu${GPU_INDEX}_${LAUNCH_ID}.log"
PID_PATH="$PID_ROOT/${TARGET_RUN_ID}_gpu${GPU_INDEX}_${LAUNCH_ID}.pid"

for path in \
  "$MANIFEST" \
  "$REPO_ROOT/$CONFIG" \
  "$SOURCE_CHECKPOINT" \
  "$DATA_ROOT/$DATASET_NAME/ppg_train_4sec.npy" \
  "$DATA_ROOT/$DATASET_NAME/ecg_train_4sec.npy" \
  "$DATA_ROOT/$DATASET_NAME/ppg_test_4sec.npy" \
  "$DATA_ROOT/$DATASET_NAME/ecg_test_4sec.npy"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
done
if [[ "$DATASET_KEY" == "wesad" ]]; then
  for path in \
    "$DATA_ROOT/WESAD/subject_ids_train.npy" \
    "$DATA_ROOT/WESAD/subject_ids_test.npy" \
    "$DATA_ROOT/WESAD/target_record_minima_train.npy" \
    "$DATA_ROOT/WESAD/target_record_ranges_train.npy" \
    "$DATA_ROOT/WESAD/condition_record_minima_train.npy" \
    "$DATA_ROOT/WESAD/condition_record_ranges_train.npy" \
    "$DATA_ROOT/WESAD/target_record_minima_test.npy" \
    "$DATA_ROOT/WESAD/target_record_ranges_test.npy" \
    "$DATA_ROOT/WESAD/condition_record_minima_test.npy" \
    "$DATA_ROOT/WESAD/condition_record_ranges_test.npy"; do
    if [[ ! -f "$path" ]]; then
      echo "Missing required file: $path" >&2
      exit 1
    fi
  done
fi
if [[ -e "$TARGET_RUN_DIR" ]]; then
  echo "Refusing to overwrite existing resumed run: $TARGET_RUN_DIR" >&2
  exit 1
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable is unavailable: $PYTHON_BIN" >&2
  exit 1
fi

ACTUAL_CHECKPOINT_SHA256="$(sha256sum "$SOURCE_CHECKPOINT" | awk '{print $1}')"
if [[ "$ACTUAL_CHECKPOINT_SHA256" != "$EXPECTED_CHECKPOINT_SHA256" ]]; then
  echo "Source checkpoint SHA-256 mismatch: $ACTUAL_CHECKPOINT_SHA256" >&2
  exit 1
fi

cd "$REPO_ROOT"
"$PYTHON_BIN" -c '
import json, sys, torch
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
config = json.load(open(sys.argv[2], encoding="utf-8"))
checkpoint = torch.load(sys.argv[3], map_location="cpu")
assert manifest["status"] == "completed"
assert manifest["dataset_version"] == config["dataset_version"] == sys.argv[4]
assert manifest["split_hash"] == config["split_hash"] == sys.argv[5]
assert manifest["splits"]["train"]["windows"] == int(sys.argv[6])
assert manifest["splits"]["test"]["windows"] == int(sys.argv[7])
assert config["epochs"] == 200 and config["batch_size"] == 128 and config["seed"] == 31
assert config["use_minibatch_ot"] is True and config["ot_method"] == "exact"
assert config["ot_sampling_strategy"] == "assignment" and config["ot_strict_mode"] is True
assert checkpoint["schema_version"] == 2 and checkpoint["kind"] == "canonical_multistep_rcfm"
assert checkpoint["epoch"] == 200 and checkpoint["global_step"] == int(sys.argv[8])
assert checkpoint["config"]["dataset_version"] == sys.argv[4]
assert checkpoint["config"]["split_hash"] == sys.argv[5]
assert checkpoint["config"]["use_minibatch_ot"] is True
assert checkpoint["optimizer_state"]["param_groups"][0]["lr"] == 0.0
' "$MANIFEST" "$CONFIG" "$SOURCE_CHECKPOINT" "$EXPECTED_VERSION" "$EXPECTED_HASH" \
  "$EXPECTED_TRAIN" "$EXPECTED_HELDOUT" "$EXPECTED_GLOBAL_STEP"

if ! nvidia-smi -i "$GPU_INDEX" >/dev/null 2>&1; then
  echo "GPU $GPU_INDEX is not visible to nvidia-smi." >&2
  exit 1
fi
FREE_MEMORY_MIB="$(nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')"
if [[ ! "$FREE_MEMORY_MIB" =~ ^[0-9]+$ ]]; then
  echo "Could not determine free memory for GPU $GPU_INDEX." >&2
  exit 1
fi
if (( FREE_MEMORY_MIB <= MIN_FREE_MEMORY_MIB )); then
  echo "GPU $GPU_INDEX has ${FREE_MEMORY_MIB} MiB free; more than ${MIN_FREE_MEMORY_MIB} MiB (20 GiB) is required." >&2
  exit 1
fi

mkdir -p "$LOG_ROOT" "$PID_ROOT" "$RUN_ROOT/wandb" "$MPL_ROOT"
nohup env \
  CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
  PYTHONUNBUFFERED=1 \
  PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-$RUN_ROOT/pycache}" \
  MPLCONFIGDIR="$MPL_ROOT" \
  WANDB_DIR="$RUN_ROOT/wandb" \
  "$PYTHON_BIN" train_rcfm.py \
    --config "$CONFIG" \
    --data_root "$DATA_ROOT" \
    --output_dir "$RUN_ROOT" \
    --run_id "$TARGET_RUN_ID" \
    --epochs 500 \
    --resume_checkpoint "$SOURCE_CHECKPOINT" \
    --resume_lr_policy restart_cosine \
    --resume_restart_lr 1e-5 \
    --wandb_mode "$WANDB_MODE" \
    --wandb_group "$WANDB_GROUP_VALUE" \
    --wandb_run_name "$TARGET_RUN_ID" \
    >"$LOG_PATH" 2>&1 &

PID=$!
echo "$PID" >"$PID_PATH"
echo "Started $DATASET_NAME RCFM-OT continuation: epoch 200 -> 500."
echo "LR policy: restore AdamW moments; restart cosine at 1e-5 for 300 epochs; no warm-up."
echo "Source checkpoint SHA-256: $ACTUAL_CHECKPOINT_SHA256"
echo "Run ID: $TARGET_RUN_ID"
echo "PID: $PID"
echo "Log: $LOG_PATH"
echo "Run directory: $TARGET_RUN_DIR"
