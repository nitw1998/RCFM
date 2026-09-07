#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 || ! "$1" =~ ^[0-9]+$ || ! "$2" =~ ^(ptbxl|cpsc2018|mimic-afib|wesad|mmecg)$ || ! "$3" =~ ^(rcfm|rcfm-ot|rddm)$ ]]; then
  echo "Usage: $0 GPU_INDEX {ptbxl|cpsc2018|mimic-afib|wesad|mmecg} {rcfm|rcfm-ot|rddm}" >&2
  exit 2
fi

GPU_INDEX="$1"
DATASET_KEY="$2"
VARIANT="$3"
MIN_FREE_MEMORY_MIB=20480
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${RCFM_PYTHON:-python}"
DATA_ROOT="${RCFM_DATA_ROOT:?Set RCFM_DATA_ROOT to the preprocessing root containing the dataset directory}"
RUN_ROOT="${RCFM_RUNS_ROOT:?Set RCFM_RUNS_ROOT to the training output root}"
WANDB_MODE="${WANDB_MODE:-online}"

case "$DATASET_KEY" in
  ptbxl)
    DATASET_NAME="PTBXL"; TASK="ecg2ecg"; PREFIX="ptbxl"
    EXPECTED_VERSION="ptbxl-1.0.1-random-window80-20-record-overlap-source-record-joint12-full10s-minmax-neg1-1-v2"
    EXPECTED_HASH="9ba296dc33ef6f29f9368ae4d1dd61feceb9366100b7b4afbc8698ea7592012c"
    EXPECTED_TRAIN=34939; EXPECTED_HELDOUT=8735; HELDOUT_SPLIT="val"
    CONFIG_DIR="ptbxl"; CONFIG_SUFFIX="random_window80_20_joint12_seed31.yaml"
    ;;
  cpsc2018)
    DATASET_NAME="CPSC2018"; TASK="ecg2ecg"; PREFIX="cpsc2018"
    EXPECTED_VERSION="cpsc2018-source-fullrecord-joint12-minmax-all-nonoverlap4s-random80-20-record-overlap-v1"
    EXPECTED_HASH="b7902b112219541e795bac4f020ef268b2951f0c3f80709f0a06f18132a743d8"
    EXPECTED_TRAIN=19364; EXPECTED_HELDOUT=4842; HELDOUT_SPLIT="val"
    CONFIG_DIR="cpsc2018"; CONFIG_SUFFIX="random_window80_20_joint12_seed31.yaml"
    ;;
  mimic-afib)
    DATASET_NAME="MIMIC-AFib"; TASK="ppg2ecg"; PREFIX="mimic_afib"
    EXPECTED_VERSION="mimic-afib-all-qc-windows-random80-20-subject-record-overlap-rddm-window-minmax-v1"
    EXPECTED_HASH="8b862a432969db8e13dd5cec18928f96486b983f6147fbc2ad2d8cfb4fc96232"
    EXPECTED_TRAIN=8160; EXPECTED_HELDOUT=2040; HELDOUT_SPLIT="test"
    CONFIG_DIR="mimic_afib"; CONFIG_SUFFIX="random_window80_20_seed31.yaml"
    ;;
  wesad)
    DATASET_NAME="WESAD"; TASK="ppg2ecg"; PREFIX="wesad"
    EXPECTED_VERSION="wesad-all-windows-random80-20-subject-overlap-linear-resample-source-record-minmax-v2"
    EXPECTED_HASH="ef5687b00e5cc3809a8ac3d6b95d05671ee18b37182e04fd7635fe6657a3906c"
    EXPECTED_TRAIN=17365; EXPECTED_HELDOUT=4342; HELDOUT_SPLIT="test"
    CONFIG_DIR="wesad"; CONFIG_SUFFIX="random_window80_20_record_minmax_seed31.yaml"
    ;;
  mmecg)
    DATASET_NAME="mmECG"; TASK="rcg2ecg"; PREFIX="mmecg"
    EXPECTED_VERSION="mmecg-all-windows-random80-20-subject-record-overlap-window-minmax-v1"
    EXPECTED_HASH="6e5365be9b71c3815907eeabab2ee6b83a11a280521243a1f79c4f90da570dc2"
    EXPECTED_TRAIN=9973; EXPECTED_HELDOUT=2494; HELDOUT_SPLIT="test"
    CONFIG_DIR="mmecg"; CONFIG_SUFFIX="random_window80_20_seed31.yaml"
    ;;
esac

case "$VARIANT" in
  rcfm)
    ENTRY="train_rcfm.py"; CONFIG="configs/$CONFIG_DIR/rcfm_$CONFIG_SUFFIX"
    RUN_ID="${PREFIX}_rcfm_${CONFIG_SUFFIX%_seed31.yaml}_s31_e200_v1"
    ;;
  rcfm-ot)
    ENTRY="train_rcfm.py"; CONFIG="configs/$CONFIG_DIR/rcfm_ot_$CONFIG_SUFFIX"
    RUN_ID="${PREFIX}_rcfm_ot_${CONFIG_SUFFIX%_seed31.yaml}_s31_e200_v1"
    ;;
  rddm)
    ENTRY="train_rddm_compare.py"; CONFIG="configs/$CONFIG_DIR/rddm_$CONFIG_SUFFIX"
    RUN_ID="${PREFIX}_rddm_${CONFIG_SUFFIX%_seed31.yaml}_s31_e200_v1"
    ;;
esac

MANIFEST="$DATA_ROOT/$DATASET_NAME/dataset_manifest.json"
RUN_DIR="$RUN_ROOT/$TASK/$DATASET_NAME/$RUN_ID"
LOG_ROOT="$RUN_ROOT/logs"
PID_ROOT="$RUN_ROOT/pids"
MPL_ROOT="${MPLCONFIGDIR:-$RUN_ROOT/matplotlib}"
LAUNCH_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_PATH="$LOG_ROOT/${RUN_ID}_gpu${GPU_INDEX}_${LAUNCH_ID}.log"
PID_PATH="$PID_ROOT/${RUN_ID}_gpu${GPU_INDEX}_${LAUNCH_ID}.pid"

REQUIRED_FILES=(
  "$MANIFEST"
  "$REPO_ROOT/$CONFIG"
)
if [[ "$TASK" == "ecg2ecg" ]]; then
  REQUIRED_FILES+=(
    "$DATA_ROOT/$DATASET_NAME/X_train_resampled.npy"
    "$DATA_ROOT/$DATASET_NAME/X_${HELDOUT_SPLIT}_resampled.npy"
    "$DATA_ROOT/$DATASET_NAME/record_joint_minima_train.npy"
    "$DATA_ROOT/$DATASET_NAME/record_joint_ranges_train.npy"
    "$DATA_ROOT/$DATASET_NAME/record_joint_minima_${HELDOUT_SPLIT}.npy"
    "$DATA_ROOT/$DATASET_NAME/record_joint_ranges_${HELDOUT_SPLIT}.npy"
  )
else
  REQUIRED_FILES+=(
    "$DATA_ROOT/$DATASET_NAME/ppg_train_4sec.npy"
    "$DATA_ROOT/$DATASET_NAME/ecg_train_4sec.npy"
    "$DATA_ROOT/$DATASET_NAME/ppg_test_4sec.npy"
    "$DATA_ROOT/$DATASET_NAME/ecg_test_4sec.npy"
  )
fi
if [[ "$DATASET_KEY" == "wesad" ]]; then
  REQUIRED_FILES+=(
    "$DATA_ROOT/WESAD/subject_ids_train.npy"
    "$DATA_ROOT/WESAD/subject_ids_test.npy"
    "$DATA_ROOT/WESAD/target_record_minima_train.npy"
    "$DATA_ROOT/WESAD/target_record_ranges_train.npy"
    "$DATA_ROOT/WESAD/condition_record_minima_train.npy"
    "$DATA_ROOT/WESAD/condition_record_ranges_train.npy"
    "$DATA_ROOT/WESAD/target_record_minima_test.npy"
    "$DATA_ROOT/WESAD/target_record_ranges_test.npy"
    "$DATA_ROOT/WESAD/condition_record_minima_test.npy"
    "$DATA_ROOT/WESAD/condition_record_ranges_test.npy"
  )
fi
for path in "${REQUIRED_FILES[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
done
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable is unavailable: $PYTHON_BIN" >&2
  exit 1
fi
if [[ -e "$RUN_DIR" ]]; then
  echo "Refusing to overwrite existing training run: $RUN_DIR" >&2
  exit 1
fi

"$PYTHON_BIN" -c '
import json, sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
config = json.load(open(sys.argv[2], encoding="utf-8"))
assert manifest["status"] == "completed"
assert manifest["dataset_version"] == config["dataset_version"] == sys.argv[3]
assert manifest["split_hash"] == config["split_hash"] == sys.argv[4]
assert manifest["splits"]["train"]["windows"] == int(sys.argv[5])
assert manifest["splits"][sys.argv[6]]["windows"] == int(sys.argv[7])
assert config["epochs"] == 200 and config["batch_size"] == 128 and config["seed"] == 31
' "$MANIFEST" "$REPO_ROOT/$CONFIG" "$EXPECTED_VERSION" "$EXPECTED_HASH" "$EXPECTED_TRAIN" "$HELDOUT_SPLIT" "$EXPECTED_HELDOUT"

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
echo "GPU $GPU_INDEX has ${FREE_MEMORY_MIB} MiB free; launching despite any existing compute processes."

mkdir -p "$LOG_ROOT" "$PID_ROOT" "$RUN_ROOT/wandb" "$MPL_ROOT"
cd "$REPO_ROOT"
nohup env \
  CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
  PYTHONUNBUFFERED=1 \
  PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-$RUN_ROOT/pycache}" \
  MPLCONFIGDIR="$MPL_ROOT" \
  WANDB_DIR="$RUN_ROOT/wandb" \
  "$PYTHON_BIN" "$ENTRY" \
    --config "$CONFIG" \
    --data_root "$DATA_ROOT" \
    --output_dir "$RUN_ROOT" \
    --run_id "$RUN_ID" \
    --wandb_mode "$WANDB_MODE" \
    >"$LOG_PATH" 2>&1 &

PID=$!
echo "$PID" >"$PID_PATH"
echo "Started $DATASET_NAME $VARIANT random-window 80:20 training."
echo "Run ID: $RUN_ID"
echo "PID: $PID"
echo "Log: $LOG_PATH"
echo "Run directory: $RUN_DIR"
