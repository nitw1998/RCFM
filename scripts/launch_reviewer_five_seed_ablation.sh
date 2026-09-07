#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 GPU_INDEX {ptbxl|cpsc2018|mimic-afib|wesad|mmecg} {cfm|rcfm|rcfm-ot|rddm} [all|31|32|33|34|35|comma-separated-seeds]" >&2
}
if [[ $# -lt 3 || $# -gt 4 || ! "$1" =~ ^[0-9]+$ || ! "$2" =~ ^(ptbxl|cpsc2018|mimic-afib|wesad|mmecg)$ || ! "$3" =~ ^(cfm|rcfm|rcfm-ot|rddm)$ ]]; then
  usage
  exit 2
fi

GPU_INDEX="$1"
DATASET_KEY="$2"
VARIANT="$3"
SEED_REQUEST="${4:-all}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${RCFM_PYTHON:-python}"
DATA_ROOT="${RCFM_DATA_ROOT:?Set RCFM_DATA_ROOT to the preprocessing root containing the dataset directory}"
RUN_ROOT="${RCFM_RUNS_ROOT:?Set RCFM_RUNS_ROOT to the training output root}"
WANDB_MODE="${WANDB_MODE:-online}"
RUN_TAG="${RCFM_RUN_TAG:-reviewer_5seed_v1}"
DRY_RUN="${RCFM_DRY_RUN:-0}"

[[ "$RUN_TAG" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "Invalid RCFM_RUN_TAG." >&2; exit 2; }
if [[ "$SEED_REQUEST" == "all" ]]; then
  SEEDS=(31 32 33 34 35)
else
  IFS=',' read -r -a SEEDS <<<"$SEED_REQUEST"
fi
declare -A SEEN_SEEDS=()
for seed in "${SEEDS[@]}"; do
  [[ "$seed" =~ ^(31|32|33|34|35)$ ]] || { echo "Seeds are locked to 31,32,33,34,35; got '$seed'." >&2; exit 2; }
  [[ -z "${SEEN_SEEDS[$seed]:-}" ]] || { echo "Duplicate seed '$seed'." >&2; exit 2; }
  SEEN_SEEDS[$seed]=1
done

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
  cfm)
    ENTRY="train_cfm_compare.py"
    if [[ "$TASK" == "ecg2ecg" ]]; then CONFIG="configs/$CONFIG_DIR/cfm_compare_$CONFIG_SUFFIX"; else CONFIG="configs/$CONFIG_DIR/cfm_$CONFIG_SUFFIX"; fi
    EPOCHS=200; VARIANT_ID="cfm"; MODEL_ARGS=(--region_weight 0 --no-use_minibatch_ot)
    ;;
  rcfm)
    ENTRY="train_rcfm.py"; CONFIG="configs/$CONFIG_DIR/rcfm_$CONFIG_SUFFIX"
    EPOCHS=200; VARIANT_ID="rcfm"; MODEL_ARGS=(--no-use_minibatch_ot)
    ;;
  rcfm-ot)
    ENTRY="train_rcfm.py"; CONFIG="configs/$CONFIG_DIR/rcfm_ot_$CONFIG_SUFFIX"
    EPOCHS=200; VARIANT_ID="rcfm_ot"; MODEL_ARGS=(--use_minibatch_ot --ot_method exact --ot_sampling_strategy assignment)
    ;;
  rddm)
    ENTRY="train_rddm_compare.py"; CONFIG="configs/$CONFIG_DIR/rddm_$CONFIG_SUFFIX"
    EPOCHS=400; VARIANT_ID="rddm"; MODEL_ARGS=()
    ;;
esac

run_worker() {
  MANIFEST="$DATA_ROOT/$DATASET_NAME/dataset_manifest.json"
  if [[ "$DRY_RUN" != "1" ]]; then
    REQUIRED_FILES=("$MANIFEST" "$REPO_ROOT/$CONFIG")
    if [[ "$TASK" == "ecg2ecg" ]]; then
      REQUIRED_FILES+=("$DATA_ROOT/$DATASET_NAME/X_train_resampled.npy" "$DATA_ROOT/$DATASET_NAME/X_${HELDOUT_SPLIT}_resampled.npy" "$DATA_ROOT/$DATASET_NAME/record_joint_minima_train.npy" "$DATA_ROOT/$DATASET_NAME/record_joint_ranges_train.npy" "$DATA_ROOT/$DATASET_NAME/record_joint_minima_${HELDOUT_SPLIT}.npy" "$DATA_ROOT/$DATASET_NAME/record_joint_ranges_${HELDOUT_SPLIT}.npy")
    else
      REQUIRED_FILES+=("$DATA_ROOT/$DATASET_NAME/ppg_train_4sec.npy" "$DATA_ROOT/$DATASET_NAME/ecg_train_4sec.npy" "$DATA_ROOT/$DATASET_NAME/ppg_test_4sec.npy" "$DATA_ROOT/$DATASET_NAME/ecg_test_4sec.npy")
    fi
    if [[ "$DATASET_KEY" == "wesad" ]]; then
      REQUIRED_FILES+=("$DATA_ROOT/WESAD/subject_ids_train.npy" "$DATA_ROOT/WESAD/subject_ids_test.npy" "$DATA_ROOT/WESAD/target_record_minima_train.npy" "$DATA_ROOT/WESAD/target_record_ranges_train.npy" "$DATA_ROOT/WESAD/condition_record_minima_train.npy" "$DATA_ROOT/WESAD/condition_record_ranges_train.npy" "$DATA_ROOT/WESAD/target_record_minima_test.npy" "$DATA_ROOT/WESAD/target_record_ranges_test.npy" "$DATA_ROOT/WESAD/condition_record_minima_test.npy" "$DATA_ROOT/WESAD/condition_record_ranges_test.npy")
    fi
    for path in "${REQUIRED_FILES[@]}"; do [[ -f "$path" ]] || { echo "Missing required file: $path" >&2; exit 1; }; done
    if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && [[ ! -x "$PYTHON_BIN" ]]; then echo "Python unavailable: $PYTHON_BIN" >&2; exit 1; fi
    "$PYTHON_BIN" -c '
import json, sys
m=json.load(open(sys.argv[1], encoding="utf-8")); c=json.load(open(sys.argv[2], encoding="utf-8"))
assert m["status"] == "completed"
assert m["dataset_version"] == c["dataset_version"] == sys.argv[3]
assert m["split_hash"] == c["split_hash"] == sys.argv[4]
assert m["splits"]["train"]["windows"] == int(sys.argv[5])
assert m["splits"][sys.argv[6]]["windows"] == int(sys.argv[7])
assert c["epochs"] == 200 and c["batch_size"] == 128 and c["seed"] == 31
' "$MANIFEST" "$REPO_ROOT/$CONFIG" "$EXPECTED_VERSION" "$EXPECTED_HASH" "$EXPECTED_TRAIN" "$HELDOUT_SPLIT" "$EXPECTED_HELDOUT"
  fi

  cd "$REPO_ROOT"
  for seed in "${SEEDS[@]}"; do
    RUN_ID="${PREFIX}_${VARIANT_ID}_random_window80_20_s${seed}_e${EPOCHS}_${RUN_TAG}"
    RUN_DIR="$RUN_ROOT/$TASK/$DATASET_NAME/$RUN_ID"
    if [[ -e "$RUN_DIR" ]]; then
      if "$PYTHON_BIN" -c '
import json, pathlib, sys
d = pathlib.Path(sys.argv[1])
mp, cp = d / "run_metadata.json", d / "resolved_config.yaml"
m = json.loads(mp.read_text()) if mp.is_file() else {}
c = json.loads(cp.read_text()) if cp.is_file() else {}
ok = m.get("status") == "completed" and c.get("epochs") == int(sys.argv[2]) and c.get("seed") == int(sys.argv[3])
raise SystemExit(0 if ok else 1)
' "$RUN_DIR" "$EPOCHS" "$seed"; then
        echo "Skipping completed run: $RUN_ID"
        continue
      fi
      echo "Refusing incomplete/incompatible run: $RUN_DIR; diagnose it and use a new RCFM_RUN_TAG." >&2
      exit 1
    fi
    COMMAND=("$PYTHON_BIN" "$ENTRY" --config "$CONFIG" --data_root "$DATA_ROOT" --output_dir "$RUN_ROOT" --run_id "$RUN_ID" --epochs "$EPOCHS" --seed "$seed" --wandb_mode "$WANDB_MODE" --wandb_group "reviewer-five-seed-${DATASET_KEY}-${VARIANT_ID}-${RUN_TAG}" "${MODEL_ARGS[@]}")
    if [[ "$DRY_RUN" == "1" ]]; then
      printf 'DRY-RUN dataset=%s model=%s seed=%s epochs=%s command=' "$DATASET_KEY" "$VARIANT" "$seed" "$EPOCHS"; printf '%q ' "${COMMAND[@]}"; printf '\n'
    else
      echo "Starting dataset=$DATASET_KEY model=$VARIANT seed=$seed epochs=$EPOCHS run_id=$RUN_ID"
      "${COMMAND[@]}"
    fi
  done
}

if [[ "${RCFM_FIVE_SEED_WORKER:-0}" == "1" || "$DRY_RUN" == "1" ]]; then
  run_worker
  exit 0
fi

MIN_FREE_MEMORY_MIB="${RCFM_MIN_FREE_MEMORY_MIB:-20480}"
nvidia-smi -i "$GPU_INDEX" >/dev/null 2>&1 || { echo "GPU $GPU_INDEX is not visible." >&2; exit 1; }
FREE_MEMORY_MIB="$(nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')"
[[ "$FREE_MEMORY_MIB" =~ ^[0-9]+$ ]] || { echo "Could not determine free GPU memory." >&2; exit 1; }
(( FREE_MEMORY_MIB > MIN_FREE_MEMORY_MIB )) || { echo "GPU $GPU_INDEX has ${FREE_MEMORY_MIB} MiB free; more than ${MIN_FREE_MEMORY_MIB} MiB is required." >&2; exit 1; }

LOG_ROOT="$RUN_ROOT/logs/reviewer_five_seed"; PID_ROOT="$RUN_ROOT/pids/reviewer_five_seed"
LAUNCH_ID="$(date -u +%Y%m%dT%H%M%SZ)"; JOB_ID="${DATASET_KEY}_${VARIANT}_${SEED_REQUEST//,/-}_${LAUNCH_ID}"
LOG_PATH="$LOG_ROOT/${JOB_ID}_gpu${GPU_INDEX}.log"; PID_PATH="$PID_ROOT/${JOB_ID}_gpu${GPU_INDEX}.pid"
mkdir -p "$LOG_ROOT" "$PID_ROOT" "$RUN_ROOT/wandb" "${MPLCONFIGDIR:-$RUN_ROOT/matplotlib}"
cd "$REPO_ROOT"
nohup env CUDA_VISIBLE_DEVICES="$GPU_INDEX" RCFM_FIVE_SEED_WORKER=1 PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-$RUN_ROOT/pycache}" MPLCONFIGDIR="${MPLCONFIGDIR:-$RUN_ROOT/matplotlib}" WANDB_DIR="$RUN_ROOT/wandb" "$0" "$GPU_INDEX" "$DATASET_KEY" "$VARIANT" "$SEED_REQUEST" >"$LOG_PATH" 2>&1 &
PID=$!; echo "$PID" >"$PID_PATH"
echo "Started reviewer five-seed job: dataset=$DATASET_KEY model=$VARIANT seeds=$SEED_REQUEST"
echo "PID: $PID"
echo "Log: $LOG_PATH"
echo "Run root: $RUN_ROOT"
