#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ABLATION="${1:-chunk1}"
ENV_NAME="${ENV_NAME:-vla-adapter}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

DATA_ROOT_DIR="${DATA_ROOT_DIR:-$PROJECT_ROOT/data/libero}"
DATASET_NAME="${DATASET_NAME:-libero_suction_no_noops}"
VLM_PATH="${VLM_PATH:-pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b}"
CONFIG_FILE_PATH="${CONFIG_FILE_PATH:-pretrained_models/configs}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-outputs}"
WANDB_PROJECT="${WANDB_PROJECT:-suction-ablation}"

MAX_STEPS="${MAX_STEPS:-400}"
SAVE_FREQ="${SAVE_FREQ:-200}"
NUM_STEPS_BEFORE_DECAY="${NUM_STEPS_BEFORE_DECAY:-400}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUMULATION_STEPS="${GRAD_ACCUMULATION_STEPS:-8}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
LORA_RANK="${LORA_RANK:-64}"
NUM_IMAGES_IN_INPUT="${NUM_IMAGES_IN_INPUT:-2}"
USE_PROPRIO="${USE_PROPRIO:-True}"
USE_MINIVLM="${USE_MINIVLM:-True}"
MERGE_LORA_DURING_TRAINING="${MERGE_LORA_DURING_TRAINING:-False}"
WANDB_LOG_FREQ="${WANDB_LOG_FREQ:-1}"

DIAG_START_STEP="${DIAG_START_STEP:-0}"
DIAG_END_STEP="${DIAG_END_STEP:-120}"
DIAG_FOCUS_STEPS="${DIAG_FOCUS_STEPS:-0,5,10,15,70,85,100,110}"
DIAG_NUM_OPEN_LOOP_STEPS="${DIAG_NUM_OPEN_LOOP_STEPS:-}"

CURRENT_TIME="$(date +%Y%m%d_%H%M%S)"
RUN_TAG="ablation-${ABLATION}-${CURRENT_TIME}"
RUN_ID_OVERRIDE="configs+${DATASET_NAME}+b$((BATCH_SIZE * GRAD_ACCUMULATION_STEPS))+lr-${LEARNING_RATE}+lora-r${LORA_RANK}+${RUN_TAG}"
DIAG_OUTPUT_DIR="$PROJECT_ROOT/tmp/${RUN_TAG}-diagnose"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs/ablations}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/${RUN_TAG}.log}"

IMAGE_AUG="True"
CHUNK_OVERRIDE=""

case "$ABLATION" in
    chunk1)
        CHUNK_OVERRIDE="1"
        ;;
    no_image_aug)
        IMAGE_AUG="False"
        ;;
    *)
        echo "Unsupported ablation: $ABLATION"
        echo "Usage: $0 [chunk1|no_image_aug]"
        exit 1
        ;;
esac

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

CHECKPOINT_DIR="$PROJECT_ROOT/$RUN_ROOT_DIR/${RUN_ID_OVERRIDE}--${MAX_STEPS}_chkpt"

echo "=== Suction short ablation ==="
echo "ablation          : $ABLATION"
echo "env               : $ENV_NAME"
echo "gpu               : $CUDA_DEVICE"
echo "run_id_override   : $RUN_ID_OVERRIDE"
echo "checkpoint_dir    : $CHECKPOINT_DIR"
echo "diagnose_output   : $DIAG_OUTPUT_DIR"
echo "log_file          : $LOG_FILE"
echo "image_aug         : $IMAGE_AUG"
if [[ -n "$CHUNK_OVERRIDE" ]]; then
    echo "chunk_override    : $CHUNK_OVERRIDE"
fi

cd "$PROJECT_ROOT"

TRAIN_ENV=(
    "CUDA_VISIBLE_DEVICES=$CUDA_DEVICE"
    "TOKENIZERS_PARALLELISM=false"
)

if [[ -n "$CHUNK_OVERRIDE" ]]; then
    TRAIN_ENV+=("VLA_NUM_ACTIONS_CHUNK_OVERRIDE=$CHUNK_OVERRIDE")
fi

env "${TRAIN_ENV[@]}" conda run -n "$ENV_NAME" torchrun \
    --standalone --nnodes 1 --nproc-per-node 1 \
    vla-scripts/finetune.py \
    --vlm_path "$VLM_PATH" \
    --config_file_path "$CONFIG_FILE_PATH" \
    --data_root_dir "$DATA_ROOT_DIR" \
    --dataset_name "$DATASET_NAME" \
    --run_root_dir "$RUN_ROOT_DIR" \
    --use_film False \
    --num_images_in_input "$NUM_IMAGES_IN_INPUT" \
    --use_proprio "$USE_PROPRIO" \
    --use_lora True \
    --use_fz False \
    --use_minivlm "$USE_MINIVLM" \
    --image_aug "$IMAGE_AUG" \
    --num_steps_before_decay "$NUM_STEPS_BEFORE_DECAY" \
    --max_steps "$MAX_STEPS" \
    --save_freq "$SAVE_FREQ" \
    --save_latest_checkpoint_only False \
    --merge_lora_during_training "$MERGE_LORA_DURING_TRAINING" \
    --batch_size "$BATCH_SIZE" \
    --grad_accumulation_steps "$GRAD_ACCUMULATION_STEPS" \
    --learning_rate "$LEARNING_RATE" \
    --lora_rank "$LORA_RANK" \
    --use_pro_version True \
    --wandb_project "$WANDB_PROJECT" \
    --run_id_override "$RUN_ID_OVERRIDE" \
    --wandb_log_freq "$WANDB_LOG_FREQ"

if [[ ! -d "$CHECKPOINT_DIR" ]]; then
    echo "Expected checkpoint directory not found: $CHECKPOINT_DIR"
    exit 1
fi

DIAG_ENV=("CUDA_VISIBLE_DEVICES=$CUDA_DEVICE")
if [[ -n "$CHUNK_OVERRIDE" ]]; then
    DIAG_ENV+=("VLA_NUM_ACTIONS_CHUNK_OVERRIDE=$CHUNK_OVERRIDE")
fi

env "${DIAG_ENV[@]}" conda run -n "$ENV_NAME" python scripts/diagnose_suction_alignment.py \
    --dataset-dir "$PROJECT_ROOT/data/libero/$DATASET_NAME/1.0.0" \
    --checkpoint "$CHECKPOINT_DIR" \
    --unnorm-key "$DATASET_NAME" \
    --episode-index 0 \
    --start-step "$DIAG_START_STEP" \
    --end-step "$DIAG_END_STEP" \
    --focus-steps "$DIAG_FOCUS_STEPS" \
    --num-open-loop-steps "${DIAG_NUM_OPEN_LOOP_STEPS:-${CHUNK_OVERRIDE:-8}}" \
    --output-dir "$DIAG_OUTPUT_DIR" \
    --overwrite

echo
echo "Finished ablation: $ABLATION"
echo "Checkpoint: $CHECKPOINT_DIR"
echo "Diagnosis:  $DIAG_OUTPUT_DIR"
