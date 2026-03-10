#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ENV_NAME="${ENV_NAME:-vla-adapter}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
INPUT_DIR="${INPUT_DIR:-/home/x/vla/libero/data/suction_dataset/converted_hdf5}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/data/libero}"
DATASET_NAME="${DATASET_NAME:-libero_suction_no_noops}"
MAX_EPISODES_PER_SHARD="${MAX_EPISODES_PER_SHARD:-8}"
ROTATE_180="${ROTATE_180:-0}"

VLM_PATH="${VLM_PATH:-pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b}"
CONFIG_FILE_PATH="${CONFIG_FILE_PATH:-pretrained_models/configs}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-outputs}"
WANDB_ENTITY="${WANDB_ENTITY:-your-wandb-entity}"
WANDB_PROJECT="${WANDB_PROJECT:-$DATASET_NAME}"
NUM_STEPS_BEFORE_DECAY="${NUM_STEPS_BEFORE_DECAY:-20000}"
MAX_STEPS="${MAX_STEPS:-20005}"
SAVE_FREQ="${SAVE_FREQ:-2000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUMULATION_STEPS="${GRAD_ACCUMULATION_STEPS:-8}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
LORA_RANK="${LORA_RANK:-64}"
NUM_IMAGES_IN_INPUT="${NUM_IMAGES_IN_INPUT:-2}"

CURRENT_TIME="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE="$LOG_DIR/VLA-Adapter--${DATASET_NAME}--${CURRENT_TIME}.log"

mkdir -p "$LOG_DIR"

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "未找到 converted_hdf5 目录: $INPUT_DIR"
    exit 1
fi

if [[ ! -d "$PROJECT_ROOT/$VLM_PATH" ]]; then
    echo "未找到 VLM 权重目录: $PROJECT_ROOT/$VLM_PATH"
    exit 1
fi

cd "$PROJECT_ROOT"

echo "[1/2] 转换 converted_hdf5 -> RLDS"
CONVERT_ARGS=(
    scripts/convert_libero_hdf5_to_rlds.py
    --input_dir "$INPUT_DIR"
    --output_root "$OUTPUT_ROOT"
    --dataset_name "$DATASET_NAME"
    --max_episodes_per_shard "$MAX_EPISODES_PER_SHARD"
    --overwrite
)

if [[ "$ROTATE_180" == "1" ]]; then
    CONVERT_ARGS+=(--rotate_180)
fi

conda run -n "$ENV_NAME" python "${CONVERT_ARGS[@]}"

echo "[2/2] 启动训练"
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" conda run -n "$ENV_NAME" torchrun \
    --standalone --nnodes 1 --nproc-per-node 1 \
    vla-scripts/finetune.py \
    --vlm_path "$VLM_PATH" \
    --config_file_path "$CONFIG_FILE_PATH" \
    --data_root_dir data/libero \
    --dataset_name "$DATASET_NAME" \
    --run_root_dir "$RUN_ROOT_DIR" \
    --use_film False \
    --num_images_in_input "$NUM_IMAGES_IN_INPUT" \
    --use_proprio True \
    --use_lora True \
    --use_fz False \
    --use_minivlm True \
    --image_aug True \
    --num_steps_before_decay "$NUM_STEPS_BEFORE_DECAY" \
    --max_steps "$MAX_STEPS" \
    --save_freq "$SAVE_FREQ" \
    --save_latest_checkpoint_only False \
    --merge_lora_during_training True \
    --batch_size "$BATCH_SIZE" \
    --grad_accumulation_steps "$GRAD_ACCUMULATION_STEPS" \
    --learning_rate "$LEARNING_RATE" \
    --lora_rank "$LORA_RANK" \
    --use_pro_version True \
    --wandb_entity "$WANDB_ENTITY" \
    --wandb_project "$WANDB_PROJECT" \
    --run_id_note "suction-${CURRENT_TIME}" \
    > "$LOG_FILE" 2>&1 &

echo "训练已在后台启动"
echo "日志文件: $LOG_FILE"