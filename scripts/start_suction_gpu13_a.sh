#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

ENV_NAME="vla-adapter" \
CUDA_DEVICES="1,3" \
NPROC_PER_NODE="2" \
SKIP_CONVERT="1" \
DATASET_NAME="libero_suction_no_noops" \
RUN_ROOT_DIR="outputs" \
LEARNING_RATE="1e-4" \
LR_WARMUP_STEPS="500" \
SAVE_FREQ="500" \
BATCH_SIZE="2" \
GRAD_ACCUMULATION_STEPS="2" \
NUM_STEPS_BEFORE_DECAY="20000" \
MAX_STEPS="20005" \
CURR_ACTION_LOSS_WEIGHT="1.0" \
NEXT_ACTIONS_LOSS_WEIGHT="0.25" \
NUM_IMAGES_IN_INPUT="2" \
LORA_RANK="64" \
bash scripts/run_libero_suction_pipeline.sh
