#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ENV_NAME="${ENV_NAME:-vla-adapter}"
CONDA_EXE="${CONDA_EXE:-$(command -v conda)}"
CUDA_DEVICES="${CUDA_DEVICES:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
INPUT_DIR="${INPUT_DIR:-/home/x/vla/libero/data/suction_dataset/converted_hdf5}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/data/libero}"
DATASET_NAME="${DATASET_NAME:-libero_suction_no_noops}"
MAX_EPISODES_PER_SHARD="${MAX_EPISODES_PER_SHARD:-8}"
ROTATE_180="${ROTATE_180:-0}"
SKIP_CONVERT="${SKIP_CONVERT:-1}"

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
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-0}"
LORA_RANK="${LORA_RANK:-64}"
NUM_IMAGES_IN_INPUT="${NUM_IMAGES_IN_INPUT:-2}"
MERGE_LORA_DURING_TRAINING="${MERGE_LORA_DURING_TRAINING:-False}"
CURR_ACTION_LOSS_WEIGHT="${CURR_ACTION_LOSS_WEIGHT:-1.0}"
NEXT_ACTIONS_LOSS_WEIGHT="${NEXT_ACTIONS_LOSS_WEIGHT:-1.0}"
LOSS_TYPE="${LOSS_TYPE:-smooth_l1}"
FUTURE_HORIZON_WEIGHTS="${FUTURE_HORIZON_WEIGHTS:-1.0,1.2,1.5,2.0,2.5,3.0,3.0}"
TRANSITION_CHUNK_WEIGHT="${TRANSITION_CHUNK_WEIGHT:-4.0}"
TRANSITION_STEP_RADIUS="${TRANSITION_STEP_RADIUS:-1}"
TRANSITION_DIM_WEIGHTS="${TRANSITION_DIM_WEIGHTS:-1.25,1.25,2.5,1.0,1.0,1.0,4.0}"
TRANSITION_GRIPPER_THRESHOLD="${TRANSITION_GRIPPER_THRESHOLD:-0.5}"
TRANSITION_WEIGHT_WARMUP_STEPS="${TRANSITION_WEIGHT_WARMUP_STEPS:-2000}"
ENABLE_CHECKPOINT_ROLLOUT_EVAL="${ENABLE_CHECKPOINT_ROLLOUT_EVAL:-True}"
CHECKPOINT_ROLLOUT_NUM_EPISODES="${CHECKPOINT_ROLLOUT_NUM_EPISODES:-10}"
CHECKPOINT_ROLLOUT_TASK_SUITE="${CHECKPOINT_ROLLOUT_TASK_SUITE:-libero_spatial}"
CHECKPOINT_ROLLOUT_TASK_ID="${CHECKPOINT_ROLLOUT_TASK_ID:-0}"
CHECKPOINT_ROLLOUT_RESOLUTION="${CHECKPOINT_ROLLOUT_RESOLUTION:-256}"
CHECKPOINT_ROLLOUT_NUM_STEPS_WAIT="${CHECKPOINT_ROLLOUT_NUM_STEPS_WAIT:-10}"
LAUNCH_METHOD="${LAUNCH_METHOD:-tmux}"

CURRENT_TIME="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE="$LOG_DIR/VLA-Adapter--${DATASET_NAME}--${CURRENT_TIME}.log"
TMUX_SESSION_NAME="${TMUX_SESSION_NAME:-suction_train_${CURRENT_TIME}}"
CONDA_BASE="$(dirname "$(dirname "$CONDA_EXE")")"
ENV_BIN="$CONDA_BASE/envs/$ENV_NAME/bin"
TORCHRUN_BIN="$ENV_BIN/torchrun"

mkdir -p "$LOG_DIR"

if [[ ! -d "$PROJECT_ROOT/$VLM_PATH" ]]; then
    echo "未找到 VLM 权重目录: $PROJECT_ROOT/$VLM_PATH"
    exit 1
fi

if [[ ! -x "$TORCHRUN_BIN" ]]; then
    echo "未找到 torchrun 可执行文件: $TORCHRUN_BIN"
    exit 1
fi

cd "$PROJECT_ROOT"

if [[ "$SKIP_CONVERT" == "0" ]]; then
    if [[ ! -d "$INPUT_DIR" ]]; then
        echo "未找到 converted_hdf5 目录: $INPUT_DIR"
        exit 1
    fi

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

        "$ENV_BIN/python" "${CONVERT_ARGS[@]}"
else
    echo "[1/2] 跳过 converted_hdf5 -> RLDS 转换 (SKIP_CONVERT=1)"
fi

echo "[2/2] 启动训练"
echo "ENV=$ENV_NAME GPUs=$CUDA_DEVICES nproc=$NPROC_PER_NODE bs=$BATCH_SIZE grad_acc=$GRAD_ACCUMULATION_STEPS lr=$LEARNING_RATE warmup=$LR_WARMUP_STEPS save_freq=$SAVE_FREQ curr_w=$CURR_ACTION_LOSS_WEIGHT next_w=$NEXT_ACTIONS_LOSS_WEIGHT loss_type=$LOSS_TYPE rollout_eval=$ENABLE_CHECKPOINT_ROLLOUT_EVAL launch=$LAUNCH_METHOD"

TRAIN_CMD_STR=$(
    cat <<EOF
cd "$PROJECT_ROOT" && export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" && export PATH="$ENV_BIN:\$PATH" && "$TORCHRUN_BIN" \
--standalone --nnodes 1 --nproc-per-node "$NPROC_PER_NODE" \
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
--merge_lora_during_training "$MERGE_LORA_DURING_TRAINING" \
--batch_size "$BATCH_SIZE" \
--grad_accumulation_steps "$GRAD_ACCUMULATION_STEPS" \
--learning_rate "$LEARNING_RATE" \
--lr_warmup_steps "$LR_WARMUP_STEPS" \
--curr_action_loss_weight "$CURR_ACTION_LOSS_WEIGHT" \
--next_actions_loss_weight "$NEXT_ACTIONS_LOSS_WEIGHT" \
--loss_type "$LOSS_TYPE" \
--future_horizon_weights "$FUTURE_HORIZON_WEIGHTS" \
--transition_chunk_weight "$TRANSITION_CHUNK_WEIGHT" \
--transition_step_radius "$TRANSITION_STEP_RADIUS" \
--transition_dim_weights "$TRANSITION_DIM_WEIGHTS" \
--transition_gripper_threshold "$TRANSITION_GRIPPER_THRESHOLD" \
--transition_weight_warmup_steps "$TRANSITION_WEIGHT_WARMUP_STEPS" \
--enable_checkpoint_rollout_eval "$ENABLE_CHECKPOINT_ROLLOUT_EVAL" \
--checkpoint_rollout_num_episodes "$CHECKPOINT_ROLLOUT_NUM_EPISODES" \
--checkpoint_rollout_task_suite "$CHECKPOINT_ROLLOUT_TASK_SUITE" \
--checkpoint_rollout_task_id "$CHECKPOINT_ROLLOUT_TASK_ID" \
--checkpoint_rollout_resolution "$CHECKPOINT_ROLLOUT_RESOLUTION" \
--checkpoint_rollout_num_steps_wait "$CHECKPOINT_ROLLOUT_NUM_STEPS_WAIT" \
--lora_rank "$LORA_RANK" \
--use_pro_version True \
--wandb_entity "$WANDB_ENTITY" \
--wandb_project "$WANDB_PROJECT" \
--run_id_note "suction-${CURRENT_TIME}"
EOF
)

TRAIN_CMD_WITH_LOG="$TRAIN_CMD_STR > \"$LOG_FILE\" 2>&1"

case "$LAUNCH_METHOD" in
    tmux)
        if ! command -v tmux >/dev/null 2>&1; then
            echo "tmux 不可用，无法使用 LAUNCH_METHOD=tmux"
            exit 1
        fi
        if tmux has-session -t "$TMUX_SESSION_NAME" 2>/dev/null; then
            echo "tmux session 已存在: $TMUX_SESSION_NAME"
            exit 1
        fi
        printf -v TMUX_SHELL_CMD '%q' "$TRAIN_CMD_WITH_LOG"
        tmux new-session -d -s "$TMUX_SESSION_NAME" "bash -lc $TMUX_SHELL_CMD"
        TRAIN_PID="$(tmux list-panes -t "$TMUX_SESSION_NAME" -F '#{pane_pid}' | head -n 1)"
        ;;
    nohup)
        nohup bash -lc "$TRAIN_CMD_WITH_LOG" >/dev/null 2>&1 &
        TRAIN_PID=$!
        ;;
    background)
        bash -lc "$TRAIN_CMD_WITH_LOG" &
        TRAIN_PID=$!
        ;;
    *)
        echo "不支持的 LAUNCH_METHOD: $LAUNCH_METHOD（可选: tmux, nohup, background）"
        exit 1
        ;;
esac
echo "训练已在后台启动"
echo "训练 PID: $TRAIN_PID"
echo "日志文件: $LOG_FILE"
if [[ "$LAUNCH_METHOD" == "tmux" ]]; then
    echo "tmux session: $TMUX_SESSION_NAME"
    echo "查看命令: tmux attach -t $TMUX_SESSION_NAME"
fi
