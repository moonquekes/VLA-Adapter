#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "用法: bash setup_suction_collection.sh <conda_env_name> [libero_root]"
    exit 1
fi

ENV_NAME="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIBERO_ROOT="${2:-/home/x/vla/libero}"

if [[ ! -d "$LIBERO_ROOT" ]]; then
    echo "未找到 libero 根目录: $LIBERO_ROOT"
    exit 1
fi

if [[ ! -f "$LIBERO_ROOT/setup.py" && ! -f "$LIBERO_ROOT/pyproject.toml" ]]; then
    echo "$LIBERO_ROOT 不是可安装的 Python 项目（缺少 setup.py/pyproject.toml）"
    exit 1
fi

if [[ ! -f "$LIBERO_ROOT/requirements.txt" ]]; then
    echo "未找到依赖文件: $LIBERO_ROOT/requirements.txt"
    exit 1
fi

echo "[1/5] 安装 libero 依赖"
conda run -n "$ENV_NAME" python -m pip install -r "$LIBERO_ROOT/requirements.txt"

echo "[2/5] 以 editable 方式安装你的 libero"
conda run -n "$ENV_NAME" python -m pip install -e "$LIBERO_ROOT"

echo "[3/5] 打补丁到 robosuite（SuctionGripper）"
bash "$SCRIPT_DIR/suction.sh" "$ENV_NAME"

echo "[4/5] 校验 robosuite 与 libero 是否可导入"
conda run -n "$ENV_NAME" python - <<'PY'
from robosuite.models.grippers import GRIPPER_MAPPING
assert "SuctionGripper" in GRIPPER_MAPPING, "SuctionGripper 未注册到 GRIPPER_MAPPING"

from libero.libero.envs.robots import SuctionPanda
print("check_ok", SuctionPanda.__name__)
PY

echo "[5/5] 输出下一步命令"
echo "环境已就绪。开始采集示例："
echo "ENV_NAME=$ENV_NAME LIBERO_ROOT=$LIBERO_ROOT bash $LIBERO_ROOT/scripts/collect_only.sh"
echo "离线转数据集示例："
echo "ENV_NAME=$ENV_NAME LIBERO_ROOT=$LIBERO_ROOT bash $LIBERO_ROOT/scripts/offline_convert.sh"
