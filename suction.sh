#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "用法: bash suction.sh <conda_env_name>"
    exit 1
fi

ENV_NAME="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ROBOSUITE=$(conda run -n "$ENV_NAME" python -c "import pathlib, robosuite; print(pathlib.Path(robosuite.__file__).resolve().parent)")

if [[ ! -d "$ROBOSUITE/models/grippers" ]]; then
    echo "未找到 robosuite grippers 目录: $ROBOSUITE/models/grippers"
    exit 1
fi

echo "[1/3] 复制吸盘 XML 到 $ROBOSUITE/models/assets/grippers/"
cp -f "$SCRIPT_DIR/suction_gripper.xml" "$ROBOSUITE/models/assets/grippers/"

echo "[2/3] 复制吸盘 Python 类到 $ROBOSUITE/models/grippers/"
cp -f "$SCRIPT_DIR/suction_gripper.py" "$ROBOSUITE/models/grippers/"

INIT_FILE="$ROBOSUITE/models/grippers/__init__.py"

echo "[3/3] 修改 $INIT_FILE"
if ! grep -q "from .suction_gripper import SuctionGripper" "$INIT_FILE"; then
    sed -i "/from .null_gripper import NullGripper/a from .suction_gripper import SuctionGripper" "$INIT_FILE"
fi

if ! grep -q '"SuctionGripper": SuctionGripper' "$INIT_FILE"; then
    sed -i "s/None: NullGripper,/\"SuctionGripper\": SuctionGripper,\n    None: NullGripper,/" "$INIT_FILE"
fi

echo "完成。请在目标环境验证："
echo "conda run -n $ENV_NAME python -c \"from robosuite.models.grippers import GRIPPER_MAPPING; print('SuctionGripper' in GRIPPER_MAPPING)\""