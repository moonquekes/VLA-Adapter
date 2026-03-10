#!/bin/bash
# 下载完成后，将 modified_libero_rlds 子目录重命名（去掉 modified_ 前缀）
# 按照 README 要求的目录结构

DATA_DIR="/home/x/vla/VLA-Adapter-main/data/libero"
SRC_DIR="$DATA_DIR/modified_libero_rlds"

SUBSETS=(
    "modified_libero_spatial_no_noops:libero_spatial_no_noops"
    "modified_libero_object_no_noops:libero_object_no_noops"
    "modified_libero_goal_no_noops:libero_goal_no_noops"
    "modified_libero_10_no_noops:libero_10_no_noops"
)

echo "检查数据集下载状态..."

if [ ! -d "$SRC_DIR" ]; then
    echo "数据集目录 $SRC_DIR 不存在，请先下载数据集。"
    exit 1
fi

echo "源目录内容："
ls "$SRC_DIR"

for item in "${SUBSETS[@]}"; do
    SRC="${item%%:*}"
    DST="${item##*:}"

    if [ -d "$SRC_DIR/$SRC" ]; then
        echo "重命名: $SRC -> $DST"
        mv "$SRC_DIR/$SRC" "$DATA_DIR/$DST"
    elif [ -d "$DATA_DIR/$DST" ]; then
        echo "已存在: $DST，跳过"
    else
        echo "未找到: $SRC，可能名称不同，请手动检查"
    fi
done

echo "目录结构："
ls "$DATA_DIR"
