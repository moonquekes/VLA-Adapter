# 最小交付：让别人采集你的自定义 suction 数据集

## 你需要交付什么

只交付两部分：

1. 你的 `VLA-Adapter-main`（包含 `suction.sh`、`suction_gripper.py`、`suction_gripper.xml`、`setup_suction_collection.sh`）
2. 你的“已改动版” `libero` 整个目录（不是原版）

## 对方机器最少步骤

假设对方把两个目录放在：

- `/path/to/VLA-Adapter-main`
- `/path/to/libero`

并且已有 conda 环境 `<env>`。

### 1) 一键安装与补丁

```bash
cd /path/to/VLA-Adapter-main
bash setup_suction_collection.sh <env> /path/to/libero
```

该命令会自动做：

- 安装 `libero/requirements.txt`
- `pip install -e /path/to/libero`
- 给当前环境内的 `robosuite` 注入 `SuctionGripper`
- 校验 `SuctionGripper` 和 `SuctionPanda` 是否可导入

### 2) 开始采集

```bash
ENV_NAME=<env> LIBERO_ROOT=/path/to/libero bash /path/to/libero/scripts/collect_only.sh
```

### 3) 离线转换

```bash
ENV_NAME=<env> LIBERO_ROOT=/path/to/libero bash /path/to/libero/scripts/offline_convert.sh
```

## 建议的交付方式（最稳）

- 用一个压缩包交付：`VLA-Adapter-main + 修改后的 libero`
- 或者把修改后的 `libero` 单独建分支/仓库，让别人直接克隆

## 常见问题

- 若报 `SuctionGripper` 不存在：重新执行 `bash suction.sh <env>`
- 若报找不到自定义 bddl 或资产：确认使用的是你交付的“改动版” `libero`，不是官方原版
- 若环境包冲突：优先新建干净 conda 环境后重跑 `setup_suction_collection.sh`
