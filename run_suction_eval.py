"""
run_suction_eval.py
───────────────────
把 VLA-Adapter（LIBERO-Spatial-Pro）的输出动作直接套到吸盘夹爪（SuctionPanda）上运行，
生成 5 条 rollout 视频（agentview + 腕部相机横排拼图）。

用法（在 VLA-Adapter-main/ 目录下执行）：
  conda activate vla-adapter
  cd /home/x/vla/VLA-Adapter-main

  python run_suction_eval.py
  # 或自定义参数：
  python run_suction_eval.py \
      --checkpoint outputs/LIBERO-Spatial-Pro \
      --task_suite libero_spatial \
      --task_id 0 \
      --num_episodes 5 \
      --resolution 256

动作映射说明（7 维）：
  前 6 维：OSC_POSE 位姿增量（xyz + roll/pitch/yaw）——直接传入环境
  第 7 维：夹爪/吸盘
    VLA 原始输出（训练集约定）：0 = 关闭, 1 = 张开
    normalize_gripper_action → [-1, +1]，binarize=True
    invert_gripper_action    → 符号取反（配合 openvla 数据集约定）
    最终传入 robosuite 环境的 SuctionPanda：
      +1 → 吸盘启动（抓取）
      -1 → 吸盘关闭（释放）
    此映射与平行夹爪 +1=close/-1=open 语义一致，无需额外转换。
"""

import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path

import imageio
import numpy as np
import torch

# ── 路径注册 ──────────────────────────────────────────────────────────────────
VLA_ROOT = os.path.dirname(os.path.abspath(__file__))
LIBERO_ROOT = os.path.join(os.path.dirname(VLA_ROOT), "libero")
for p in [VLA_ROOT, LIBERO_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

# 注册 SuctionPanda 到 robosuite
import libero.libero.envs.robots  # noqa: F401
from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv

from experiments.robot.libero.libero_utils import quat2axisangle
from experiments.robot.openvla_utils import (
    get_action_head,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE,
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK

# ── 配置 ──────────────────────────────────────────────────────────────────────
TASK_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object":  280,
    "libero_goal":    300,
    "libero_10":      520,
}


# ── 简单 dataclass 替代 draccus 配置 ─────────────────────────────────────────
class Cfg:
    model_family             = "openvla"
    pretrained_checkpoint    = "outputs/LIBERO-Spatial-Pro"
    use_l1_regression        = True
    use_minivlm              = True
    num_diffusion_steps      = 50
    use_film                 = False
    num_images_in_input      = 2
    use_proprio              = True
    center_crop              = True
    num_open_loop_steps      = NUM_ACTIONS_CHUNK
    load_in_8bit             = False
    load_in_4bit             = False
    unnorm_key               = ""
    task_suite_name          = "libero_spatial"
    use_pro_version          = True
    save_version             = "suction"
    phase                    = "Inference"


def parse_args():
    parser = argparse.ArgumentParser(description="VLA-Adapter × 吸盘夹爪 rollout 评估")
    parser.add_argument("--checkpoint",    default="outputs/LIBERO-Spatial-Pro")
    parser.add_argument("--task_suite",    default="libero_spatial",
                        choices=list(TASK_MAX_STEPS.keys()))
    parser.add_argument("--task_id",       type=int, default=0,
                        help="任务编号（默认 0）")
    parser.add_argument("--num_episodes",  type=int, default=5,
                        help="录制 episode 数量（默认 5）")
    parser.add_argument("--resolution",    type=int, default=256,
                        help="环境渲染分辨率 px（默认 256）")
    parser.add_argument("--seed",          type=int, default=7)
    parser.add_argument("--num_steps_wait",type=int, default=10,
                        help="等待物体稳定的步数")
    return parser.parse_args()


# ── 构建吸盘环境 ──────────────────────────────────────────────────────────────
def make_suction_env(bddl_file: str, resolution: int) -> ControlEnv:
    """创建使用 SuctionPanda 的 LIBERO 环境（离屏，双摄像头）。"""
    env = ControlEnv(
        bddl_file_name=bddl_file,
        robots=["SuctionPanda"],
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=resolution,
        camera_widths=resolution,
        control_freq=20,
        horizon=600,
        ignore_done=False,
    )
    env.seed(0)
    return env


# ── 从 obs 提取图像 ──────────────────────────────────────────────────────────
def get_images(obs):
    """返回 (agentview RGB, wrist RGB)，均已旋转 180° 以对齐训练预处理。"""
    av  = obs["agentview_image"][::-1, ::-1]          # (H,W,3) uint8
    wr  = obs["robot0_eye_in_hand_image"][::-1, ::-1]
    return av, wr


# ── 动作处理 ─────────────────────────────────────────────────────────────────
def process_action(action: np.ndarray) -> np.ndarray:
    """
    标准 openvla 约定：
      normalize [0,1]→[-1,+1]（binarize）→ invert（符号取反）
    结果：
      +1 = close/吸盘启动，-1 = open/吸盘关闭
    """
    action = normalize_gripper_action(action, binarize=True)
    action = invert_gripper_action(action)
    return action


# ── 保存拼图视频 ──────────────────────────────────────────────────────────────
def save_video(frames, episode_idx, success, task_desc, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    slug = task_desc.lower().replace(" ", "_").replace(".", "_")[:50]
    path = os.path.join(save_dir,
        f"{DATE_TIME}--ep{episode_idx:02d}--success={success}--{slug}.mp4")
    writer = imageio.get_writer(path, fps=30)
    for f in frames:
        writer.append_data(f)
    writer.close()
    print(f"  [视频] 已保存 → {path}")
    return path


# ── 单 episode rollout ────────────────────────────────────────────────────────
def run_episode(env, task_desc, cfg, model, resize_size, processor,
                action_head, proprio_projector, initial_state,
                max_steps, num_steps_wait):
    """执行一次 rollout，返回 (success, mosaic_frames)。"""
    env.reset()
    obs = env.set_init_state(initial_state)

    action_queue = deque(maxlen=cfg.num_open_loop_steps)
    mosaic_frames = []   # agentview + wrist 横排
    t = 0
    success = False

    try:
        while t < max_steps + num_steps_wait:
            # 等待物体稳定
            if t < num_steps_wait:
                obs, _, done, _ = env.step([0, 0, 0, 0, 0, 0, -1])
                t += 1
                continue

            av_img, wr_img = get_images(obs)

            # 拼图帧存入视频缓冲（原始分辨率，不缩放）
            mosaic_frames.append(np.hstack([av_img, wr_img]))

            # 推理
            if len(action_queue) == 0:
                av_resized = resize_image_for_policy(av_img, resize_size)
                wr_resized = resize_image_for_policy(wr_img, resize_size)
                # robot0_gripper_qpos 在吸盘夹爪下是标量，重复一次补成 (2,)
                # 确保 proprio 向量保持 8 维（与 proprio_projector 输入维度一致）
                gq = obs["robot0_gripper_qpos"]
                gripper_qpos = np.repeat(np.atleast_1d(gq), 2)[:2]  # 保证正好 2 维
                observation = {
                    "full_image":  av_resized,
                    "wrist_image": wr_resized,
                    "state": np.concatenate((
                        obs["robot0_eef_pos"],
                        quat2axisangle(obs["robot0_eef_quat"]),
                        gripper_qpos,
                    )),
                }
                actions = get_action(
                    cfg, model, observation, task_desc,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    use_film=cfg.use_film,
                    use_minivlm=cfg.use_minivlm,
                )
                action_queue.extend(actions)

            action = process_action(action_queue.popleft())
            obs, reward, done, info = env.step(action.tolist())

            if done:
                success = True
                break
            t += 1

    except Exception as e:
        print(f"  [警告] Episode 异常: {e}")

    return success, mosaic_frames


# ── 主函数 ────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    set_seed_everywhere(args.seed)

    # 填写 Cfg
    cfg = Cfg()
    cfg.pretrained_checkpoint = args.checkpoint
    cfg.task_suite_name       = args.task_suite
    cfg.unnorm_key            = ""   # 后面由 check_unnorm_key 填

    print(f"\n{'='*60}")
    print(f"  VLA-Adapter × 吸盘夹爪 Rollout 评估")
    print(f"  Checkpoint : {cfg.pretrained_checkpoint}")
    print(f"  任务套件   : {args.task_suite}  任务 #{args.task_id}")
    print(f"  Episode 数 : {args.num_episodes}")
    print(f"  分辨率     : {args.resolution} px")
    print(f"{'='*60}\n")

    # 加载模型
    print("[1/3] 加载模型 ...")
    model = get_model(cfg)
    model.set_version(cfg.save_version)

    # unnorm_key
    if args.task_suite not in model.norm_stats:
        key = f"{args.task_suite}_no_noops"
    else:
        key = args.task_suite
    assert key in model.norm_stats, \
        f"norm_stats 中找不到 key '{key}'，可用：{list(model.norm_stats.keys())}"
    cfg.unnorm_key = key
    print(f"    unnorm_key = {cfg.unnorm_key}")

    proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)
    action_head       = get_action_head(cfg, model.llm_dim)
    processor         = get_processor(cfg)
    resize_size       = get_image_resize_size(cfg)   # 224

    # 构建 LIBERO 基准
    print("[2/3] 初始化 LIBERO 环境（SuctionPanda）...")
    bm_dict   = benchmark.get_benchmark_dict()
    task_suite = bm_dict[args.task_suite]()
    task      = task_suite.get_task(args.task_id)
    task_desc = task.language
    bddl_file = os.path.join(
        get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file,
    )
    initial_states = task_suite.get_task_init_states(args.task_id)
    max_steps = TASK_MAX_STEPS[args.task_suite]

    print(f"    任务描述  : {task_desc}")
    print(f"    BDDL 文件 : {bddl_file}")

    env = make_suction_env(bddl_file, args.resolution)

    # rollout
    print(f"\n[3/3] 开始 rollout（共 {args.num_episodes} 个 episode）...")
    save_dir = os.path.join("rollouts", "suction", DATE)
    successes = 0

    for ep in range(args.num_episodes):
        init_state = initial_states[ep % len(initial_states)]
        print(f"\n  Episode {ep+1}/{args.num_episodes} ...")
        success, frames = run_episode(
            env, task_desc, cfg, model, resize_size,
            processor, action_head, proprio_projector,
            init_state, max_steps, args.num_steps_wait,
        )
        if success:
            successes += 1
        print(f"  结果: {'✓ 成功' if success else '✗ 失败'}  "
              f"（当前成功率 {successes}/{ep+1}）")
        save_video(frames, ep + 1, success, task_desc, save_dir)

    env.env.close()

    print(f"\n{'='*60}")
    print(f"  汇总：{successes}/{args.num_episodes} 成功")
    print(f"  视频目录：{os.path.abspath(save_dir)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
