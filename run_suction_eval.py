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
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import imageio
import numpy as np
import torch
from robosuite import load_controller_config

# ── 路径注册 ──────────────────────────────────────────────────────────────────
VLA_ROOT = os.path.dirname(os.path.abspath(__file__))
LIBERO_ROOT = os.path.join(os.path.dirname(VLA_ROOT), "libero")
for p in [VLA_ROOT, LIBERO_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

# 注册 SuctionPanda 到 robosuite
import libero.libero.envs.robots  # noqa: F401
import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import TASK_MAPPING
from libero.libero.envs.env_wrapper import ControlEnv
from libero.libero.envs.suction_sticky_wrapper import SuctionStickyWrapper

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
    parser.add_argument(
        "--bddl-file",
        default="",
        help="如果提供，则按 collect_only.sh 的 custom BDDL 环境运行，而不是 benchmark task suite。",
    )
    parser.add_argument("--task_id",       type=int, default=0,
                        help="任务编号（默认 0）")
    parser.add_argument("--num_episodes",  type=int, default=5,
                        help="录制 episode 数量（默认 5）")
    parser.add_argument("--resolution",    type=int, default=256,
                        help="环境渲染分辨率 px（默认 256）")
    parser.add_argument(
        "--rotate-images-180",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否在送入策略前将 agentview / wrist 图像旋转 180°。",
    )
    parser.add_argument("--seed",          type=int, default=7)
    parser.add_argument("--num_steps_wait",type=int, default=10,
                        help="等待物体稳定的步数")
    parser.add_argument(
        "--translation-scale",
        type=float,
        default=1.0,
        help="对动作前 3 维平移增量做统一缩放。",
    )
    parser.add_argument(
        "--rotation-scale",
        type=float,
        default=1.0,
        help="对动作第 4-6 维旋转增量做统一缩放。",
    )
    parser.add_argument(
        "--max-translation-norm",
        type=float,
        default=0.0,
        help="平移向量范数上限；<=0 表示不裁剪。",
    )
    parser.add_argument(
        "--max-rotation-norm",
        type=float,
        default=0.0,
        help="旋转向量范数上限；<=0 表示不裁剪。",
    )
    parser.add_argument(
        "--debug-actions",
        action="store_true",
        help="打印模型输出动作，并将动作轨迹保存为 JSONL 方便排查。",
    )
    parser.add_argument(
        "--debug-action-steps",
        type=int,
        default=10,
        help="最多打印前多少个控制步的动作细节（默认 10）。",
    )
    parser.add_argument(
        "--debug-save-path",
        default="",
        help="可选，动作调试 JSONL 输出路径；为空则自动写入 rollouts/suction/<DATE>/。",
    )
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


def make_collect_like_suction_env(bddl_file: str, resolution: int) -> SuctionStickyWrapper:
    """Create the same custom-task suction environment style used by libero/scripts/collect_only.sh."""
    controller_config = load_controller_config(default_controller="OSC_POSE")
    problem_info = BDDLUtils.get_problem_info(bddl_file)
    problem_name = problem_info["problem_name"]

    env_kwargs = dict(
        bddl_file_name=bddl_file,
        robots=["SuctionPanda"],
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="agentview",
        ignore_done=True,
        use_camera_obs=True,
        reward_shaping=True,
        control_freq=20,
        camera_names=["robot0_eye_in_hand", "agentview"],
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env = TASK_MAPPING[problem_name](**env_kwargs)
    env = SuctionStickyWrapper(env)
    env.seed(0)
    return env


# ── 从 obs 提取图像 ──────────────────────────────────────────────────────────
def get_images(obs, rotate_180: bool = True):
    """返回 (agentview RGB, wrist RGB)，可选旋转 180°。"""
    av = obs["agentview_image"]
    wr = obs["robot0_eye_in_hand_image"]
    if rotate_180:
        av = av[::-1, ::-1]
        wr = wr[::-1, ::-1]
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


def process_action_with_details(action: np.ndarray):
    """Return raw / normalized / final actions for debugging."""
    raw_action = np.asarray(action, dtype=np.float32).copy()
    normalized_action = normalize_gripper_action(raw_action, binarize=True)
    final_action = invert_gripper_action(normalized_action)
    return raw_action, normalized_action, final_action


def clip_vector_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
    """Clip a vector by its L2 norm."""
    if max_norm <= 0:
        return vector
    norm = np.linalg.norm(vector)
    if norm <= max_norm or norm == 0:
        return vector
    return vector * (max_norm / norm)


def apply_motion_limits(action: np.ndarray, args) -> np.ndarray:
    """Apply optional scaling and norm clipping to translation / rotation before stepping the env."""
    limited_action = np.asarray(action, dtype=np.float32).copy()
    limited_action[:3] *= args.translation_scale
    limited_action[3:6] *= args.rotation_scale
    limited_action[:3] = clip_vector_norm(limited_action[:3], args.max_translation_norm)
    limited_action[3:6] = clip_vector_norm(limited_action[3:6], args.max_rotation_norm)
    return limited_action


def _to_rounded_list(array_like, ndigits: int = 4):
    return np.asarray(array_like, dtype=np.float32).round(ndigits).tolist()


def emit_action_debug(debug_file, payload):
    """Print and optionally persist action debug information."""
    if payload["event"] == "chunk":
        print(
            "    [debug] "
            f"ep={payload['episode']} policy_step={payload['policy_step']} "
            f"chunk={payload['chunk_index']} queue_len={payload['chunk_size']} "
            f"model_actions={payload['model_actions']}"
        )
    else:
        print(
            "    [debug] "
            f"ep={payload['episode']} policy_step={payload['policy_step']} "
            f"queue_after_pop={payload['queue_remaining']} "
            f"raw={payload['raw_action']} "
            f"norm={payload['normalized_action']} "
            f"final={payload['final_action']}"
        )

    if debug_file is not None:
        debug_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
        debug_file.flush()


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
def run_episode(env, args, task_desc, cfg, model, resize_size, processor,
                action_head, proprio_projector, initial_state,
                max_steps, num_steps_wait, episode_idx,
                debug_actions=False, debug_action_steps=0, debug_file=None):
    """执行一次 rollout，返回 (success, mosaic_frames)。"""
    obs = env.reset()
    if initial_state is not None:
        obs = env.set_init_state(initial_state)

    action_queue = deque(maxlen=cfg.num_open_loop_steps)
    mosaic_frames = []   # agentview + wrist 横排
    t = 0
    success = False
    chunk_index = 0

    try:
        while t < max_steps + num_steps_wait:
            # 等待物体稳定
            if t < num_steps_wait:
                obs, _, done, _ = env.step([0, 0, 0, 0, 0, 0, -1])
                t += 1
                continue

            av_img, wr_img = get_images(obs, rotate_180=args.rotate_images_180)

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
                actions_arr = np.asarray(actions, dtype=np.float32)
                if actions_arr.ndim == 1:
                    actions_arr = actions_arr[None, :]
                action_queue.extend(actions)
                if debug_actions and (t - num_steps_wait) < debug_action_steps:
                    emit_action_debug(
                        debug_file,
                        {
                            "event": "chunk",
                            "episode": episode_idx,
                            "policy_step": int(t - num_steps_wait),
                            "chunk_index": chunk_index,
                            "chunk_size": int(len(actions_arr)),
                            "model_actions": _to_rounded_list(actions_arr),
                            "eef_pos": _to_rounded_list(obs["robot0_eef_pos"]),
                            "eef_quat": _to_rounded_list(obs["robot0_eef_quat"]),
                            "gripper_qpos": _to_rounded_list(np.atleast_1d(obs["robot0_gripper_qpos"])),
                        },
                    )
                chunk_index += 1

            raw_action, normalized_action, final_action = process_action_with_details(action_queue.popleft())
            limited_action = apply_motion_limits(final_action, args)
            if debug_actions and (t - num_steps_wait) < debug_action_steps:
                emit_action_debug(
                    debug_file,
                    {
                        "event": "step",
                        "episode": episode_idx,
                        "policy_step": int(t - num_steps_wait),
                        "queue_remaining": int(len(action_queue)),
                        "raw_action": _to_rounded_list(raw_action),
                        "normalized_action": _to_rounded_list(normalized_action),
                        "final_action": _to_rounded_list(final_action),
                        "limited_action": _to_rounded_list(limited_action),
                        "translation_norm_final": round(float(np.linalg.norm(final_action[:3])), 4),
                        "translation_norm_limited": round(float(np.linalg.norm(limited_action[:3])), 4),
                        "rotation_norm_final": round(float(np.linalg.norm(final_action[3:6])), 4),
                        "rotation_norm_limited": round(float(np.linalg.norm(limited_action[3:6])), 4),
                    },
                )
            obs, reward, done, info = env.step(limited_action.tolist())

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
    print(f"  图像旋转   : {'180°' if args.rotate_images_180 else '关闭'}")
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
    if key not in model.norm_stats and len(model.norm_stats) == 1:
        key = next(iter(model.norm_stats))
    assert key in model.norm_stats, \
        f"norm_stats 中找不到 key '{key}'，可用：{list(model.norm_stats.keys())}"
    cfg.unnorm_key = key
    print(f"    unnorm_key = {cfg.unnorm_key}")

    proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)
    action_head       = get_action_head(cfg, model.llm_dim)
    processor         = get_processor(cfg)
    resize_size       = get_image_resize_size(cfg)   # 224

    print("[2/3] 初始化 LIBERO 环境（SuctionPanda）...")
    if args.bddl_file:
        bddl_file = args.bddl_file
        problem_info = BDDLUtils.get_problem_info(bddl_file)
        task_desc = problem_info["language_instruction"]
        initial_states = None
        max_steps = 600
        env = make_collect_like_suction_env(bddl_file, args.resolution)
    else:
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
        env = make_suction_env(bddl_file, args.resolution)

    print(f"    任务描述  : {task_desc}")
    print(f"    BDDL 文件 : {bddl_file}")

    # rollout
    print(f"\n[3/3] 开始 rollout（共 {args.num_episodes} 个 episode）...")
    save_dir = os.path.join("rollouts", "suction", DATE)
    debug_file = None
    if args.debug_actions:
        os.makedirs(save_dir, exist_ok=True)
        debug_path = args.debug_save_path or os.path.join(save_dir, f"{DATE_TIME}--debug-actions.jsonl")
        debug_file = open(debug_path, "w", encoding="utf-8")
        print(f"    [debug] 动作日志将写入: {os.path.abspath(debug_path)}")
    successes = 0

    for ep in range(args.num_episodes):
        init_state = None if initial_states is None else initial_states[ep % len(initial_states)]
        print(f"\n  Episode {ep+1}/{args.num_episodes} ...")
        success, frames = run_episode(
            env, args, task_desc, cfg, model, resize_size,
            processor, action_head, proprio_projector,
            init_state, max_steps, args.num_steps_wait,
            ep + 1, args.debug_actions, args.debug_action_steps, debug_file,
        )
        if success:
            successes += 1
        print(f"  结果: {'✓ 成功' if success else '✗ 失败'}  "
              f"（当前成功率 {successes}/{ep+1}）")
        save_video(frames, ep + 1, success, task_desc, save_dir)

    if debug_file is not None:
        debug_file.close()

    env.env.close()

    print(f"\n{'='*60}")
    print(f"  汇总：{successes}/{args.num_episodes} 成功")
    print(f"  视频目录：{os.path.abspath(save_dir)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
