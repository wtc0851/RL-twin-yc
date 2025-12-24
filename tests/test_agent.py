import os
import sys
import json
import gymnasium as gym
import numpy as np
import time

# 将项目根目录添加到 Python 路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.env_penalty import YardEnv
from src.environment.dataclasses import Task
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker


def mask_fn(env: gym.Env) -> np.ndarray:
    """一个辅助函数，用于从环境中提取动作掩码。"""
    return env.action_mask()


def run(use_static_data: bool = True, enable_render: bool = False, save_plot: bool = True, output_metrics: bool = True):
    """
    使用训练好的 PPO 模型进行评估。
    - use_static_data=True: 从 'static_tasks_env.json' 加载静态任务。
    - use_static_data=False: 使用环境内置的随机任务生成器。
    """
    if output_metrics:
        print(f"--- 开始评估 PPO 模型 (env)，静态数据: {use_static_data} ---")

    static_tasks = None
    if use_static_data:
        data_path = os.path.join(os.path.dirname(__file__), "static_tasks_env.json")
        try:
            with open(data_path, "r", encoding="utf-8") as f:
                raw_tasks = json.load(f)
            static_tasks = [Task(**t) for t in raw_tasks]

            total_service_time = 0
            for task in static_tasks:
                total_service_time += task.execution_time
            if output_metrics:
                print(f"静态任务总服务时间: {total_service_time:.2f}s")
                print(f"已从 {data_path} 加载 {len(static_tasks)} 个静态任务。")
        except FileNotFoundError:
            print(f"错误：静态任务文件未找到 at {data_path}")
            return

    # 模型准备计时开始（环境创建、包装与模型加载）
    prep_start = time.perf_counter()

    # 创建环境，如果提供了静态任务，则直接注入
    render_mode = 'human' if enable_render else None
    env = YardEnv(render_mode=render_mode, static_tasks=static_tasks, max_tasks_in_obs=30)
    env = ActionMasker(env, mask_fn)

    # 加载模型
    model_path = os.path.join("models", "ppo_env_model.zip")
    if not os.path.exists(model_path):
        if output_metrics:
            print(f"错误：找不到模型文件 at {model_path}")
            print("请先运行 src/agents/train_ppo.py 脚本来训练和保存模型。")
        env.close()
        return

    model = MaskablePPO.load(model_path, env=env)
    if output_metrics:
        print(f"模型已从 {model_path} 加载。")

    obs, info = env.reset()
    prep_end = time.perf_counter()
    model_prep_time = prep_end - prep_start
    if output_metrics:
        print(f"模型准备时间: {model_prep_time:.6f}s")
    terminated = False
    truncated = False
    total_reward = 0

    if output_metrics:
        print("开始运行智能体...")
    loop_start = time.perf_counter()
    while not terminated and not truncated:
        action_masks = env.action_masks()
        action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
    loop_end = time.perf_counter()
    agent_loop_time = loop_end - loop_start

    if output_metrics:
        print("评估回合结束。")
        print(f"总奖励: {total_reward}")
        print(f"算法运行时间: {agent_loop_time:.6f}s")

    # 显示性能指标
    if output_metrics:
        print("\n--- 性能指标 ---")
        print(f"任务总等待时间: {info.get('total_task_wait_time', 0.0):.2f}s")
        print(f"场桥移动次数: {info.get('crane_move_count', 0)}")
        print(f"场桥移动总时间: {info.get('total_crane_move_time', 0.0):.2f}s")
        print(f"场桥等待总时间: {info.get('total_crane_wait_time', 0.0):.2f}s")
        print(f"场桥闲置总时间: {info.get('total_crane_idle_time', 0.0):.2f}s")
        print(f"已完成任务数: {info.get('completed_tasks_count', 0)}")
        print(f"总仿真时间: {info.get('simulation_time', 0.0):.2f}s")
        if 'termination_reason' in info:
            print(f"停止原因: {info['termination_reason']}")


    # 根据数据源动态命名并保存轨迹图
    if save_plot:
        save_name = "静态任务智能体轨迹图.png" if use_static_data else "随机任务智能体轨迹图.png"
        env.env.plot_crane_trajectories(save_path=save_name)
        if output_metrics:
            print(f"轨迹图已保存至 {save_name}")

    env.close()


def main():
    """解析命令行参数并运行评估。"""
    use_static = True  # 默认使用静态数据
    enable_render = True
    save_plot = True
    output_metrics = True
    for arg in sys.argv[1:]:
        if arg.startswith("--use_static="):
            val = arg.split("=", 1)[1].strip().lower()
            use_static = (val in {"1", "true", "yes", "y"})
        elif arg.startswith("--render="):
            val = arg.split("=", 1)[1].strip().lower()
            enable_render = (val in {"1", "true", "yes", "y"})
        elif arg.startswith("--save_plot="):
            val = arg.split("=", 1)[1].strip().lower()
            save_plot = (val in {"1", "true", "yes", "y"})
        elif arg.startswith("--output_metrics="):
            val = arg.split("=", 1)[1].strip().lower()
            output_metrics = (val in {"1", "true", "yes", "y"})
    run(use_static_data=use_static, enable_render=enable_render, save_plot=save_plot, output_metrics=output_metrics)


if __name__ == "__main__":
    main()
