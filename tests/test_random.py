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
from sb3_contrib.common.wrappers import ActionMasker


def mask_fn(env: gym.Env) -> np.ndarray:
    """一个辅助函数，用于从环境中提取动作掩码。"""
    return env.action_mask()


def run(use_static_data: bool = True, enable_render: bool = False, save_plot: bool = True, output_metrics: bool = True):
    """
    运行随机策略测试。
    - use_static_data=True 时，从 tests/static_tasks_env.json 加载静态任务并注入环境。
    - use_static_data=False 时，使用环境自身的随机任务生成机制。
    """
    if output_metrics:
        print("--- 开始评估随机策略 (env)，静态数据: {} ---".format(use_static_data))

    static_tasks = None
    if use_static_data:
        data_path = os.path.join(os.path.dirname(__file__), "static_tasks_env.json")
        with open(data_path, "r", encoding="utf-8") as f:
            raw_tasks = json.load(f)
        static_tasks = [Task(**t) for t in raw_tasks]

    # 创建环境：通过构造参数传入静态任务（若启用）
    render_mode = 'human' if enable_render else None
    env = YardEnv(render_mode=render_mode, static_tasks=static_tasks)
    env = ActionMasker(env, mask_fn)

    obs, info = env.reset()
    terminated = False
    truncated = False
    total_reward = 0.0

    if output_metrics:
        print("开始运行随机策略...")
    loop_start = time.perf_counter()
    
    consecutive_waits = 0
    # 设置一个较大的步数上限作为安全网
    for _ in range(10000):
        if terminated or truncated:
            break
            
        action_masks = env.action_masks()
        valid_actions = np.nonzero(action_masks)[0]
        if len(valid_actions) == 0:
            break
            
        action = np.random.choice(valid_actions)
        
        # 死锁检测逻辑：连续多次等待且有任务积压
        wait_action_idx = env.action_space.n - 1
        if action == wait_action_idx:
            consecutive_waits += 1
        else:
            consecutive_waits = 0
            
        if consecutive_waits > 100 and len(env.unwrapped.task_queue) > 0:
            if output_metrics:
                print(f"⚠️ 警告: 检测到潜在死锁! 连续 {consecutive_waits} 步随机选择等待，且仍有 {len(env.unwrapped.task_queue)} 个任务未完成。")
            break
            
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
        print(f"任务总等待时间: {info['total_task_wait_time']:.2f}s")
        print(f"场桥移动次数: {info['crane_move_count']}")
        print(f"场桥移动总时间: {info['total_crane_move_time']:.2f}s")
        print(f"场桥等待总时间: {info['total_crane_wait_time']:.2f}s")
        print(f"场桥闲置总时间: {info['total_crane_idle_time']:.2f}s")
        print(f"已完成任务数: {info['completed_tasks_count']}")
        print(f"总仿真时间: {info['simulation_time']:.2f}s")
        print(f"步数: {info['step_count']}")
        print(f"停止原因: {info['termination_reason']}")


    # 保存轨迹图
    if save_plot:
        save_name = "静态任务随机选择轨迹图.png" if use_static_data else "随机生成任务随机选择轨迹图.png"
        env.env.plot_crane_trajectories(save_path=save_name)
        if output_metrics:
            print(f"轨迹图已保存至 {save_name}")

    env.close()


def main():
    # 命令行参数：--use_static=, --render=, --save_plot=
    use_static = True
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
