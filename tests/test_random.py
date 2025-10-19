import os
import sys
import gymnasium as gym

# 将项目根目录添加到 Python 路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.env import YardEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from tests.static_tasks import TASK_LIST
import numpy as np


def mask_fn(env: gym.Env) -> np.ndarray:
    """一个辅助函数，用于从环境中提取动作掩码。"""
    return env.action_mask()


def main():
    """
    使用静态任务数据在 YardEnv 环境中运行随机策略进行测试。
    """
    print("--- 开始使用静态任务数据评估随机策略 ---")

    # 创建一个用于评估的环境实例，使用静态任务数据
    env = YardEnv(render_mode='human', static_tasks=TASK_LIST)  # 使用 'human' 模式以收集绘图数据
    env = ActionMasker(env, mask_fn)

    obs, info = env.reset()
    terminated = False  # 终止标志
    truncated = False  # 截断标志
    total_reward = 0

    print("开始运行随机策略...")
    while not terminated and not truncated:
        action_masks = env.action_masks()

        action = np.random.choice(np.nonzero(action_masks)[0])

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    print("评估回合结束。")
    print(f"总奖励: {total_reward}")
    
    # 显示性能指标
    print("\n--- 性能指标 ---")
    print(f"任务总等待时间: {info['total_task_wait_time']:.2f}s")
    print(f"场桥移动次数: {info['crane_move_count']}")
    print(f"场桥移动总时间: {info['total_crane_move_time']:.2f}s")
    print(f"场桥等待总时间: {info['total_crane_wait_time']:.2f}s")
    print(f"已完成任务数: {info['completed_tasks_count']}")
    print(f"总仿真时间: {info['simulation_time']:.2f}s")

    # 调用环境的绘图方法来生成并保存轨迹图
    env.env.plot_crane_trajectories(save_path="静态任务随机选择轨迹图.png")
    print("轨迹图已保存至 静态任务随机选择轨迹图.png")

    env.close()


if __name__ == "__main__":
    main()
