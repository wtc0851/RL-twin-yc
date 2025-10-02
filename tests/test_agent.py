import os
import sys
import gymnasium as gym

# 将项目根目录添加到 Python 路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.env import YardEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
import numpy as np


def mask_fn(env: gym.Env) -> np.ndarray:
    """一个辅助函数，用于从环境中提取动作掩码。"""
    return env.action_mask()


def main():
    """
    加载训练好的 PPO 模型，并在 YardEnv 环境中运行它以进行可视化。
    """
    print("--- 开始评估训练好的 PPO 模型 ---")

    # 创建一个用于评估的环境实例
    env = YardEnv(render_mode='human')  # 使用 'human' 模式以收集绘图数据
    env = ActionMasker(env, mask_fn)

    # 加载之前训练并保存的模型
    model_path = os.path.join("../models", "ppo_yard_model.zip")
    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件 at {model_path}")
        print("请先运行 train_ppo.py 脚本来训练和保存模型。")
        return

    model = MaskablePPO.load(model_path, env=env)
    print(f"模型已从 {model_path} 加载。")

    obs, info = env.reset()
    terminated = False  # 终止标志
    truncated = False  # 截断标志
    total_reward = 0

    print("开始运行智能体...")
    while not terminated and not truncated:
        action_masks = env.action_masks()
        action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    print("评估回合结束。")
    print(f"总奖励: {total_reward}")

    # 调用环境的绘图方法来生成并保存轨迹图
    env.env.plot_crane_trajectories(save_path="trained_agent_trajectory.png")
    print("轨迹图已保存至 trained_agent_trajectory.png")

    env.close()


if __name__ == "__main__":
    main()
