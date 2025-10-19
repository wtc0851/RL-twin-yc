import os
import sys
import gymnasium as gym
import numpy as np

# 将项目根目录添加到 Python 路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.environment.env import YardEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

def mask_fn(env: gym.Env) -> np.ndarray:
    """一个辅助函数，用于从环境中提取动作掩码。"""
    return env.action_mask()

def main():
    # 使用 PPO 算法训练一个 Agent 来解决 YardEnv 环境中的调度问题。
    print("--- 开始 PPO 训练 ---")

    # --- 1. 环境设置 ---
    print("正在检查环境兼容性...")
    env_instance = YardEnv()
    check_env(env_instance, warn=True)
    print("环境兼容性检查通过！")

    # 为了训练，我们通常使用“向量化”的环境，它可以并行运行多个环境实例以加速数据收集。
    # DummyVecEnv 是最简单的实现，它在一个进程中按顺序运行多个环境。
    # 并行环境数（增加采样吞吐和稳定性）
    n_envs = 8
    env = DummyVecEnv([lambda: ActionMasker(YardEnv(render_mode=None), mask_fn) for _ in range(n_envs)])

    # --- 2. 模型设置 ---
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log="./ppo_yard_tensorboard/",
        learning_rate=3e-4,
        n_steps=1024,        # 每个环境每次收集的步数
        batch_size=512,      # 需能整除 n_steps * n_envs (=8192)
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        n_epochs=10,
        target_kl=0.05,
        seed=42,
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
    )

    # --- 3. 模型训练 ---
    # 我们先从一个较小的数值开始，以验证整个流程是否能跑通。
    total_timesteps = 1_000_000
    print(f"准备训练模型，总步数: {total_timesteps}...")
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    print("模型训练完成。")

    # --- 4. 保存模型 ---
    save_path = os.path.join("models", "ppo_yard_model.zip")
    model.save(save_path)
    print(f"模型已保存至: {save_path}")

    # --- 5. (可选) 评估训练好的模型 ---
    print("--- 开始评估训练好的模型 ---")
    obs = env.reset()
    for i in range(1000):  # 运行 1000 步进行测试
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        if dones.any():
            print("一个评估回合结束。")
            obs = env.reset()

    env.close()
    print("--- 训练与评估全部完成 ---")


if __name__ == "__main__":
    main()