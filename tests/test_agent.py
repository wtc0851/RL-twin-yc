import os
import sys
import gymnasium as gym

# 将项目根目录添加到 Python 路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.env import YardEnv
from stable_baselines3 import PPO

def main():
    """
    加载训练好的 PPO 模型，并在 YardEnv 环境中运行它以进行可视化。
    """
    print("--- 开始评估训练好的 PPO 模型 ---")

    # --- 1. 环境设置 ---
    # 创建一个用于评估的环境实例
    env = YardEnv(render_mode='human') # 使用 'human' 模式以收集绘图数据

    # --- 2. 加载模型 ---
    # 加载之前训练并保存的模型
    model_path = os.path.join("models", "ppo_yard_model.zip")
    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件 at {model_path}")
        print("请先运行 train_ppo.py 脚本来训练和保存模型。")
        return

    model = PPO.load(model_path, env=env)
    print(f"模型已从 {model_path} 加载。")

    # --- 3. 运行评估 ---
    obs, info = env.reset()
    terminated = False
    truncated = False
    total_reward = 0

    print("开始运行智能体...")
    while not terminated and not truncated:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    print("评估回合结束。")
    print(f"总奖励: {total_reward}")

    # --- 4. 可视化结果 ---
    # 调用环境的绘图方法来生成并保存轨迹图
    env.plot_crane_trajectories(save_path="trained_agent_trajectory.png")
    print("轨迹图已保存至 trained_agent_trajectory.png")

    env.close()
    print("--- 评估完成 ---")

if __name__ == "__main__":
    main()