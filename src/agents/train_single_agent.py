import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from src.agents.single_agent import PPO
from src.environment.env import YardEnv

def flatten_observation(obs):
    """将字典类型的观测值展平为一维Numpy数组。"""
    crane_status = obs["crane_status"].flatten()
    task_list = obs["task_list"].flatten()
    action_mask = obs["action_mask"].flatten()
    crane_to_command = obs["crane_to_command"].flatten()
    return np.concatenate([crane_status, task_list, action_mask, crane_to_command])

def main():
    # --- 环境和智能体初始化 ---
    env = YardEnv()
    # 重置环境以获取初始观测值
    obs, info = env.reset()
    flat_obs = flatten_observation(obs)
    
    state_dim = len(flat_obs)
    action_dim = env.action_space.n
    
    agent = PPO(state_dim, action_dim)

    # --- 训练参数 ---
    num_episodes = 1000
    max_steps_per_episode = 500
    update_timestep = 2000
    log_interval = 100
    
    time_step = 0
    episode_rewards = []

    # --- 训练循环 ---
    for i_episode in range(1, num_episodes + 1):
        obs, info = env.reset()
        current_reward = 0

        for t in range(max_steps_per_episode):
            flat_obs = flatten_observation(obs)
            action_mask = obs['action_mask']
            
            action, prob = agent.select_action(flat_obs, action_mask)
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            done = terminated or truncated

            agent.store_transition(flat_obs, action, reward, done, prob)
            time_step += 1

            obs = next_obs
            current_reward += reward

            if time_step % update_timestep == 0:
                agent.update()

            if done:
                break
        
        episode_rewards.append(current_reward)

        if i_episode % log_interval == 0:
            avg_reward = np.mean(episode_rewards[-log_interval:])
            print(f"Episode {i_episode}, Avg Reward: {avg_reward:.2f}")
            agent.save_model("./models")

    # --- 训练结束 ---
    env.close()
    
    # 绘制奖励曲线
    plt.plot(episode_rewards)
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig("episode_rewards.png")
    plt.show()

if __name__ == '__main__':
    main()