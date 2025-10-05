import os
import sys
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json

# 将项目根目录添加到 Python 路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.environment.env import YardEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

def mask_fn(env: gym.Env) -> np.ndarray:
    """一个辅助函数，用于从环境中提取动作掩码。"""
    return env.action_mask()

class TrainingMetricsCallback(BaseCallback):
    """自定义回调函数，用于收集训练指标"""
    
    def __init__(self, verbose=0):
        super(TrainingMetricsCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        self.success_rates = []
        self.timesteps = []
        self.episode_count = 0
        
        # 用于跟踪每个环境的累计奖励和步数
        self.current_episode_rewards = []
        self.current_episode_lengths = []
        
    def _on_step(self) -> bool:
        # 初始化当前回合数据（如果需要）
        if len(self.current_episode_rewards) == 0:
            self.current_episode_rewards = [0.0] * self.training_env.num_envs
            self.current_episode_lengths = [0] * self.training_env.num_envs
        
        # 累计每个环境的奖励和步数
        rewards = self.locals.get('rewards', [])
        dones = self.locals.get('dones', [])
        
        for i in range(len(rewards)):
            self.current_episode_rewards[i] += rewards[i]
            self.current_episode_lengths[i] += 1
            
            # 检查是否有环境完成了回合
            if dones[i]:
                # 记录完成的回合数据
                self.episode_rewards.append(self.current_episode_rewards[i])
                self.episode_lengths.append(self.current_episode_lengths[i])
                self.episode_count += 1
                
                # 计算成功率（基于奖励阈值）
                success = 1 if self.current_episode_rewards[i] > 0 else 0
                self.success_rates.append(success)
                
                if self.verbose > 0:
                    print(f"Episode {self.episode_count}: Reward={self.current_episode_rewards[i]:.2f}, Length={self.current_episode_lengths[i]}")
                
                # 重置该环境的累计数据
                self.current_episode_rewards[i] = 0.0
                self.current_episode_lengths[i] = 0
        
        return True
    
    def _on_rollout_end(self) -> None:
        # 收集损失信息（在rollout结束时）
        if hasattr(self.model, 'logger') and self.model.logger.name_to_value:
            if 'train/loss' in self.model.logger.name_to_value:
                self.losses.append(self.model.logger.name_to_value['train/loss'])
                self.timesteps.append(self.num_timesteps)
    
    def _on_training_end(self) -> None:
        """训练结束时的最终统计"""
        if self.verbose > 0:
            print(f"\n=== 训练统计 ===")
            print(f"总回合数: {len(self.episode_rewards)}")
            print(f"平均奖励: {np.mean(self.episode_rewards) if self.episode_rewards else 0:.2f}")
            print(f"平均回合长度: {np.mean(self.episode_lengths) if self.episode_lengths else 0:.2f}")
            print(f"成功率: {np.mean(self.success_rates) if self.success_rates else 0:.2%}")
            print(f"损失记录数: {len(self.losses)}")

def plot_training_results(callback, save_dir="plots"):
    """绘制训练结果的三个图表"""
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 检查数据是否为空
    has_episode_data = len(callback.episode_rewards) > 0
    has_loss_data = len(callback.losses) > 0
    
    print(f"数据统计: 回合数={len(callback.episode_rewards)}, 损失记录数={len(callback.losses)}")
    
    if not has_episode_data and not has_loss_data:
        print("警告: 没有收集到任何训练数据！")
        # 创建一个空的图表说明情况
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.text(0.5, 0.5, '没有收集到训练数据\n请检查回调函数配置', 
                ha='center', va='center', fontsize=16, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f'训练结果 - {timestamp}')
        plt.tight_layout()
        plot_path = os.path.join(save_dir, f"training_results_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        return plot_path, None
    
    # 创建一个包含四个子图的图形
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'PPO训练结果 - {timestamp}', fontsize=16)
    
    # 图1: 平均每回合奖励变化
    if has_episode_data:
        # 计算移动平均
        window_size = min(10, len(callback.episode_rewards) // 3 + 1)
        if len(callback.episode_rewards) >= window_size and window_size > 1:
            moving_avg = np.convolve(callback.episode_rewards, 
                                   np.ones(window_size)/window_size, mode='valid')
            axes[0, 0].plot(callback.episode_rewards, alpha=0.6, color='lightblue', 
                          label='原始奖励', marker='o', markersize=3)
            axes[0, 0].plot(range(window_size-1, len(callback.episode_rewards)), 
                          moving_avg, color='blue', linewidth=2, 
                          label=f'{window_size}回合移动平均')
        else:
            axes[0, 0].plot(callback.episode_rewards, color='blue', 
                          label='回合奖励', marker='o', markersize=4)
        
        axes[0, 0].set_title('平均每回合奖励变化')
        axes[0, 0].set_xlabel('回合数')
        axes[0, 0].set_ylabel('奖励')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    else:
        axes[0, 0].text(0.5, 0.5, '没有回合奖励数据', ha='center', va='center')
        axes[0, 0].set_title('平均每回合奖励变化')
    
    # 图2: 损失函数变化
    if has_loss_data:
        # 只显示前面一部分损失数据，避免图表过于密集
        max_points = 1000
        if len(callback.losses) > max_points:
            step = len(callback.losses) // max_points
            loss_data = callback.losses[::step]
            timestep_data = callback.timesteps[:len(loss_data)]
        else:
            loss_data = callback.losses
            timestep_data = callback.timesteps[:len(loss_data)]
            
        axes[0, 1].plot(timestep_data, loss_data, color='red', linewidth=1, alpha=0.8)
        axes[0, 1].set_title('损失函数变化')
        axes[0, 1].set_xlabel('训练步数')
        axes[0, 1].set_ylabel('损失值')
        axes[0, 1].grid(True, alpha=0.3)
        # 使用科学计数法显示y轴
        axes[0, 1].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    else:
        axes[0, 1].text(0.5, 0.5, '没有损失数据', ha='center', va='center')
        axes[0, 1].set_title('损失函数变化')
    
    # 图3: 成功率变化
    if has_episode_data and callback.success_rates:
        # 计算成功率的移动平均
        window_size = min(5, len(callback.success_rates) // 2 + 1)
        if len(callback.success_rates) >= window_size and window_size > 1:
            success_moving_avg = np.convolve(callback.success_rates, 
                                           np.ones(window_size)/window_size, mode='valid')
            axes[1, 0].plot(callback.success_rates, alpha=0.4, color='lightgreen', 
                          label='原始成功率', marker='s', markersize=2)
            axes[1, 0].plot(range(window_size-1, len(callback.success_rates)), 
                          success_moving_avg, color='green', linewidth=2,
                          label=f'{window_size}回合移动平均')
        else:
            axes[1, 0].plot(callback.success_rates, color='green', 
                          marker='s', markersize=3, label='成功率')
        
        axes[1, 0].set_title('成功率变化')
        axes[1, 0].set_xlabel('回合数')
        axes[1, 0].set_ylabel('成功率')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, '没有成功率数据', ha='center', va='center')
        axes[1, 0].set_title('成功率变化')
    
    # 图4: 回合长度变化
    if has_episode_data:
        window_size = min(10, len(callback.episode_lengths) // 3 + 1)
        if len(callback.episode_lengths) >= window_size and window_size > 1:
            length_moving_avg = np.convolve(callback.episode_lengths, 
                                          np.ones(window_size)/window_size, mode='valid')
            axes[1, 1].plot(callback.episode_lengths, alpha=0.5, color='lightcoral', 
                          label='原始长度', marker='^', markersize=2)
            axes[1, 1].plot(range(window_size-1, len(callback.episode_lengths)), 
                          length_moving_avg, color='darkred', linewidth=2, 
                          label=f'{window_size}回合移动平均')
        else:
            axes[1, 1].plot(callback.episode_lengths, color='darkred', 
                          label='回合长度', marker='^', markersize=3)
        
        axes[1, 1].set_title('回合长度变化')
        axes[1, 1].set_xlabel('回合数')
        axes[1, 1].set_ylabel('步数')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, '没有回合长度数据', ha='center', va='center')
        axes[1, 1].set_title('回合长度变化')
    
    # 调整布局并保存
    plt.tight_layout()
    
    # 保存图片
    plot_path = os.path.join(save_dir, f"training_results_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"训练结果图表已保存至: {plot_path}")
    
    # 保存数据到JSON文件
    data_path = os.path.join(save_dir, f"training_data_{timestamp}.json")
    training_data = {
        'episode_rewards': [float(r) for r in callback.episode_rewards],
        'episode_lengths': [int(l) for l in callback.episode_lengths],
        'losses': [float(l) for l in callback.losses],
        'success_rates': [float(s) for s in callback.success_rates],
        'timesteps': [int(t) for t in callback.timesteps],
        'episode_count': int(getattr(callback, 'episode_count', len(callback.episode_rewards)))
    }
    
    with open(data_path, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    print(f"训练数据已保存至: {data_path}")
    
    # 显示图表
    plt.show()
    
    return plot_path, data_path

def main():
    """
    使用 PPO 算法训练一个 Agent 来解决 YardEnv 环境中的调度问题。
    """
    print("--- 开始 PPO 训练 ---")

    # --- 1. 环境设置 ---
    # 在开始训练前，检查自定义环境是否符合 aPI 标准，这是一个好习惯。
    # 如果环境不兼容，检查程序会抛出错误。
    print("正在检查环境兼容性...")
    env_instance = YardEnv()
    check_env(env_instance, warn=True)
    print("环境兼容性检查通过！")

    # 为了训练，我们通常使用“向量化”的环境，它可以并行运行多个环境实例以加速数据收集。
    # DummyVecEnv 是最简单的实现，它在一个进程中按顺序运行多个环境。
    env = DummyVecEnv([lambda: ActionMasker(YardEnv(render_mode=None), mask_fn)])


    # --- 2. 模型设置 ---
    # 'MlpPolicy' 策略适用于扁平化的观测空间。对于我们的字典(Dict)观测空间，
    # Stable Baselines3 会自动使用一个 CombinedExtractor 来处理不同部分的输入。
    # 我们也可以为 PPO 算法定义一些超参数。
    
    # 设置设备为GPU（如果可用）
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU设备名称: {torch.cuda.get_device_name(0)}")
    
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        device=device,  # 指定使用GPU
        learning_rate=0.0003,  # 降低学习率
        verbose=1,  # 设置为 1 以便在控制台看到训练进度
        policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])) # 定义一个较小的神经网络结构
    )

    # --- 3. 模型训练 ---
    # total_timesteps 指的是 Agent 与环境交互的总步数 (env.step())。
    # 我们先从一个较小的数值开始，以验证整个流程是否能跑通。
    
    # 创建自定义回调函数来收集训练指标
    metrics_callback = TrainingMetricsCallback(verbose=1)
    
    total_timesteps = 100000
    print(f"准备训练模型，总步数: {total_timesteps}...")
    model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=metrics_callback)
    print("模型训练完成。")
    
    # --- 4. 绘制训练结果 ---
    print("正在生成训练结果图表...")
    plot_path, data_path = plot_training_results(metrics_callback)
    print("训练结果可视化完成。")

    # --- 5. 保存模型 ---
    # 训练好的模型可以被保存下来，以便未来加载和使用。
    save_path = os.path.join("models", "ppo_yard_model.zip")
    model.save(save_path)
    print(f"模型已保存至: {save_path}")

    # --- 6. (可选) 评估训练好的模型 ---
    print("--- 开始评估训练好的模型 ---")
    obs = env.reset()
    for i in range(5000): # 运行 5000 步进行测试
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        if dones.any():
            print("一个评估回合结束。")
            obs = env.reset()


    env.close()
    print("--- 训练与评估全部完成 ---")
    print(f"训练结果图表保存在: {plot_path}")
    print(f"训练数据保存在: {data_path}")


if __name__ == "__main__":
    main()