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
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback


def mask_fn(env: gym.Env) -> np.ndarray:
    """一个辅助函数，用于从环境中提取动作掩码。"""
    return env.action_mask()


class RewardLoggingCallback(BaseCallback):
    """将关键指标写入TensorBoard（英文+中文标签），便于观察奖励与队列相关信号。"""
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", None)

        def record_both(en_key: str, zh_key: str, value: float):
            try:
                self.logger.record(en_key, float(value))
            except Exception:
                pass
            try:
                self.logger.record(zh_key, float(value))
            except Exception:
                pass

        # 记录每步奖励的均值/方差（向量环境会有多个）
        if rewards is not None:
            try:
                mean_v = float(np.mean(rewards))
                std_v = float(np.std(rewards))
                record_both("reward/step_mean", "奖励/每步均值", mean_v)
                record_both("reward/step_std", "奖励/每步标准差", std_v)
            except Exception:
                pass

        # 从 info 聚合环境指标（均值）
        def record_mean_info(key: str, en_key: str, zh_key: str):
            if isinstance(infos, list):
                vals = [info[key] for info in infos if isinstance(info, dict) and key in info]
                if len(vals) > 0:
                    try:
                        value = float(np.mean(vals))
                        record_both(en_key, zh_key, value)
                    except Exception:
                        pass
            elif isinstance(infos, dict) and key in infos:
                try:
                    value = float(infos[key])
                    record_both(en_key, zh_key, value)
                except Exception:
                    pass

        record_mean_info("pending_tasks_count", "env/pending_tasks", "环境/待处理任务数")
        record_mean_info("completed_tasks_count", "env/completed_tasks", "环境/完成任务数")
        record_mean_info("total_task_wait_time", "env/total_task_wait_time", "环境/总任务等待时间")
        record_mean_info("total_crane_move_time", "env/total_crane_move_time", "环境/总场桥移动时间")
        record_mean_info("total_crane_wait_time", "env/total_crane_wait_time", "环境/总场桥等待时间")
        record_mean_info("simulation_time", "env/simulation_time", "环境/仿真时间")
        # 势函数塑形
        record_mean_info("reward_shaping", "reward/shaping", "奖励/塑形项")
        record_mean_info("reward_window", "reward/window_penalty", "奖励/时间窗惩罚")
        record_mean_info("avg_wait_norm", "env/avg_wait_norm", "环境/平均等待_归一化")
        record_mean_info("queue_len_norm", "env/queue_len_norm", "环境/队列长度_归一化")

        # 记录回合级别（来自 Monitor 的 episode 信息）——中文别名
        try:
            if isinstance(infos, list):
                for info in infos:
                    if isinstance(info, dict) and "episode" in info:
                        ep = info["episode"]
                        if "r" in ep:
                            record_both("episode/reward", "回合/总奖励", float(ep["r"]))
                        if "l" in ep:
                            record_both("episode/length", "回合/步数", float(ep["l"]))
            elif isinstance(infos, dict) and "episode" in infos:
                ep = infos["episode"]
                if "r" in ep:
                    record_both("episode/reward", "回合/总奖励", float(ep["r"]))
                if "l" in ep:
                    record_both("episode/length", "回合/步数", float(ep["l"]))
        except Exception:
            pass

        return True


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
    base_envs = [lambda: ActionMasker(YardEnv(render_mode=None), mask_fn) for _ in range(n_envs)]
    env = DummyVecEnv(base_envs)
    # 关键：包装 VecMonitor，让SB3自动记录 rollout/ep_rew_mean 与 ep_len_mean
    env = VecMonitor(env)

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
    model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=RewardLoggingCallback())
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