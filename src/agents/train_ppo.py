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
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])

        def log_from_info(info: dict):
            ep = info.get("episode")
            if isinstance(ep, dict) and "r" in ep:
                try:
                    # 记录每回合总奖励
                    self.logger.record("回合/总奖励", float(ep["r"]))

                    # 新增：计算并记录任务平均等待时间
                    total_wait_time = info.get("total_task_wait_time")
                    completed_tasks = info.get("completed_tasks_count")

                    if total_wait_time is not None and completed_tasks is not None and completed_tasks > 0:
                        average_wait_time = total_wait_time / completed_tasks
                        self.logger.record("回合/任务平均等待时间", average_wait_time)

                except Exception as e:
                    # 增加打印异常，方便调试
                    print(f"Error in RewardLoggingCallback: {e}")

        if isinstance(infos, list):
            for info in infos:
                if isinstance(info, dict):
                    log_from_info(info)
        elif isinstance(infos, dict):
            log_from_info(infos)

        return True

def train_ppo_env1022(
    timesteps: int = 1_000_000,
    n_envs: int = 8,
    tb_dir: str = "./ppo_tensorboard/",
    save_path: str = os.path.join("models", "ppo_env1022_model.zip"),
    skip_check: bool = True,
    env_kwargs: dict | None = None,
    policy_kwargs: dict | None = None,
    model_kwargs: dict | None = None,
    evaluate_steps: int = 1000,
    seed: int = 42,
):
    """
    训练并评估基于 MaskablePPO 的智能体（适配 env）。不使用 CLI，提供函数式参数。

    参数说明：
    - timesteps: 训练总步数。
    - n_envs: 并行环境数量（使用 DummyVecEnv）。
    - tb_dir: TensorBoard 日志目录。
    - save_path: 模型保存路径（.zip）。
    - skip_check: 是否跳过 `check_env` 环境校验（建议 True，避免掩码动作兼容性问题）。
    - env_kwargs: 传递给 `YardEnv` 的构造参数字典，例如 `{"static_tasks": tasks}`。
    - policy_kwargs: 传递给策略的结构参数（如网络结构）。
    - model_kwargs: 覆盖 PPO 的超参数字典（如学习率、n_steps、batch_size 等）。
    - evaluate_steps: 训练完成后评估的步数。
    - seed: 随机种子。

    返回：
    - model: 训练完成的 `MaskablePPO` 模型实例。
    - save_path: 最终保存的模型路径。

    使用示例：
        from src.agents.train_ppo import train_ppo_env1022
        model, path = train_ppo_env1022(timesteps=200_000, n_envs=4,
                                        env_kwargs={"render_mode": None})
    """
    print("--- 开始 PPO 训练 (env) ---")

    env_kwargs = env_kwargs or {}

    # 可选：环境校验
    if not skip_check:
        print("正在检查环境兼容性...")
        env_instance = YardEnv(**env_kwargs)
        try:
            check_env(env_instance, warn=True)
            print("环境兼容性检查通过！")
        except Exception as e:
            print(f"环境校验失败: {e}，跳过并继续训练。")

    # 并行向量环境
    # 注意：避免与 env_kwargs 中的 render_mode 重复传参
    base_envs = [lambda: ActionMasker(YardEnv(**env_kwargs), mask_fn) for _ in range(n_envs)]
    env = DummyVecEnv(base_envs)
    env = VecMonitor(env)

    # 模型超参数（可被外部覆盖）
    default_policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
    if policy_kwargs is None:
        policy_kwargs = default_policy_kwargs

    cfg = dict(
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=512,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        n_epochs=10,
        target_kl=0.05,
        seed=seed,
    )
    if model_kwargs:
        cfg.update(model_kwargs)

    # 构建模型
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log=tb_dir,
        policy_kwargs=policy_kwargs,
        **cfg,
    )

    # 训练
    total_timesteps = int(timesteps)
    print(f"准备训练模型，总步数: {total_timesteps}...")
    model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=RewardLoggingCallback())
    print("模型训练完成。")

    # 保存模型
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"模型已保存至: {save_path}")

    # 评估（带动作掩码）
    print("--- 开始评估训练好的模型 ---")
    obs = env.reset()
    for _ in range(int(evaluate_steps)):
        action_masks_list = env.env_method("action_masks")
        action, _states = model.predict(obs, deterministic=True, action_masks=action_masks_list)
        obs, rewards, dones, infos = env.step(action)
        if dones.any():
            obs = env.reset()
    env.close()
    print("--- 训练与评估全部完成 ---")

    return model, save_path


if __name__ == "__main__":
    # 直接调用训练函数，不使用命令行参数
    model, path = train_ppo_env1022(
        timesteps=3_000_000,
        n_envs=4,
        tb_dir="./ppo_tensorboard/",
        save_path=os.path.join("models", "ppo_env_model.zip"),
        skip_check=True,
        env_kwargs={"render_mode": None},
        evaluate_steps=1000,
        seed=42,
    )
    print(f"训练完成，模型保存在: {path}")