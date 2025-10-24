import os
import sys
import json
import gymnasium as gym
import numpy as np
from typing import List, Tuple

# 将项目根目录添加到 Python 路径中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src.environment.env_1022 import YardEnv
from src.environment.dataclasses import Task
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback


def mask_fn(env: gym.Env) -> np.ndarray:
    """一个辅助函数，用于从环境中提取动作掩码。"""
    return env.action_mask()


def load_static_tasks(data_path: str) -> List[Task]:
    """从 JSON 文件加载静态任务，转换为 Task 列表。"""
    if not os.path.isabs(data_path):
        data_path = os.path.join(PROJECT_ROOT, data_path)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"未找到静态任务文件: {data_path}")
    with open(data_path, "r", encoding="utf-8") as f:
        raw_tasks = json.load(f)
    tasks = [Task(**t) for t in raw_tasks]
    return tasks


class RewardLoggingCallback(BaseCallback):
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])

        def log_from_info(info: dict):
            ep = info.get("episode")
            if isinstance(ep, dict) and "r" in ep:
                try:
                    # 仅记录中文标签的每回合总奖励
                    self.logger.record("回合/总奖励", float(ep["r"]))
                except Exception:
                    pass

        if isinstance(infos, list):
            for info in infos:
                if isinstance(info, dict):
                    log_from_info(info)
        elif isinstance(infos, dict):
            log_from_info(infos)

        return True


def train_ppo_static(
    timesteps: int = 1_000_000,
    n_envs: int = 8,
    tb_dir: str = "./ppo_tensorboard/",
    save_path: str = os.path.join("models", "ppo_env1022_model_static.zip"),
    skip_check: bool = True,
    static_data_path: str = os.path.join("tests", "static_tasks_env1022.json"),
    env_kwargs: dict | None = None,
    policy_kwargs: dict | None = None,
    model_kwargs: dict | None = None,
    evaluate_steps: int = 1000,
    seed: int = 42,
) -> Tuple[MaskablePPO, str]:
    """
    使用静态任务数据训练 MaskablePPO（适配 env_1022）。
    - 从 `static_data_path` 加载任务并注入环境。
    - 仅记录中文标签“回合/总奖励”。
    """
    print("--- 开始 PPO 训练 (env_1022, 静态数据) ---")

    # 加载静态任务
    static_tasks = load_static_tasks(static_data_path)
    print(f"已加载静态任务数量: {len(static_tasks)}")

    env_kwargs = env_kwargs or {}
    env_kwargs.update({"static_tasks": static_tasks})

    # 可选：环境校验
    if not skip_check:
        print("正在检查环境兼容性（静态数据）...")
        env_instance = YardEnv(**env_kwargs)
        try:
            check_env(env_instance, warn=True)
            print("环境兼容性检查通过！")
        except Exception as e:
            print(f"环境校验失败: {e}，跳过并继续训练。")

    # 并行向量环境
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
    print("--- 开始评估训练好的模型（静态数据） ---")
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
    default_json = os.path.join(PROJECT_ROOT, "tests", "static_tasks_env1022.json")
    model, path = train_ppo_static(
        timesteps=1_000_000,
        n_envs=4,
        tb_dir="./ppo_tensorboard/",
        save_path=os.path.join("models", "ppo_env1022_model_static.zip"),
        skip_check=True,
        static_data_path=default_json,
        env_kwargs={"render_mode": None},
        evaluate_steps=1000,
        seed=42,
    )
    print(f"训练完成，模型保存在: {path}")