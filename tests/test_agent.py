import os
import sys
import json
import types
import gymnasium as gym
import numpy as np

# 将项目根目录添加到 Python 路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.env_1022 import YardEnv, Event, EventType
from src.environment.dataclasses import Task
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker


def mask_fn(env: gym.Env) -> np.ndarray:
    """一个辅助函数，用于从环境中提取动作掩码。"""
    return env.action_mask()


def patch_static_generator(inner_env: YardEnv, static_tasks: list[Task]) -> None:
    """将 env_1022 的任务生成函数替换为静态数据驱动的版本。"""
    inner_env.static_tasks = sorted(static_tasks, key=lambda t: t.available_time)
    inner_env.static_task_index = 0

    def _generate_new_tasks_static(self: YardEnv) -> None:
        window_start = self.current_time
        window_end = window_start + self.task_interval
        while self.static_task_index < len(self.static_tasks):
            t = self.static_tasks[self.static_task_index]
            if t.available_time >= window_end:
                break
            if window_start <= t.available_time < window_end:
                self.task_queue.append(t)
                self.static_task_index += 1
            else:
                if t.available_time < window_start:
                    self.static_task_index += 1
                else:
                    break
        next_window_time = window_start + self.task_interval
        if next_window_time < self.max_simulation_time:
            inner_env.event_queue.append(Event(time=next_window_time, type=EventType.TASK_GENERATION, data={}))
        else:
            self.task_generation_stopped = True

    inner_env._generate_new_tasks = types.MethodType(_generate_new_tasks_static, inner_env)


def main():
    """
    使用静态任务数据加载训练好的 PPO 模型，并在 YardEnv_1022 环境中运行它以进行可视化。
    """
    print("--- 开始使用静态任务数据评估训练好的 PPO 模型 (env_1022) ---")

    # 读取静态任务数据（env_1022 生成机制）
    data_path = os.path.join(os.path.dirname(__file__), "static_tasks_env1022.json")
    with open(data_path, "r", encoding="utf-8") as f:
        raw_tasks = json.load(f)
    static_tasks = [Task(**t) for t in raw_tasks]

    # 创建一个用于评估的环境实例，使用静态任务数据
    env = YardEnv(render_mode='human')  # 使用 'human' 模式以收集绘图数据
    env = ActionMasker(env, mask_fn)
    patch_static_generator(env.env, static_tasks)

    # 加载之前训练并保存的模型
    model_path = os.path.join("models", "ppo_env1022_model.zip")
    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件 at {model_path}")
        print("请先运行 src/agents/train_ppo.py 脚本来训练和保存模型。")
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
    
    # 显示性能指标
    print("\n--- 性能指标 ---")
    print(f"任务总等待时间: {info['total_task_wait_time']:.2f}s")
    print(f"场桥移动次数: {info['crane_move_count']}")
    print(f"场桥移动总时间: {info['total_crane_move_time']:.2f}s")
    print(f"场桥等待总时间: {info['total_crane_wait_time']:.2f}s")
    print(f"场桥闲置总时间: {info.get('total_crane_idle_time', 0.0):.2f}s")
    print(f"已完成任务数: {info['completed_tasks_count']}")
    print(f"总仿真时间: {info['simulation_time']:.2f}s")

    # 调用环境的绘图方法来生成并保存轨迹图
    env.env.plot_crane_trajectories(save_path="静态任务智能体轨迹图.png")
    print("轨迹图已保存至 静态任务智能体轨迹图.png")

    env.close()


if __name__ == "__main__":
    main()
