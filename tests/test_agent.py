import os
import sys
import json
import gymnasium as gym
import numpy as np

# 将项目根目录添加到 Python 路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.env import YardEnv
from src.environment.dataclasses import Task
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker


def mask_fn(env: gym.Env) -> np.ndarray:
    """一个辅助函数，用于从环境中提取动作掩码。"""
    return env.action_mask()


def run(use_static_data: bool = True):
    """
    使用训练好的 PPO 模型进行评估。
    - use_static_data=True: 从 'static_tasks_env.json' 加载静态任务。
    - use_static_data=False: 使用环境内置的随机任务生成器。
    """
    print(f"--- 开始评估 PPO 模型 (env)，静态数据: {use_static_data} ---")

    static_tasks = None
    if use_static_data:
        data_path = os.path.join(os.path.dirname(__file__), "static_tasks_env.json")
        try:
            with open(data_path, "r", encoding="utf-8") as f:
                raw_tasks = json.load(f)
            static_tasks = [Task(**t) for t in raw_tasks]

            time = 0
            for task in static_tasks:
                time += task.execution_time
            print(f"静态任务总服务时间: {time:.2f}s")
            print(f"已从 {data_path} 加载 {len(static_tasks)} 个静态任务。")
        except FileNotFoundError:
            print(f"错误：静态任务文件未找到 at {data_path}")
            return

    # 创建环境，如果提供了静态任务，则直接注入
    env = YardEnv(render_mode='human', static_tasks=static_tasks)
    env = ActionMasker(env, mask_fn)

    # 加载模型
    model_path = os.path.join("models", "ppo_env_model.zip")
    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件 at {model_path}")
        print("请先运行 src/agents/train_ppo.py 脚本来训练和保存模型。")
        env.close()
        return

    model = MaskablePPO.load(model_path, env=env)
    print(f"模型已从 {model_path} 加载。")

    obs, info = env.reset()
    terminated = False
    truncated = False
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
    print(f"任务总等待时间: {info.get('total_task_wait_time', 0.0):.2f}s")
    print(f"场桥移动次数: {info.get('crane_move_count', 0)}")
    print(f"场桥移动总时间: {info.get('total_crane_move_time', 0.0):.2f}s")
    print(f"场桥等待总时间: {info.get('total_crane_wait_time', 0.0):.2f}s")
    print(f"场桥闲置总时间: {info.get('total_crane_idle_time', 0.0):.2f}s")
    print(f"已完成任务数: {info.get('completed_tasks_count', 0)}")
    print(f"总仿真时间: {info.get('simulation_time', 0.0):.2f}s")
    if 'termination_reason' in info:
        print(f"停止原因: {info['termination_reason']}")


    # 根据数据源动态命名并保存轨迹图
    save_name = "静态任务智能体轨迹图.png" if use_static_data else "随机任务智能体轨迹图.png"
    env.env.plot_crane_trajectories(save_path=save_name)
    print(f"轨迹图已保存至 {save_name}")

    env.close()


def main():
    """解析命令行参数并运行评估。"""
    use_static = True  # 默认使用静态数据
    for arg in sys.argv[1:]:
        if arg.startswith("--use_static="):
            val = arg.split("=", 1)[1].strip().lower()
            use_static = (val in {"1", "true", "yes", "y"})
    run(use_static)


if __name__ == "__main__":
    main()
