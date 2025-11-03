import os
import sys
import json
import gymnasium as gym
import numpy as np

# 将项目根目录添加到 Python 路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.env import YardEnv
from src.environment.dataclasses import Task
from sb3_contrib.common.wrappers import ActionMasker


def mask_fn(env: gym.Env) -> np.ndarray:
    """一个辅助函数，用于从环境中提取动作掩码。"""
    return env.action_mask()


def run(use_static_data: bool = True):
    """
    运行随机策略测试。
    - use_static_data=True 时，从 tests/static_tasks_env.json 加载静态任务并注入环境。
    - use_static_data=False 时，使用环境自身的随机任务生成机制。
    """
    print("--- 开始评估随机策略 (env)，静态数据: {} ---".format(use_static_data))

    static_tasks = None
    if use_static_data:
        data_path = os.path.join(os.path.dirname(__file__), "static_tasks_env.json")
        with open(data_path, "r", encoding="utf-8") as f:
            raw_tasks = json.load(f)
        static_tasks = [Task(**t) for t in raw_tasks]

    # 创建环境：通过构造参数传入静态任务（若启用）
    env = YardEnv(render_mode='human', static_tasks=static_tasks)
    env = ActionMasker(env, mask_fn)

    obs, info = env.reset()
    terminated = False
    truncated = False
    total_reward = 0.0

    print("开始运行随机策略...")
    while not terminated and not truncated:
        action_masks = env.action_masks()
        valid_actions = np.nonzero(action_masks)[0]
        if len(valid_actions) == 0:
            break
        action = np.random.choice(valid_actions)
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
    print(f"场桥闲置总时间: {info['total_crane_idle_time']:.2f}s")
    print(f"已完成任务数: {info['completed_tasks_count']}")
    print(f"总仿真时间: {info['simulation_time']:.2f}s")
    print(f"步数: {info['step_count']}")
    print(f"停止原因: {info['termination_reason']}")


    # 保存轨迹图
    save_name = "静态任务随机选择轨迹图.png" if use_static_data else "随机生成任务随机选择轨迹图.png"
    env.env.plot_crane_trajectories(save_path=save_name)
    print(f"轨迹图已保存至 {save_name}")

    env.close()


def main():
    # 简单的命令行开关支持：--use_static=true/false
    use_static = True
    for arg in sys.argv[1:]:
        if arg.startswith("--use_static="):
            val = arg.split("=", 1)[1].strip().lower()
            use_static = (val in {"1", "true", "yes", "y"})
    run(use_static)


if __name__ == "__main__":
    main()
