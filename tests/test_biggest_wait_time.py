import os
import sys
import json
import gymnasium as gym
import numpy as np

# 将项目根目录添加到 Python 路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.env_1022 import YardEnv
from src.environment.dataclasses import Task
from sb3_contrib.common.wrappers import ActionMasker


def mask_fn(env: gym.Env) -> np.ndarray:
    return env.action_mask()


def run(use_static_data: bool = True):
    """
    运行“优先选择等待时间最长的任务”的贪心策略。
    - use_static_data=True：使用静态任务 JSON 注入环境。
    - use_static_data=False：使用环境随机生成任务。
    """
    print("--- 开始评估最大等待时间策略 (env_1022)，静态数据: {} ---".format(use_static_data))

    static_tasks = None
    if use_static_data:
        data_path = os.path.join(os.path.dirname(__file__), "static_tasks_env1022.json")
        with open(data_path, "r", encoding="utf-8") as f:
            raw_tasks = json.load(f)
        static_tasks = [Task(**t) for t in raw_tasks]

    env = YardEnv(render_mode='human', static_tasks=static_tasks)
    env = ActionMasker(env, mask_fn)

    obs, info = env.reset()
    terminated = False
    truncated = False
    total_reward = 0.0

    print("开始运行最大等待时间贪心策略...")
    for _ in range(2000):
        if terminated or truncated:
            break
        action_masks = env.action_masks()
        valid_actions = np.nonzero(action_masks)[0]
        if len(valid_actions) == 0:
            break

        if len(valid_actions) == 1 and valid_actions[0] == env.env.max_tasks_in_obs:
            action = valid_actions[0]
        else:
            best_action = None
            max_wait = -1.0
            current_time = env.env.current_time
            current_obs = env.env._get_observation(env.env.crane_to_command)

            for action_idx in valid_actions:
                if action_idx < env.env.max_tasks_in_obs:
                    task_info = current_obs["task_list"][action_idx]
                    task_available_time = task_info[2]
                    if task_available_time > 0:
                        wait_time = max(0.0, current_time - task_available_time)
                        if wait_time > max_wait:
                            max_wait = wait_time
                            best_action = action_idx
            if best_action is not None:
                action = best_action
            else:
                wait_action = env.env.max_tasks_in_obs
                if wait_action in valid_actions:
                    action = wait_action
                else:
                    action = valid_actions[0]

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    print("评估回合结束。")
    print(f"总奖励: {total_reward}")

    print("\n--- 性能指标 ---")
    print(f"任务总等待时间: {info['total_task_wait_time']:.2f}s")
    print(f"场桥移动次数: {info['crane_move_count']}")
    print(f"场桥移动总时间: {info['total_crane_move_time']:.2f}s")
    print(f"场桥等待总时间: {info['total_crane_wait_time']:.2f}s")
    print(f"已完成任务数: {info['completed_tasks_count']}")
    print(f"总仿真时间: {info['simulation_time']:.2f}s")
    if 'termination_reason' in info:
        print(f"终止原因: {info['termination_reason']}")

    save_name = "最大等待时间贪心轨迹图.png" if use_static_data else "随机生成任务最大等待时间轨迹图.png"
    env.env.plot_crane_trajectories(save_path=save_name)
    print(f"轨迹图已保存至 {save_name}")

    env.close()


def main():
    use_static = True
    for arg in sys.argv[1:]:
        if arg.startswith("--use_static="):
            val = arg.split("=", 1)[1].strip().lower()
            use_static = (val in {"1", "true", "yes", "y"})
    run(use_static)


if __name__ == "__main__":
    main()
