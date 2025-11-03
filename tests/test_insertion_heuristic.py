#!/usr/bin/env python3
"""
使用插入启发式 (Insertion Heuristic) 算法来解决动态调度问题。
该算法通过迭代地将任务插入到现有序列中的最佳位置来构建一个高质量的执行计划。
"""
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


def calculate_sequence_cost(sequence: list[Task], env: YardEnv) -> float:
    """
    计算一个给定的任务序列的总执行成本（总等待时间）。
    这是插入启发式算法中的核心评估函数。
    """
    internal_env = env.env
    crane_id = internal_env.crane_to_command
    crane = internal_env.cranes[crane_id]
    current_time = internal_env.current_time
    
    total_wait_time = 0.0
    temp_crane_location = crane.location
    temp_current_time = current_time

    for task in sequence:
        travel_time = abs(temp_crane_location - task.location) / internal_env.crane_speed
        
        # 任务的等待时间 = (起重机到达时间) - (任务可用时间)
        arrival_at_task_time = temp_current_time + travel_time
        wait_time = max(0, arrival_at_task_time - task.available_time)
        total_wait_time += wait_time

        # 更新时间和起重机位置以进行下一个任务的计算
        temp_current_time = arrival_at_task_time + task.execution_time
        temp_crane_location = task.location
        
    return total_wait_time


def plan_tasks_with_insertion_heuristic(candidate_tasks: list[Task], env: YardEnv) -> list[Task]:
    """
    使用插入启发式算法为给定的候选任务制定一个执行计划（序列）。
    """
    if not candidate_tasks:
        return []

    # 1. 初始化：选择一个任务作为初始序列。
    #    这里我们选择已经等待时间最长的任务作为起点，以体现紧迫性。
    initial_task = max(candidate_tasks, key=lambda t: env.env.current_time - t.available_time)
    sequence = [initial_task]
    
    remaining_tasks = [t for t in candidate_tasks if t != initial_task]
    
    # 2. 迭代插入：将其余任务逐一插入到序列中的最佳位置。
    for task_to_insert in remaining_tasks:
        best_cost = float('inf')
        best_sequence = []

        # 尝试将任务插入到序列的每个可能位置（包括开头和结尾）
        for i in range(len(sequence) + 1):
            temp_sequence = sequence[:i] + [task_to_insert] + sequence[i:]
            current_cost = calculate_sequence_cost(temp_sequence, env)
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_sequence = temp_sequence
        
        sequence = best_sequence

    return sequence


def select_action_from_plan(valid_actions: np.ndarray, env: YardEnv) -> int:
    """
    根据插入启发式算法制定的计划，选择要执行的下一个动作。
    """
    wait_action = env.env.max_tasks_in_obs
    internal_env = env.env
    visible_tasks_map = {i: task for i, task in enumerate(internal_env._last_visible_tasks) if task.location > 0}

    # 过滤掉等待动作，只在任务中选择
    candidate_action_indices = [a for a in valid_actions if a != wait_action]

    if not candidate_action_indices:
        return wait_action if wait_action in valid_actions else valid_actions[0]

    # 将有效的动作索引转换为 Task 对象列表
    candidate_tasks = [visible_tasks_map[idx] for idx in candidate_action_indices if idx in visible_tasks_map]

    if not candidate_tasks:
        # 如果没有有效的任务对象，则执行等待或默认动作
        return wait_action if wait_action in valid_actions else valid_actions[0]

    # 使用插入启发式算法规划任务
    planned_sequence = plan_tasks_with_insertion_heuristic(candidate_tasks, env)
    
    if not planned_sequence:
        return wait_action if wait_action in valid_actions else valid_actions[0]

    # 从计划中选择第一个任务
    next_task_to_execute = planned_sequence[0]
    
    # 找到这个任务对应的原始动作索引
    for idx, task in visible_tasks_map.items():
        if task == next_task_to_execute:
            return idx
            
    # 如果找不到，则返回等待动作
    return wait_action


def run(use_static_data: bool = True):
    """
    使用插入启发式策略在 YardEnv 环境中运行。
    """
    print("--- 开始评估插入启发式 (Insertion Heuristic) 策略，静态数据: {} ---".format(use_static_data))

    static_tasks = None
    if use_static_data:
        data_path = os.path.join(os.path.dirname(__file__), "static_tasks_env.json")
        with open(data_path, "r", encoding="utf-8") as f:
            raw_tasks = json.load(f)
        static_tasks = [Task(**t) for t in raw_tasks]

    env = YardEnv(render_mode='human', static_tasks=static_tasks)
    env = ActionMasker(env, mask_fn)

    obs, info = env.reset()
    terminated = False
    truncated = False
    total_reward = 0.0

    print("开始运行插入启发式策略...")
    for _ in range(2000):  # 增加步数上限
        if terminated or truncated:
            break
        
        action_masks = env.action_masks()
        valid_actions = np.nonzero(action_masks)[0]
        
        if len(valid_actions) == 0:
            break

        action = select_action_from_plan(valid_actions, env)
        
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
    print(f"已完成任务数: {info['completed_tasks_count']}")
    print(f"总仿真时间: {info['simulation_time']:.2f}s")
    if 'termination_reason' in info:
        print(f"终止原因: {info['termination_reason']}")

    # 保存轨迹图
    save_name = "InsertionHeuristic策略轨迹图_静态.png" if use_static_data else "InsertionHeuristic策略轨迹图_动态.png"
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