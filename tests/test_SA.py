#!/usr/bin/env python3
"""
使用模拟退火 (Simulated Annealing, SA) 启发式算法来解决动态调度问题。
"""
import os
import sys
import json
import gymnasium as gym
import numpy as np
import math
import random
import time

# 将项目根目录添加到 Python 路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.env_penalty import YardEnv
from src.environment.dataclasses import Task
from sb3_contrib.common.wrappers import ActionMasker


def mask_fn(env: gym.Env) -> np.ndarray:
    """一个辅助函数，用于从环境中提取动作掩码。"""
    return env.action_mask()


def calculate_sequence_cost(sequence: list[Task], env: YardEnv) -> float:
    """
    计算一个给定的任务序列的总执行成本（总等待时间）。
    这是模拟退火算法中用于评估一个完整“路径”或“计划”的成本函数。
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


def select_action_sa(valid_actions: np.ndarray, env: YardEnv) -> int:
    """
    使用模拟退火 (SA) 算法来规划一个任务执行序列，并返回序列中的第一个动作。
    SA 在由任务排列构成的解空间中搜索，以找到总等待时间最小的序列。
    """
    wait_action = env.action_space.n - 1
    internal_env = env.env
    # 创建一个从动作索引到完整任务对象的映射
    visible_tasks_map = {i: task for i, task in enumerate(internal_env._last_visible_tasks) if task.location > 0}

    # 过滤掉等待动作，只考虑实际的任务
    candidate_action_indices = [a for a in valid_actions if a != wait_action]

    if not candidate_action_indices:
        return wait_action if wait_action in valid_actions else valid_actions[0]

    # 将有效的动作索引转换为 Task 对象列表
    candidate_tasks = [visible_tasks_map[idx] for idx in candidate_action_indices if idx in visible_tasks_map]

    if not candidate_tasks:
        return wait_action if wait_action in valid_actions else valid_actions[0]
    
    if len(candidate_tasks) == 1:
        # 如果只有一个可选任务，直接执行
        return candidate_action_indices[0]

    # SA 参数
    initial_temp = 5000.0
    final_temp = 0.1
    alpha = 0.99  # 冷却速率
    iterations_per_temp = 20  # 在每个温度下迭代的次数

    # 初始解：随机打乱任务顺序作为一个初始“路径”
    current_solution = random.sample(candidate_tasks, len(candidate_tasks))
    best_solution = current_solution
    
    current_cost = calculate_sequence_cost(current_solution, env)
    best_cost = current_cost
    
    current_temp = initial_temp

    while current_temp > final_temp:
        for _ in range(iterations_per_temp):
            # 生成邻居解：随机交换序列中的两个任务，这是路径规划中常见的邻域操作
            neighbor_solution = list(current_solution)
            n = [task.id for task in neighbor_solution]
            if len(neighbor_solution) > 1:
                i, j = random.sample(range(len(neighbor_solution)), 2)
                neighbor_solution[i], neighbor_solution[j] = neighbor_solution[j], neighbor_solution[i]

            neighbor_cost = calculate_sequence_cost(neighbor_solution, env)
            
            cost_diff = neighbor_cost - current_cost

            # 如果邻居解更好，或者根据Metropolis准则接受更差的解
            if cost_diff < 0 or random.random() < math.exp(-cost_diff / current_temp):
                current_solution = neighbor_solution
                current_cost = neighbor_cost
            
            # 更新全局最优解
            if current_cost < best_cost:
                best_solution = current_solution
                best_cost = current_cost
        
        # 冷却
        current_temp *= alpha

    # 从找到的最佳序列（路径）中，选择第一个任务来执行
    next_task_to_execute = best_solution[0]
    
    # 找到这个任务对应的原始动作索引
    for idx, task in visible_tasks_map.items():
        if task == next_task_to_execute:
            return idx
            
    # 如果发生意外，返回等待动作
    return wait_action


def run(use_static_data: bool = True, enable_render: bool = False, save_plot: bool = True, output_metrics: bool = True):
    """
    使用模拟退火策略在 YardEnv 环境中运行。
    """
    print("--- 开始评估模拟退火 (SA) 策略，静态数据: {} ---".format(use_static_data))

    static_tasks = None
    if use_static_data:
        data_path = os.path.join(os.path.dirname(__file__), "static_tasks_env.json")
        with open(data_path, "r", encoding="utf-8") as f:
            raw_tasks = json.load(f)
        static_tasks = [Task(**t) for t in raw_tasks]

    render_mode = 'human' if enable_render else None
    env = YardEnv(render_mode=render_mode, static_tasks=static_tasks)
    env = ActionMasker(env, mask_fn)

    obs, info = env.reset()
    terminated = False
    truncated = False
    total_reward = 0.0

    if output_metrics:
        print("开始运行模拟退火策略...")
    loop_start = time.perf_counter()
    # 增加步数上限，避免无限循环
    for _ in range(2000):
        if terminated or truncated:
            break
        
        action_masks = env.action_masks()
        valid_actions = np.nonzero(action_masks)[0]
        
        if len(valid_actions) == 0:
            break

        action = select_action_sa(valid_actions, env)
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
    loop_end = time.perf_counter()
    agent_loop_time = loop_end - loop_start

    if output_metrics:
        print("评估回合结束。")
        print(f"总奖励: {total_reward}")
        print(f"算法运行时间: {agent_loop_time:.6f}s")

    # 显示性能指标
    if output_metrics:
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
    if save_plot:
        save_name = "SA策略轨迹图_静态.png" if use_static_data else "SA策略轨迹图_动态.png"
        env.env.plot_crane_trajectories(save_path=save_name)
        if output_metrics:
            print(f"轨迹图已保存至 {save_name}")

    env.close()


def main():
    use_static = True
    enable_render = True
    save_plot = True
    output_metrics = True
    for arg in sys.argv[1:]:
        if arg.startswith("--use_static="):
            val = arg.split("=", 1)[1].strip().lower()
            use_static = (val in {"1", "true", "yes", "y"})
        elif arg.startswith("--render="):
            val = arg.split("=", 1)[1].strip().lower()
            enable_render = (val in {"1", "true", "yes", "y"})
        elif arg.startswith("--save_plot="):
            val = arg.split("=", 1)[1].strip().lower()
            save_plot = (val in {"1", "true", "yes", "y"})
        elif arg.startswith("--output_metrics="):
            val = arg.split("=", 1)[1].strip().lower()
            output_metrics = (val in {"1", "true", "yes", "y"})
    run(use_static_data=use_static, enable_render=enable_render, save_plot=save_plot, output_metrics=output_metrics)


if __name__ == "__main__":
    main()