#!/usr/bin/env python3
"""
使用 A* (A-star) 搜索算法来解决动态调度问题。
A* 算法通过结合实际成本 (g(n)) 和启发式预估成本 (h(n)) 来寻找最优路径，
从而高效地在巨大的解空间中进行搜索。
"""
import os
import sys
import json
import heapq
import gymnasium as gym
import numpy as np
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
    这个函数作为 A* 算法中的 g(n)，即从起点到当前节点的实际成本。
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
        arrival_at_task_time = temp_current_time + travel_time
        wait_time = max(0, arrival_at_task_time - task.available_time)
        total_wait_time += wait_time

        temp_current_time = arrival_at_task_time + task.execution_time
        temp_crane_location = task.location
        
    return total_wait_time


def heuristic_cost_estimate(remaining_tasks: set[Task], env: YardEnv) -> float:
    """
    A* 算法的启发式函数 h(n)，用于预估完成剩余任务的成本。
    为了保证 A* 找到最优解，启发式函数必须是“可接受的”（admissible），
    即它永远不能高估未来的实际成本。
    
    最简单、最安全的启发式是 h(n) = 0。
    这虽然不会提供太多前瞻信息，但能保证找到最优解。
    """
    # 在这个实现中，我们使用最基础的 h(n) = 0。
    # 未来可以探索更智能的启发式，例如“剩余任务的最小可能移动时间之和”等。
    return 0.0


def plan_with_astar(candidate_tasks: list[Task], env: YardEnv) -> list[Task]:
    """
    使用 A* 算法为给定的候选任务寻找最优的执行序列。
    """
    if not candidate_tasks:
        return []

    # 创建一个从任务ID到任务对象的映射，方便之后查找
    task_map = {task.id: task for task in candidate_tasks}

    # 优先队列，存储 (f_cost, g_cost, counter, sequence, remaining_tasks_ids)
    # f_cost = g_cost + h_cost
    # counter 用于在成本相同时打破平局
    # remaining_tasks_ids 是一个存储任务ID的frozenset
    counter = 0
    initial_remaining_ids = frozenset(task.id for task in candidate_tasks)
    open_set = [(0, 0, counter, [], initial_remaining_ids)]
    counter += 1
    
    # 用于记录到达某个状态（由已排序的序列定义）的最小 g_cost
    closed_set = {}

    while open_set:
        f_cost, g_cost, _, sequence, remaining_ids = heapq.heappop(open_set)

        # 将序列元组化以用作字典的键
        sequence_tuple = tuple(t.id for t in sequence)
        if sequence_tuple in closed_set and closed_set[sequence_tuple] <= g_cost:
            continue
        closed_set[sequence_tuple] = g_cost

        # 如果所有任务都已安排，则找到了一个完整的路径
        if not remaining_ids:
            return sequence

        # 扩展节点：尝试将每一个剩余任务添加到当前序列的末尾
        for task_id_to_add in remaining_ids:
            task_to_add = task_map[task_id_to_add]
            new_sequence = sequence + [task_to_add]
            new_remaining_ids = remaining_ids - {task_id_to_add}
            
            # 将剩余任务ID集合转换为任务对象列表以供启发式函数使用
            remaining_tasks_for_heuristic = [task_map[tid] for tid in new_remaining_ids]

            # 计算新路径的成本
            new_g_cost = calculate_sequence_cost(new_sequence, env)
            new_h_cost = heuristic_cost_estimate(remaining_tasks_for_heuristic, env)
            new_f_cost = new_g_cost + new_h_cost
            
            heapq.heappush(open_set, (new_f_cost, new_g_cost, counter, new_sequence, new_remaining_ids))
            counter += 1

    return [] # 理论上对于有限任务总能找到解


def select_action_from_plan(valid_actions: np.ndarray, env: YardEnv) -> int:
    """
    根据 A* 算法制定的计划，选择要执行的下一个动作。
    """
    wait_action = env.action_space.n - 1
    internal_env = env.env
    visible_tasks_map = {i: task for i, task in enumerate(internal_env._last_visible_tasks) if task.location > 0}

    candidate_action_indices = [a for a in valid_actions if a != wait_action]

    if not candidate_action_indices:
        return wait_action if wait_action in valid_actions else valid_actions[0]

    candidate_tasks = [visible_tasks_map[idx] for idx in candidate_action_indices if idx in visible_tasks_map]

    if not candidate_tasks:
        return wait_action if wait_action in valid_actions else valid_actions[0]

    # A* 的计算复杂度非常高，为了避免在每一步都进行过长时间的搜索，
    # 我们在这里限制只对一部分“最紧急”的任务进行规划。
    # 我们根据“已等待时间”对任务进行排序，并选择前 N 个任务。
    current_time = internal_env.current_time
    candidate_tasks.sort(key=lambda task: current_time - task.available_time, reverse=True)
    
    PLANNING_TASK_LIMIT = 5 # 限制 A* 规划的任务数量
    tasks_to_plan = candidate_tasks[:PLANNING_TASK_LIMIT]

    # 使用 A* 算法规划任务
    planned_sequence = plan_with_astar(tasks_to_plan, env)
    
    if not planned_sequence:
        # 如果A*没有返回计划（理论上不应发生，除非输入为空），则选择等待
        return wait_action if wait_action in valid_actions else valid_actions[0]

    next_task_to_execute = planned_sequence[0]
    
    for idx, task in visible_tasks_map.items():
        if task == next_task_to_execute:
            return idx
            
    return wait_action


def run(use_static_data: bool = True, enable_render: bool = False, save_plot: bool = True, output_metrics: bool = True):
    """
    使用 A* 策略在 YardEnv 环境中运行。
    """
    if output_metrics:
        print("--- 开始评估 A* (A-star) 策略，静态数据: {} ---".format(use_static_data))

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
        print("开始运行 A* 策略...")
    loop_start = time.perf_counter()
    for _ in range(2000):
        if terminated or truncated:
            break
        
        action_masks = env.action_masks()
        valid_actions = np.nonzero(action_masks)[0]
        
        if len(valid_actions) == 0:
            break

        action = select_action_from_plan(valid_actions, env)
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
    loop_end = time.perf_counter()
    agent_loop_time = loop_end - loop_start

    if output_metrics:
        print("评估回合结束。")
        print(f"总奖励: {total_reward}")
        print(f"算法运行时间: {agent_loop_time:.6f}s")

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

    if save_plot:
        save_name = "AStar策略轨迹图_静态.png" if use_static_data else "AStar策略轨迹图_动态.png"
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