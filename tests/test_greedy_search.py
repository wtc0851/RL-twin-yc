#!/usr/bin/env python3
"""
使用贪心集束搜索 (Greedy Beam Search) 算法来解决动态调度问题。
该算法在每一步扩展时，只保留固定数量（集束宽度 k）的最优候选序列，
从而在计算开销和解的质量之间取得平衡。
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


def plan_with_greedy_search(candidate_tasks: list[Task], env: YardEnv, beam_width: int = 3) -> list[Task]:
    """
    使用集束搜索算法为给定的候选任务寻找一个高质量的执行序列。

    :param candidate_tasks: 待规划的任务列表。
    :param env: 环境实例，用于成本计算。
    :param beam_width: 集束宽度 (k)，控制搜索的广度。
    :return: 找到的执行序列。
    """
    if not candidate_tasks:
        return []

    # 初始化集束，每个元素是 (cost, sequence)
    # 初始时，集束包含所有单个任务构成的序列
    beam = []
    for task in candidate_tasks:
        sequence = [task]
        cost = calculate_sequence_cost(sequence, env)
        beam.append((cost, sequence))
    
    # 根据成本排序并剪枝到 beam_width
    beam.sort(key=lambda x: x[0])
    beam = beam[:beam_width]

    # 迭代扩展序列，直到序列长度等于任务总数
    for _ in range(1, len(candidate_tasks)):
        all_candidates = []
        for cost, sequence in beam:
            # 获取当前序列中已包含的任务ID
            used_task_ids = {t.id for t in sequence}
            
            # 尝试将每个未使用的任务追加到序列末尾
            for task_to_add in candidate_tasks:
                if task_to_add.id not in used_task_ids:
                    new_sequence = sequence + [task_to_add]
                    new_cost = calculate_sequence_cost(new_sequence, env)
                    all_candidates.append((new_cost, new_sequence))
        
        # 如果没有生成新的候选（例如所有任务都已在一个序列中），则跳出
        if not all_candidates:
            break

        # 从所有新生成的候选中选择最优的 k 个
        all_candidates.sort(key=lambda x: x[0])
        beam = all_candidates[:beam_width]

    # 返回最终集束中成本最低的序列
    if not beam:
        return []
    
    best_sequence = min(beam, key=lambda x: x[0])[1]
    return best_sequence


def select_action_from_plan(valid_actions: np.ndarray, env: YardEnv) -> int:
    """
    根据贪心集束搜索算法制定的计划，选择要执行的下一个动作。
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

    # 为了平衡性能，我们仍然可以限制送入规划算法的任务数量
    current_time = internal_env.current_time
    candidate_tasks.sort(key=lambda task: current_time - task.available_time, reverse=True)
    
    BEAM_WIDTH = 5
    # 使用贪心集束搜索算法规划任务
    planned_sequence = plan_with_greedy_search(candidate_tasks, env, beam_width=BEAM_WIDTH)
    
    if not planned_sequence:
        # 如果算法没有返回计划，则选择等待
        return wait_action if wait_action in valid_actions else valid_actions[0]

    next_task_to_execute = planned_sequence[0]
    
    for idx, task in visible_tasks_map.items():
        if task.id == next_task_to_execute.id:
            return idx
            
    return wait_action

def run(use_static_data: bool = True, enable_render: bool = False, save_plot: bool = True, output_metrics: bool = True):
    """
    使用贪心集束搜索策略在 YardEnv 环境中运行。
    """
    if output_metrics:
        print("--- 开始评估贪心集束搜索 (Greedy Beam Search) 策略，静态数据: {} ---".format(use_static_data))

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
        print("开始运行贪心集束搜索策略...")
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
        save_name = "GreedySearch策略轨迹图_静态.png" if use_static_data else "GreedySearch策略轨迹图_动态.png"
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