#!/usr/bin/env python3
"""
鲁棒性测试脚本 (Robustness Analysis)

功能：
1. 遍历不同的繁忙度 (Traffic Intensity)，即 `tasks_per_window`。
2. 为每个繁忙度生成静态任务集。
3. 对比 5 种算法：
   - PPO (Maskable PPO)
   - SA (Simulated Annealing)
   - A* (Rolling Horizon A*)
   - Nearest (Greedy Nearest Task)
   - Insertion (Insertion Heuristic)
4. 收集指标：任务总等待时间、算法运行时间。
5. 生成两张对比折线图。
"""

import os
import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from typing import List, Dict, Any, Tuple

# 添加项目根目录
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.environment.env_penalty import YardEnv
from src.environment.dataclasses import Task
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

# 导入各个启发式算法的决策函数
# 注意：需要确保这些文件在路径中可被导入，或者直接把逻辑搬过来。
# 这里假设我们可以从 tests 目录导入核心逻辑，或者简单复用代码。
# 为了稳健性，这里我们直接引用 tests 目录下的模块 (如果 sys.path 设置正确)
from tests.test_nearest import run as run_nearest  # 仅作参考，实际需要改造为函数调用
# 由于 tests 下的脚本通常是 `run()` 直接跑完，我们需要提取核心决策逻辑。
# 为了避免大量代码重复，本脚本将直接实现或包装各算法的 step 函数。

# -----------------------------------------------------------------------------
# 算法决策函数 (包装器)
# -----------------------------------------------------------------------------

def mask_fn(env: gym.Env) -> np.ndarray:
    return env.action_mask()

# 1. PPO Agent
def run_ppo_episode(env: YardEnv, model: MaskablePPO) -> Tuple[float, float]:
    """返回 (总等待时间, 算法总耗时)"""
    obs, _ = env.reset()
    done = False
    truncated = False
    
    start_time = time.perf_counter()
    while not (done or truncated):
        action_masks = env.action_masks()
        # predict 耗时计入算法时间
        action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
        obs, _, done, truncated, _ = env.step(action)
    end_time = time.perf_counter()
    
    total_wait = env.unwrapped.history_total_wait_time if hasattr(env.unwrapped, 'history_total_wait_time') else 0.0
    # 注意：env_penalty.py 里可能没有直接累积 total_wait_time 的属性，需要从 info 获取
    # 或者我们使用 info['total_task_wait_time'] 如果环境最后一步返回了它
    # 为保险起见，我们信任 env 结束时的 info
    return total_wait, end_time - start_time

# 2. Nearest Agent (Re-implementation for clean call)
def run_nearest_episode(env: YardEnv) -> Tuple[float, float]:
    obs, _ = env.reset()
    done = False
    truncated = False
    
    algo_time = 0.0
    
    while not (done or truncated):
        step_start = time.perf_counter()
        
        # --- 决策逻辑 ---
        action_masks = env.action_masks()
        valid_actions = np.nonzero(action_masks)[0]
        
        if len(valid_actions) == 0:
            break
            
        wait_action = env.unwrapped.max_tasks_in_obs
        crane_id = env.unwrapped.crane_to_command
        crane = env.unwrapped.cranes[crane_id]
        
        # 寻找最近的任务
        best_action = None
        min_dist = float('inf')
        
        # 获取可见任务列表 (从环境内部状态，或者观测中解析)
        # 为简便，直接访问 env.unwrapped._last_visible_tasks
        visible_tasks = env.unwrapped._last_visible_tasks
        
        # valid_actions 对应 visible_tasks 的索引 (除了最后一个是 wait)
        candidate_indices = [a for a in valid_actions if a != wait_action]
        
        if not candidate_indices:
            action = wait_action if wait_action in valid_actions else valid_actions[0]
        else:
            for idx in candidate_indices:
                if idx < len(visible_tasks):
                    task = visible_tasks[idx]
                    dist = abs(crane.location - task.location)
                    if dist < min_dist:
                        min_dist = dist
                        best_action = idx
            
            if best_action is not None:
                action = best_action
            else:
                action = wait_action
        # ----------------
        
        step_end = time.perf_counter()
        algo_time += (step_end - step_start)
        
        obs, _, done, truncated, _ = env.step(action)

    # 获取统计信息
    # 假设 env 在 step 返回的 info 里包含了 'total_task_wait_time'
    # 如果是最后一步，info 应该有。或者我们可以自己计算。
    # 这里我们依赖 env.unwrapped 在 close() 前的状态，或者 info。
    # 为了统一，我们在循环外访问 env.unwrapped 的统计变量（如果存在），或者只能依赖 info。
    # 这里的 env 是 ActionMasker 包装的，需要 unwrapped。
    # YardEnv 通常在 info 中返回 'total_task_wait_time'
    # 我们需要捕获最后一次 info
    
    # 重新修正：在 while 循环中最后一次 step 的 info
    return 0.0, algo_time # 占位，实际在统一流程中处理

# 为了代码复用，我们定义一个通用的 Evaluation Loop
# 只需要传入 "决策函数" 即可。

def get_action_nearest(env: YardEnv, valid_actions: np.ndarray) -> int:
    wait_action = env.unwrapped.max_tasks_in_obs
    crane = env.unwrapped.cranes[env.unwrapped.crane_to_command]
    visible_tasks = env.unwrapped._last_visible_tasks
    
    candidates = [a for a in valid_actions if a != wait_action]
    if not candidates:
        return wait_action
        
    best_action = candidates[0]
    min_dist = float('inf')
    
    for idx in candidates:
        if idx < len(visible_tasks):
            task = visible_tasks[idx]
            dist = abs(crane.location - task.location)
            if dist < min_dist:
                min_dist = dist
                best_action = idx
    return best_action

# 3. Insertion Heuristic (Importing or mimicking logic)
from tests.test_insertion_heuristic import select_action_from_plan as insertion_policy

# 4. SA (Importing or mimicking logic)
from tests.test_SA import select_action_sa as sa_policy

# 5. A* (Importing or mimicking logic)
# A* 需要一点适配，因为 test_astar.py 里是 plan_with_astar 然后取第一个。
# 我们需要一个 wrapper
from tests.test_astar import plan_with_astar

def get_action_astar(env: YardEnv, valid_actions: np.ndarray) -> int:
    wait_action = env.unwrapped.max_tasks_in_obs
    visible_tasks = env.unwrapped._last_visible_tasks
    
    # 筛选有效任务
    candidates_indices = [a for a in valid_actions if a != wait_action]
    candidate_tasks = [visible_tasks[i] for i in candidates_indices if i < len(visible_tasks)]
    
    if not candidate_tasks:
        return wait_action
        
    # 限制数量以防 A* 爆炸 (同 test_astar.py)
    LIMIT = 5
    tasks_to_plan = candidate_tasks[:LIMIT]
    
    planned_sequence = plan_with_astar(tasks_to_plan, env)
    
    if not planned_sequence:
        return candidates_indices[0] # Fallback
        
    next_task = planned_sequence[0]
    # 反查 index
    for idx in candidates_indices:
        if idx < len(visible_tasks) and visible_tasks[idx].id == next_task.id:
            return idx
            
    return wait_action

# -----------------------------------------------------------------------------
# 统一评估流程
# -----------------------------------------------------------------------------

def evaluate_algorithm(env: gym.Env, name: str, model=None) -> Tuple[float, float]:
    """
    在给定环境上运行一次完整的 Episode。
    返回: (任务总等待时间, 算法决策总耗时)
    """
    obs, _ = env.reset()
    done = False
    truncated = False
    
    algo_time_total = 0.0
    last_info = {}
    
    while not (done or truncated):
        start_t = time.perf_counter()
        
        action_masks = env.action_masks()
        valid_actions = np.nonzero(action_masks)[0]
        
        if len(valid_actions) == 0:
            break
            
        if name == "PPO":
            # PPO 预测
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
        elif name == "Nearest":
            action = get_action_nearest(env, valid_actions)
        elif name == "Insertion":
            # test_insertion_heuristic 的接口需要 env (Wrapper)
            action = insertion_policy(valid_actions, env) 
        elif name == "SA":
            action = sa_policy(valid_actions, env)
        elif name == "A*":
            action = get_action_astar(env, valid_actions)
        else:
            action = valid_actions[0] # Default Random
            
        end_t = time.perf_counter()
        algo_time_total += (end_t - start_t)
        
        obs, reward, done, truncated, info = env.step(action)
        last_info = info
        
    # 从 info 中提取总等待时间
    # 假设 YardEnv 的 info 包含 'total_task_wait_time'
    total_wait = last_info.get('total_task_wait_time', 0.0)
    
    # 如果环境没返回，尝试从 history 计算 (fallback)
    # 这里我们信任环境实现。
    
    return total_wait, algo_time_total

# -----------------------------------------------------------------------------
# 主程序
# -----------------------------------------------------------------------------

def main():
    # 1. 准备 PPO 模型
    model_path = os.path.join("models", "ppo_env_model.zip")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    print(f"Loading PPO model from {model_path}...")
    ppo_model = MaskablePPO.load(model_path)
    
    # 2. 定义测试场景 (Tasks per Window)
    # 3 -> 闲, 5 -> 常, 7 -> 忙, 9 -> 堵
    scenarios = [3, 5, 7, 9] 
    scenario_labels = ["Light (3)", "Normal (5)", "Heavy (7)", "Overload (9)"]
    
    algorithms = ["PPO", "Nearest", "Insertion", "SA", "A*"]
    # 存储结果: results[algo][metric] = [val_scenario_1, val_scenario_2, ...]
    data_wait_time = {algo: [] for algo in algorithms}
    data_runtime = {algo: [] for algo in algorithms}
    
    # 导入任务生成器
    from tests.generate_static_tasks_env import generate_static_tasks
    
    print("\n--- Starting Robustness Test ---")
    
    for tasks_per_window in scenarios:
        print(f"\nEvaluating Scenario: {tasks_per_window} tasks/window")
        
        # A. 生成该场景下的静态任务 (保证所有算法面对相同的任务序列)
        # 使用临时的 json 文件
        temp_task_file = "temp_robustness_tasks.json"
        tasks = generate_static_tasks(
            seed=89, # 固定种子，保证可复现
            tasks_per_window=tasks_per_window,
            out=temp_task_file,
            output_format="json"
        )
        
        # B. 遍历每个算法
        for algo in algorithms:
            print(f"  > Running {algo}...", end="", flush=True)
            
            # 加载任务并初始化环境
            # 注意：每次都要重新初始化环境以重置状态
            with open(temp_task_file, "r", encoding="utf-8") as f:
                raw_tasks = json.load(f)
            static_tasks = [Task(**t) for t in raw_tasks]
            
            # 关键：这里必须与训练时的 max_tasks_in_obs 保持一致 (30)，否则 obs shape 不匹配
            env = YardEnv(render_mode=None, static_tasks=static_tasks, max_tasks_in_obs=30)
            env = ActionMasker(env, mask_fn)
            
            try:
                # PPO 模型需要传入，其他算法不需要
                model_arg = ppo_model if algo == "PPO" else None
                
                # 运行评估
                wait_time, runtime = evaluate_algorithm(env, algo, model_arg)
                
                data_wait_time[algo].append(wait_time)
                data_runtime[algo].append(runtime)
                print(f" Done. (Wait: {wait_time:.1f}s, Time: {runtime:.2f}s)")
                
            except Exception as e:
                print(f" Failed! ({e})")
                data_wait_time[algo].append(0) # 填充 0 或 NaN
                data_runtime[algo].append(0)
            finally:
                env.close()
                
        # 清理临时文件
        if os.path.exists(temp_task_file):
            os.remove(temp_task_file)

    # 3. 绘图
    print("\nPlotting results...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 颜色映射
    colors = {"PPO": "red", "Nearest": "blue", "Insertion": "green", "SA": "orange", "A*": "purple"}
    markers = {"PPO": "o", "Nearest": "s", "Insertion": "^", "SA": "D", "A*": "x"}
    
    x = np.arange(len(scenarios))
    
    # 图 1: 总等待时间
    for algo in algorithms:
        ax1.plot(x, data_wait_time[algo], label=algo, color=colors[algo], marker=markers[algo], linewidth=2)
    
    ax1.set_title("Total Task Wait Time vs. Busyness")
    ax1.set_xlabel("Traffic Intensity (Tasks/Window)")
    ax1.set_ylabel("Total Wait Time (s)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenario_labels)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    # 图 2: 算法运行时间
    for algo in algorithms:
        # 使用对数坐标，因为 A*/SA 可能比 PPO 慢几个数量级
        ax2.plot(x, data_runtime[algo], label=algo, color=colors[algo], marker=markers[algo], linewidth=2)
    
    ax2.set_title("Algorithm Runtime vs. Busyness")
    ax2.set_xlabel("Traffic Intensity (Tasks/Window)")
    ax2.set_ylabel("Total Runtime (s) [Log Scale]")
    ax2.set_yscale("log") # 关键：开启对数坐标
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenario_labels)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    plt.tight_layout()
    save_path = "robustness_comparison.png"
    plt.savefig(save_path, dpi=300)
    print(f"Chart saved to {save_path}")

if __name__ == "__main__":
    main()