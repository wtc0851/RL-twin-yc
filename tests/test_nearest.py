import os
import sys
import gymnasium as gym

# 将项目根目录添加到 Python 路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.env import YardEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from tests.static_tasks import TASK_LIST
import numpy as np


def mask_fn(env: gym.Env) -> np.ndarray:
    """一个辅助函数，用于从环境中提取动作掩码。"""
    return env.action_mask()


def main():
    """
    使用静态任务数据在 YardEnv 环境中运行最近任务选择策略进行测试。
    """
    print("--- 开始使用静态任务数据评估最近任务选择策略 ---")

    # 创建一个用于评估的环境实例，使用静态任务数据
    env = YardEnv(render_mode='human', static_tasks=TASK_LIST)  # 使用 'human' 模式以收集绘图数据
    env = ActionMasker(env, mask_fn)

    obs, info = env.reset()
    terminated = False  # 终止标志
    truncated = False  # 截断标志
    total_reward = 0

    print("开始运行最近任务选择策略...")
    # 加入步数上限，避免策略在等待循环中无限运行
    for _ in range(2000):
        if terminated or truncated:
            break
        action_masks = env.action_masks()
        
        # 选择最近的任务
        valid_actions = np.nonzero(action_masks)[0]
        
        if len(valid_actions) == 0:
            # 如果没有有效动作，跳过这一步
            break
            
        # 获取当前观察
        current_obs = env.env._get_observation(env.env.crane_to_command)
        
        # 获取当前需要决策的场桥位置
        crane_to_command_id = env.env.crane_to_command
        current_crane_location = env.env.cranes[crane_to_command_id].location
        
        # 如果只有等待动作可用，选择等待
        if len(valid_actions) == 1 and valid_actions[0] == env.env.max_tasks_in_obs:
            action = valid_actions[0]
        else:
            # 计算每个有效任务动作到当前场桥的距离
            best_action = None
            min_distance = float('inf')
            
            for action_idx in valid_actions:
                if action_idx < env.env.max_tasks_in_obs:  # 排除等待动作
                    # 获取任务位置
                    task_location = current_obs["task_list"][action_idx][0]
                    if task_location > 0:  # 确保是有效任务
                        distance = abs(current_crane_location - task_location)
                        if distance < min_distance:
                            min_distance = distance
                            best_action = action_idx
            
            # 如果找到了最近的任务，选择它；否则选择等待
            if best_action is not None:
                action = best_action
            else:
                # 选择等待动作
                wait_action = env.env.max_tasks_in_obs
                if wait_action in valid_actions:
                    action = wait_action
                else:
                    action = valid_actions[0]  # 选择第一个有效动作作为备选

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
    
    # 调用环境的绘图方法来生成并保存轨迹图
    env.env.plot_crane_trajectories(save_path="最近任务选择轨迹图.png")
    print("轨迹图已保存至 最近任务选择轨迹图.png")

    env.close()


if __name__ == "__main__":
    main()
