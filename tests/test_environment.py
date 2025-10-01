import random
import sys
import os

# 将项目根目录添加到 Python 路径中，以便能够找到 src 模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.env import YardEnv

def main():
    """
    一个简单的测试脚本，用于验证 YardEnv 环境是否能正常工作。
    它会执行随机动作，并打印出每一步的信息。
    """
    # 1. 初始化环境
    print("Initializing environment...")
    env = YardEnv(render_mode="human")
    num_actions = env.action_space.n
    total_reward = 0
    
    # 2. 重置环境
    print("Resetting environment...")
    observation, info = env.reset()
    crane_to_command = info.get('crane_to_command')
    print(f"Initial state: Crane {crane_to_command} needs a command.")
    # print(f"Observation: {observation}")
    print("-" * 50)

    # 3. 运行一个 episode
    terminated = False
    truncated = False
    step_count = 0

    while not terminated and not truncated:
        step_count += 1
        print(f"\n----- Step {step_count} -----")

        # 4. 从有效动作中选择一个随机动作
        action_mask = observation['action_mask']
        valid_actions = [i for i, valid in enumerate(action_mask) if valid == 1]
        
        if not valid_actions:
            # 理论上，由于我们有“等待”动作，有效动作列表不应为空。
            # 如果为空，说明可能存在逻辑问题。
            print("Error: No valid actions available. Breaking loop.")
            break
        
        action = random.choice(valid_actions)

        print(f"Executing action {action} for crane {crane_to_command}...")

        # 5. 执行动作
        observation, reward, terminated, truncated, info = env.step(action)
        crane_to_command = info.get('crane_to_command')
        
        total_reward += reward

        # 6. 打印信息
        print(f"Next crane to command: {crane_to_command}")
        # print(f"Observation: {observation}")
        print(f"Reward received: {reward}")
        print(f"Total reward: {total_reward}")
        print(f"Is terminated: {terminated}")
        print(f"Is truncated: {truncated}")
        print(f"Info: {info}")

        if terminated or truncated:
            print("\n----- Episode Finished -----")
            print(f"Termination reason: {info.get('termination_reason', 'N/A')}")
            print(f"Total steps: {step_count}")
            print(f"Final total reward: {total_reward}")
            break

    print("\n----- Simulation Finished -----")
    env.render()
    env.close()


if __name__ == "__main__":
    main()