import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import csv
import traceback

# 检查CUDA是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):
    """
    Actor-Critic网络，适用于单智能体环境。
    """
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.common = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Linear(128, 1)

    def forward(self, state):
        features = self.common(state)
        action_probs = self.actor(features)
        state_value = self.critic(features)
        return action_probs, state_value

class PPO:
    """
    PPO算法的单智能体实现。
    """
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, clip_param=0.2, load_from_folder_path=None):
        self.gamma = gamma
        self.clip_param = clip_param
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.trajectory = []
        self.is_eval_mode = False

        if load_from_folder_path:
            self.load_model(load_from_folder_path)

    def set_eval_mode(self, eval_mode_on=True):
        self.is_eval_mode = eval_mode_on
        if self.is_eval_mode:
            self.policy.eval()
        else:
            self.policy.train()

    def select_action(self, state, action_mask):
        state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
        
        if not self.is_eval_mode:
            self.policy.eval()

        with torch.no_grad():
            action_probs, _ = self.policy(state_tensor)

        if not self.is_eval_mode:
            self.policy.train()

        action_probs = action_probs.squeeze()
        
        # 应用action mask
        valid_indices = np.nonzero(action_mask)[0]
        if len(valid_indices) == 0:
            # 如果没有有效动作，这是一个问题，但我们还是需要返回一个动作
            # 默认选择第一个动作，但这应该在环境中处理
            return 0, 0.0

        valid_indices_tensor = torch.tensor(valid_indices, dtype=torch.long).to(device)
        valid_action_probs = action_probs[valid_indices_tensor]
        
        if valid_action_probs.sum().item() == 0:
            # 如果所有有效动作的概率都为0，则均匀选择
            valid_action_probs = torch.ones_like(valid_action_probs) / len(valid_action_probs)
        else:
            valid_action_probs = valid_action_probs / valid_action_probs.sum()

        dist = torch.distributions.Categorical(valid_action_probs)
        action_index = dist.sample()
        selected_action = valid_indices[action_index.item()]
        action_prob = dist.log_prob(action_index).exp().item()

        return selected_action, action_prob

    def store_transition(self, state, action, reward, done, old_prob):
        if self.is_eval_mode:
            return
        self.trajectory.append((state, action, reward, done, old_prob))

    def compute_loss(self, old_probs, states, actions, rewards_to_go, advantages):
        states_tensor = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        actions_tensor = torch.tensor(actions, dtype=torch.int64).to(device)
        rewards_to_go_tensor = torch.tensor(rewards_to_go, dtype=torch.float32).to(device)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32).to(device)
        old_probs_tensor = torch.tensor(old_probs, dtype=torch.float32).to(device)

        action_probs_full, state_values = self.policy(states_tensor)
        current_action_probs = action_probs_full.gather(1, actions_tensor.unsqueeze(-1)).squeeze(-1)
        state_values = state_values.squeeze(-1)

        # PPO损失
        ratio = (current_action_probs / (old_probs_tensor + 1e-10))
        surrogate1 = ratio * advantages_tensor
        surrogate2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages_tensor
        actor_loss = -torch.min(surrogate1, surrogate2).mean()

        # Critic损失
        critic_loss = nn.MSELoss()(state_values, rewards_to_go_tensor)

        # 熵损失
        entropy = -torch.sum(action_probs_full * torch.log(action_probs_full + 1e-10), dim=-1).mean()

        return actor_loss + 0.5 * critic_loss - 0.01 * entropy

    def update(self):
        if self.is_eval_mode or not self.trajectory:
            self.trajectory = []
            return

        states, actions, rewards, dones, old_probs = zip(*self.trajectory)
        
        # 计算Rewards-to-go
        rewards_to_go = []
        discounted_reward = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards_to_go.insert(0, discounted_reward)

        # 计算GAE
        states_tensor = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        with torch.no_grad():
            _, state_values = self.policy(states_tensor)
        state_values = state_values.squeeze().tolist()

        advantages = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * (state_values[i+1] if i+1 < len(state_values) else 0) * (1 - dones[i]) - state_values[i]
            gae = delta + self.gamma * 0.95 * (1 - dones[i]) * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        loss = self.compute_loss(old_probs, states, actions, rewards_to_go, advantages)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.trajectory = []

    def save_model(self, folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        torch.save(self.policy.state_dict(), os.path.join(folder_path, "actor_critic.pth"))

    def load_model(self, folder_path):
        model_path = os.path.join(folder_path, "actor_critic.pth")
        if os.path.exists(model_path):
            self.policy.load_state_dict(torch.load(model_path, map_location=device))
            print(f"模型已从 {model_path} 加载")
        else:
            print(f"在 {model_path} 未找到模型")