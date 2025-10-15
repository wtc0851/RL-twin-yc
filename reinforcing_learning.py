import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import traceback # 导入traceback用于调试

# 检查CUDA是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Actor-Critic网络定义
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_space_func):
        super(ActorCritic, self).__init__()
        self.action_space_func = action_space_func
        self.state_dim_cache = state_dim # 缓存state_dim用于潜在的actor初始化
        self.common = nn.Sequential(
            nn.Linear(state_dim, 128),  # 确保state_dim与状态长度匹配
            nn.ReLU()
        )
        self.actors = None  # 动态定义
        self.critic = nn.Linear(128, 1)

    def _ensure_actors_initialized(self, agent_idx, example_state_for_size=None):
        """
        确保self.actors已初始化。
        如果未提供example_state_for_size，将尝试使用缓存的state_dim创建虚拟状态。
        """
        if self.actors is None:
            # print(f"Agent {agent_idx} Actor is None. Attempting initialization for loading.")
            # 为了确定action_space_size，action_space_func可能需要一个状态
            # 我们需要一个原型状态或其维度信息
            if example_state_for_size is None:
                # 如果外部没有提供状态示例，我们基于state_dim_cache创建一个虚拟零向量状态
                # 这假设action_space_func可以从这样的状态（或其形状）推断大小
                # 或者action_space_func实际上只依赖于agent_idx
                # print(f"Agent {agent_idx}: Creating dummy state for actor initialization using state_dim: {self.state_dim_cache}")
                example_state_for_size = torch.zeros(self.state_dim_cache).to(device) # 确保在正确的设备上
                if example_state_for_size.dim() == 1: # forward期望批次
                    example_state_for_size = example_state_for_size.unsqueeze(0)


            # action_space_func可能需要state和agent_idx
            # 如果状态本身很复杂（例如，列表的列表），创建虚拟状态会更复杂
            # 这里我们简单处理，假设action_space_func可以很好地处理它
            action_space_size = self.action_space_func(example_state_for_size, agent_idx)
            if action_space_size <= 0: # 添加验证
                raise ValueError(f"Agent {agent_idx}: action_space_size必须为正数，得到{action_space_size}。检查action_space_funcs。")

            self.actors = nn.Sequential(
                nn.Linear(128, action_space_size),
                nn.Softmax(dim=-1)
            ).to(device)
            # print(f"Agent {agent_idx}: Actor initialized with action_space_size={action_space_size} for loading.")


    def forward(self, state, agent_idx):
        # 确保state是张量并在正确的设备上
        if not isinstance(state, torch.Tensor):
            state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
        else:
            state_tensor = state.to(device, dtype=torch.float32) # 确保设备和类型

        if state_tensor.dim() == 1: # 如果是单个样本，添加批次维度
            state_tensor = state_tensor.unsqueeze(0)

        # 提取公共特征
        features = self.common(state_tensor) # 现在state_tensor是[batch_size, state_dim]
        action_space_size = self.action_space_func(state_tensor, agent_idx) # state_tensor有批次维度

        # 动态创建/验证Actor网络
        if (
            self.actors is None
            or not isinstance(self.actors[-1], nn.Linear)  # 确保最后一层是Linear（在Softmax之前）
            or self.actors[-1].out_features != action_space_size
        ):
            # print(f"Agent {agent_idx}: Re/Initializing Actor in forward. Current action_space_size: {action_space_size}")
            self.actors = nn.Sequential(
                nn.Linear(128, action_space_size),
                nn.Softmax(dim=-1)
            ).to(device)

        action_probs = self.actors(features)
        state_value = self.critic(features)

        # 如果原始输入是单个样本，移除批次维度
        if state.dim() == 1:
            action_probs = action_probs.squeeze(0)
            state_value = state_value.squeeze(0)

        return action_probs, state_value

# MAPPO算法实现
class MAPPO:
    def __init__(self, state_dims, action_space_funcs, num_agents, lr=1e-3, gamma=0.99, clip_param=0.2,
                 load_from_folder_path=None):
        self.num_agents = num_agents
        self.gamma = gamma
        self.clip_param = clip_param
        self.action_space_funcs = action_space_funcs
        self.state_dims_cache = state_dims

        self.agents = [
            ActorCritic(state_dims[i], self.action_space_funcs[i]).to(device) for i in range(num_agents)
        ]
        self.optimizers = [optim.Adam(agent.parameters(), lr=lr) for agent in self.agents]
        self.trajectories = [[] for _ in range(num_agents)]
        self.cumulative_rewards = [[] for _ in range(num_agents)]
        self.update_counts = [0 for _ in range(num_agents)]
        self.policy_entropy = [[] for _ in range(num_agents)]
        self.value_losses = [[] for _ in range(num_agents)]
        self.gradient_norms = [[] for _ in range(num_agents)]

        self.current_results_folder = "."
        self.gradient_csv_file_basename = "gradient_norms.csv"

        # 添加：评估模式标志，默认为False（训练模式）
        self.is_eval_mode = False

        if load_from_folder_path:
            self.current_results_folder = load_from_folder_path
            gradient_csv_full_path = os.path.join(load_from_folder_path, self.gradient_csv_file_basename)
            if not os.path.exists(gradient_csv_full_path):
                 with open(gradient_csv_full_path, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    header = ["Update Step"] + [f"Agent {i} Gradient Norm" for i in range(num_agents)]
                    writer.writerow(header)
        
        if load_from_folder_path is not None:
            print(f"尝试从CSV文件加载模型参数：{load_from_folder_path}")
            self.load_from_csv(load_from_folder_path)

    # 添加：设置评估模式的方法（可选，但推荐）
    def set_eval_mode(self, eval_mode_on=True):
        """
        将智能体设置为评估模式（无学习/更新）或训练模式。
        """
        self.is_eval_mode = eval_mode_on
        if self.is_eval_mode:
            print("MAPPO智能体设置为评估模式。策略更新将被跳过。")
            # PyTorch的eval()模式对nn.Module影响Dropout、BatchNorm等层
            for agent_model in self.agents:
                agent_model.eval()
        else:
            print("MAPPO智能体设置为训练模式。将执行策略更新。")
            for agent_model in self.agents:
                agent_model.train()


    def select_action(self, action_state, env_state, agent_idx):
        state = action_state + env_state
        if not isinstance(state, (list, np.ndarray)) or len(np.shape(state)) != 1:
            raise ValueError(f"无效的状态格式：{state}。状态必须是平坦的列表或1D数组。")
        state_tensor = torch.tensor(state, dtype=torch.float32).to(device)

        # 如果不在评估模式（即仍在训练且可能在回合期间切换），在每次动作选择前将PyTorch模型设置为eval()
        # 如果在全局评估模式（self.is_eval_mode为True），模型已在set_eval_mode中设置为.eval()
        if not self.is_eval_mode:
            self.agents[agent_idx].eval() 
        
        with torch.no_grad(): # 始终对动作选择使用no_grad，因为它不涉及学习步骤
            action_probs, _ = self.agents[agent_idx](state_tensor, agent_idx)
        
        if not self.is_eval_mode: # 如果之前切换到eval()，现在恢复train()
            self.agents[agent_idx].train()
            self.agents[agent_idx].train()

        action_probs = action_probs.squeeze()

        valid_indices_np = np.nonzero(action_state)[0]
        if len(valid_indices_np) == 0:
            raise ValueError(f"智能体{agent_idx}：在select_action期间action_state中没有可用的有效动作：{action_state}。如果在之前检查过，这不应该发生。")

        valid_indices = torch.tensor(valid_indices_np, dtype=torch.long).to(device)
        
        valid_action_probs = action_probs[valid_indices]

        if torch.isnan(valid_action_probs).any() or valid_action_probs.sum().item() == 0:
            if len(valid_indices) > 0:
                valid_action_probs = torch.ones_like(valid_action_probs) / len(valid_indices)
            else:
                 raise ValueError(f"智能体{agent_idx}：没有valid_indices但action_state的和>0。不一致。")
        else:
            valid_action_probs = valid_action_probs / valid_action_probs.sum()
        
        if torch.isnan(valid_action_probs).any():
            raise ValueError(f"智能体{agent_idx}：恢复尝试后valid_action_probs仍为NaN：{valid_action_probs}")

        selected_local_index = torch.multinomial(valid_action_probs, 1).item()
        selected_action = valid_indices[selected_local_index].item()

        # 在评估模式下，我们通常不需要存储动作概率用于后续的PPO更新（因为没有更新）
        # 但store_transition可能仍会被调用，所以返回一个合理的概率值
        action_probability_to_return = valid_action_probs[selected_local_index].item()
        
        return selected_action, action_probability_to_return


    def store_transition(self, agent_idx, state, action, reward, done, old_prob):
        # 在评估模式下，我们不需要存储经验用于训练
        if self.is_eval_mode:
            return

        if not isinstance(state, (list, np.ndarray)):
            raise ValueError(f"状态必须是列表或数组，但得到{type(state)}，{state}。")
        if not isinstance(action, int):
            raise ValueError(f"动作必须是整数，但得到{type(action)}，{action}。")
        if not isinstance(reward, (int, float)):
            raise ValueError(f"奖励必须是数字，但得到{type(reward)}，{reward}。")
        self.trajectories[agent_idx].append((state, action, reward, done, old_prob))

    def compute_loss(self, agent, agent_idx, old_probs, states, actions, rewards, dones):
        # ... (compute_loss方法保持不变) ...
        try:
            states_tensor = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        except ValueError as e:
            raise ValueError(f"无效的状态格式：{states}。确保状态一致。") from e

        actions_tensor = torch.tensor(actions, dtype=torch.int64).to(device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32).to(device)
        old_probs_tensor = torch.tensor(old_probs, dtype=torch.float32).to(device)
        
        action_probs_full, state_values = agent(states_tensor, agent_idx)
        current_action_probs = action_probs_full.gather(1, actions_tensor.unsqueeze(-1)).squeeze(-1)
        state_values = state_values.squeeze(-1)
        
        ratio = current_action_probs / (old_probs_tensor + 1e-10)
        advantages_for_actor = rewards_tensor + self.gamma * state_values * (1 - dones_tensor) - state_values.detach()
        
        surrogate1 = ratio * advantages_for_actor
        surrogate2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages_for_actor
        actor_loss = -torch.min(surrogate1, surrogate2).mean()
        
        critic_target = rewards_tensor
        critic_loss = nn.MSELoss()(state_values, critic_target)
        
        entropy = -torch.sum(action_probs_full * torch.log(action_probs_full + 1e-10), dim=-1).mean()

        self.value_losses[agent_idx].append(critic_loss.item())
        self.policy_entropy[agent_idx].append(entropy.item())

        return actor_loss + critic_loss - 0.01 * entropy


    def update(self):
        # 检查评估模式标志
        if self.is_eval_mode:
            # 即使在评估模式下也清除轨迹，以防止内存泄漏或旧数据的干扰
            self.trajectories = [[] for _ in range(self.num_agents)]
            # print("评估模式：跳过策略更新。") # 可选的调试信息
            return # 如果在评估模式，不执行任何更新

        # ... (原始更新逻辑从这里开始) ...
        for agent_idx, trajectory in enumerate(self.trajectories):
            if not trajectory:
                continue
            try:
                states, actions, rewards, dones, old_probs = zip(*trajectory)
                
                if np.isnan(rewards).any():
                    print(f"[错误] 智能体{agent_idx}：在损失计算前在奖励中检测到NaN。轨迹：{trajectory}")
                    self.trajectories[agent_idx] = [] 
                    continue

                loss = self.compute_loss(self.agents[agent_idx], agent_idx, old_probs, states, actions, rewards, dones)
                
                if torch.isnan(loss):
                    print(f"[错误] 智能体{agent_idx}：损失为NaN。跳过优化器步骤。检查奖励/状态/动作值。")
                    self.trajectories[agent_idx] = [] 
                    continue

                optimizer = self.optimizers[agent_idx]
                optimizer.zero_grad()
                loss.backward()

                grad_norm_val = 0.0
                params_with_grad = [p for p in self.agents[agent_idx].parameters() if p.grad is not None]
                if params_with_grad:
                    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in params_with_grad]), 2)
                    grad_norm_val = total_norm.item()
                self.gradient_norms[agent_idx].append(grad_norm_val)
                
                optimizer.step()

                total_reward_for_logging = sum(r for s, a, r, d, p in trajectory) 
                self.update_counts[agent_idx] += 1
                if self.update_counts[agent_idx] == 1:
                    self.cumulative_rewards[agent_idx].append(-total_reward_for_logging) 
                else:
                    prev_avg = self.cumulative_rewards[agent_idx][-1]
                    new_avg = prev_avg + (-total_reward_for_logging - prev_avg) / self.update_counts[agent_idx]
                    self.cumulative_rewards[agent_idx].append(new_avg)
            except Exception as e:
                print(f"更新智能体{agent_idx}时出错：{e}")
                traceback.print_exc() 
                continue
        self.trajectories = [[] for _ in range(self.num_agents)] # 清除轨迹


    def log_trajectory(self, agent_idx):
        trajectory = self.trajectories[agent_idx]
        print(f"[调试] 智能体{agent_idx}的轨迹：")
        for idx, (state, action, reward, done, old_prob) in enumerate(trajectory):
            print(f"  步骤{idx}：state={state}, action={action}, reward={reward}, done={done}, old_prob={old_prob}")
            
    def load_from_csv(self, folder_path):
        """
        从CSV文件加载所有智能体的策略参数。
        folder_path：包含CSV文件的目录路径
        """
        print(f"尝试从{folder_path}中的CSV文件加载模型参数...")
        for idx, agent in enumerate(self.agents):
            # 关键：在加载参数前确保Actor网络已构建
            # 由于Actor是动态创建的，如果之前没有前向调用，actors可能为None
            # 我们需要action_space_func和示例状态（或其维度）来初始化它
            if agent.actors is None:
                # print(f"智能体{idx} actor为None。在从CSV加载前初始化。")
                # 使用缓存的state_dim和action_space_func来初始化
                # ActorCritic._ensure_actors_initialized需要agent_idx和state_dim
                # 此智能体的state_dim是self.state_dims_cache[idx]
                agent._ensure_actors_initialized(idx, example_state_for_size=None) # 传递None，它将使用缓存的state_dim


            actor_file = os.path.join(folder_path, f"agent_{idx}_actor.csv")
            critic_file = os.path.join(folder_path, f"agent_{idx}_critic.csv")
            
            actor_loaded = False
            if os.path.exists(actor_file):
                if agent.actors is not None: # 再次检查，因为_ensure_actors_initialized可能失败
                    try:
                        with open(actor_file, 'r') as f:
                            # 从模型存储参数以匹配形状（如果需要）
                            actor_model_params = {name: param for name, param in agent.actors.named_parameters()}
                            reader = csv.reader(f)
                            for row in reader:
                                if not row: continue # 跳过空行
                                name_from_csv, *data_str = row
                                if name_from_csv in actor_model_params:
                                    param_to_load = actor_model_params[name_from_csv]
                                    try:
                                        loaded_data = np.array(data_str, dtype=np.float32)
                                        param_to_load.data.copy_(torch.tensor(loaded_data).view_as(param_to_load.data).to(device))
                                        actor_loaded = True
                                    except ValueError as ve:
                                        print(f"智能体{idx} actor参数{name_from_csv}的ValueError：{ve}。数据：{data_str[:10]}")
                                    except RuntimeError as re:
                                        print(f"智能体{idx} actor参数{name_from_csv}的RuntimeError：{re}。期望形状{param_to_load.data.shape}")
                                # else:
                                #     print(f"警告：CSV中的参数{name_from_csv}在智能体{idx} actor模型中未找到。")
                        if actor_loaded: print(f"智能体{idx} actor参数从{actor_file}加载")
                    except Exception as e:
                        print(f"处理智能体{idx}的actor CSV {actor_file}时出错：{e}")
                else:
                    print(f"智能体{idx} actor在初始化尝试后仍为None。无法从{actor_file}加载。")
            # else:
            #     print(f"智能体{idx}的Actor CSV文件未找到：{actor_file}")
            
            critic_loaded = False
            if os.path.exists(critic_file):
                try:
                    with open(critic_file, 'r') as f:
                        critic_model_params = {name: param for name, param in agent.critic.named_parameters()}
                        reader = csv.reader(f)
                        for row in reader:
                            if not row: continue
                            name_from_csv, *data_str = row
                            if name_from_csv in critic_model_params:
                                param_to_load = critic_model_params[name_from_csv]
                                try:
                                    loaded_data = np.array(data_str, dtype=np.float32)
                                    param_to_load.data.copy_(torch.tensor(loaded_data).view_as(param_to_load.data).to(device))
                                    critic_loaded = True
                                except ValueError as ve:
                                    print(f"智能体{idx} critic参数{name_from_csv}的ValueError：{ve}。数据：{data_str[:10]}")
                                except RuntimeError as re:
                                     print(f"智能体{idx} critic参数{name_from_csv}的RuntimeError：{re}。期望形状{param_to_load.data.shape}")
                            # else:
                            #    print(f"警告：CSV中的参数{name_from_csv}在智能体{idx} critic模型中未找到。")
                    if critic_loaded: print(f"智能体{idx} critic参数从{critic_file}加载")
                except Exception as e:
                    print(f"处理智能体{idx}的critic CSV {critic_file}时出错：{e}")
            # else:
            #    print(f"智能体{idx}的Critic CSV文件未找到：{critic_file}")

        print(f"完成从{folder_path}加载模型参数的尝试。")

                
    def save_to_csv(self, folder_path):
        # 更新self.current_results_folder，以便gradient_norms.csv也保存在这里
        self.current_results_folder = folder_path
        os.makedirs(folder_path, exist_ok=True)

        for idx, agent in enumerate(self.agents):
            # 在保存前确保actor已初始化（例如，通过前向传递）
            if agent.actors is None:
                # print(f"警告：智能体{idx} actor在保存期间为None。尝试初始化。")
                agent._ensure_actors_initialized(idx) # 尝试初始化
            
            if agent.actors is not None:
                actor_params = []
                for name, param in agent.actors.named_parameters():
                    actor_params.append((name, param.data.cpu().numpy()))
                
                actor_file = os.path.join(folder_path, f"agent_{idx}_actor.csv")
                try:
                    with open(actor_file, 'w', newline='') as f: # 确保csv的newline=''
                        writer = csv.writer(f)
                        for name, data in actor_params:
                            writer.writerow([name] + list(data.flatten()))
                except Exception as e:
                    print(f"保存智能体{idx}的actor CSV时出错：{e}")
            # else:
            #     print(f"智能体{idx} actor仍为None。无法保存actor参数。")
            
            critic_params = []
            for name, param in agent.critic.named_parameters():
                critic_params.append((name, param.data.cpu().numpy()))
            
            critic_file = os.path.join(folder_path, f"agent_{idx}_critic.csv")
            try:
                with open(critic_file, 'w', newline='') as f: # 确保newline=''
                    writer = csv.writer(f)
                    for name, data in critic_params:
                        writer.writerow([name] + list(data.flatten()))
            except Exception as e:
                print(f"保存智能体{idx}的critic CSV时出错：{e}")

        print(f"模型参数已保存到{folder_path}中的CSV文件。")


    def plot_metrics(self, folder_path, window_size=1):
        # (plot_metrics代码与修复了NaN/空列表问题的先前版本一致，此处省略以减少重复)
        # ... (确保这里正确使用folder_path并有数据可绘制) ...
        def smooth(data, window_size):
            if len(data) < window_size: return data 
            return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

        os.makedirs(folder_path, exist_ok=True)

        def _plot_metric_internal(data_all_agents, metric_name, color, aggregate_func=np.mean):
            if not data_all_agents or not any(data_all_agents):
                return
            
            valid_data_for_min_len = [d for d in data_all_agents if d] 
            if not valid_data_for_min_len: return
            min_len = min(len(d) for d in valid_data_for_min_len)
            if min_len == 0 : return

            aggregated_data = []
            if aggregate_func == np.sum :
                 aggregated_data = [np.sum([agent_data[i] for agent_data in valid_data_for_min_len if i < len(agent_data)]) for i in range(min_len)]
            else: 
                 aggregated_data = [np.mean([agent_data[i] for agent_data in valid_data_for_min_len if i < len(agent_data)]) for i in range(min_len)]

            if not aggregated_data: return
            smoothed_data = smooth(aggregated_data, window_size)
            if len(smoothed_data) == 0: return

            plt.figure(figsize=(8, 6))
            plt.plot(smoothed_data, label=metric_name, color=color)
            plt.title(metric_name)
            plt.xlabel("更新步骤")
            plt.ylabel(metric_name)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(folder_path, f"{metric_name.replace(' ', '_').lower()}.png"))
            plt.close()

        _plot_metric_internal(self.cumulative_rewards, "累积奖励", "blue", aggregate_func=np.sum)
        if any(self.value_losses):
            _plot_metric_internal(self.value_losses, "价值损失", "orange", aggregate_func=np.mean)
        if any(self.policy_entropy):
            _plot_metric_internal(self.policy_entropy, "策略熵", "green", aggregate_func=np.mean)

        
    def save_metrics_to_csv(self, folder_path):
        # (save_metrics_to_csv代码与修复了NaN/空列表问题的先前版本一致，省略)
        # ... (确保这里正确使用folder_path) ...
        os.makedirs(folder_path, exist_ok=True)
        def _save_csv_aggregated(data_all_agents, metric_name_suffix, aggregate_func=np.mean):
            if not data_all_agents or not any(data_all_agents): return
            valid_data_for_min_len = [d for d in data_all_agents if d]
            if not valid_data_for_min_len: return
            min_len = min(len(d) for d in valid_data_for_min_len)
            if min_len == 0: return

            aggregated_data = []
            if aggregate_func == np.sum:
                aggregated_data = [np.sum([agent_data[i] for agent_data in valid_data_for_min_len if i < len(agent_data)]) for i in range(min_len)]
            else: 
                aggregated_data = [np.mean([agent_data[i] for agent_data in valid_data_for_min_len if i < len(agent_data)]) for i in range(min_len)]

            if not aggregated_data: return
            file_path = os.path.join(folder_path, f"{metric_name_suffix}.csv") # 直接使用后缀作为名称
            with open(file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["步骤", metric_name_suffix])
                for step, value in enumerate(aggregated_data):
                    writer.writerow([step, value])
            # print(f"{metric_name_suffix}已保存到{file_path}")

        _save_csv_aggregated(self.cumulative_rewards, "累积奖励", aggregate_func=np.sum)
        if any(self.value_losses):
            _save_csv_aggregated(self.value_losses, "价值损失", aggregate_func=np.mean)
        if any(self.policy_entropy):
            _save_csv_aggregated(self.policy_entropy, "策略熵", aggregate_func=np.mean)

        
    def save_gradient_norms_to_csv(self, folder_path):
        # (save_gradient_norms_to_csv代码与修复了NaN/空列表问题的先前版本一致，省略)
        # ... (确保这里正确使用folder_path，而不是self.gradient_csv_file) ...
        os.makedirs(folder_path, exist_ok=True)
        # gradient_csv_file_path = os.path.join(folder_path, self.gradient_csv_file_basename)
        # self.gradient_csv_file_basename可能只是"gradient_norms.csv"
        # 我们想将其保存在此次运行的特定folder_path*内*。
        gradient_csv_file_path = os.path.join(folder_path, "gradient_norms.csv")


        max_steps = 0
        if self.gradient_norms and any(self.gradient_norms):
            valid_grad_norms = [gn for gn in self.gradient_norms if gn] 
            if valid_grad_norms:
                 max_steps = max(len(grad_norms) for grad_norms in valid_grad_norms)
            else: return
        else: return
        if max_steps == 0: return

        data_to_write = []
        for step in range(max_steps):
            row = [step]
            for agent_idx in range(self.num_agents):
                if agent_idx < len(self.gradient_norms) and self.gradient_norms[agent_idx] and step < len(self.gradient_norms[agent_idx]):
                    row.append(self.gradient_norms[agent_idx][step])
                else:
                    row.append(None)
            data_to_write.append(row)

        with open(gradient_csv_file_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            header = ["更新步骤"] + [f"智能体{i}梯度范数" for i in range(self.num_agents)]
            writer.writerow(header)
            writer.writerows(data_to_write)
        # print(f"梯度范数已保存到{gradient_csv_file_path}")


    def plot_gradient_norms(self, folder_path, window_size=1):
        # (plot_gradient_norms代码与修复了NaN/空列表问题的先前版本一致，省略)
        # ... (确保这里正确使用folder_path) ...
        def smooth(data, window_size):
            if len(data) < window_size: return data
            return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

        os.makedirs(folder_path, exist_ok=True)

        for agent_idx, grad_norms in enumerate(self.gradient_norms):
            if not grad_norms: continue
            smoothed_grad_norms = smooth(grad_norms, window_size)
            if len(smoothed_grad_norms) == 0: continue

            plt.figure(figsize=(8, 6))
            plt.plot(smoothed_grad_norms, label=f"智能体{agent_idx}梯度范数")
            plt.title(f"梯度范数 - 智能体{agent_idx}")
            plt.xlabel("更新步骤")
            plt.ylabel("梯度范数")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(folder_path, f"gradient_norm_agent_{agent_idx}.png"))
            plt.close()