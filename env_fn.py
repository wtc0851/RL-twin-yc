import heapq
import math
import random
from enum import Enum
from typing import List, Tuple, Optional, Dict, Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt

from .dataclasses import Crane, Task, CraneStatus
from dataclasses import dataclass, field


class EventType(Enum):
    """
    事件驱动模拟中的事件类型
    """
    TASK_GENERATION = 1  # 新任务生成事件
    CRANE_ARRIVAL = 2    # 场桥到达目标贝位事件
    TASK_COMPLETION = 3  # 场桥完成任务事件
    DECISION_REQUEST = 4 # 请求代理做出决策的事件


@dataclass(order=True)
class Event:
    """
    事件驱动模拟中的事件对象
    """
    time: float  # 事件发生的时间
    type: EventType = field(compare=False)  # 事件类型
    data: Dict[str, Any] = field(default_factory=dict, compare=False)  # 事件相关数据


class YardEnv(gym.Env):
    """
    集装箱码头龙门吊调度环境
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, static_tasks=None):
        super().__init__()

        # --- 环境核心参数---
        self.num_bays = 50                       # 贝位数量
        self.num_cranes = 2                      # 场桥数量
        self.crane_speed = 2.5                   # 场桥移动速度 (单位: 贝位/秒)
        self.mean_task_interval = 30.0           # 任务生成间隔 (单位: 秒)
        self.mean_task_execution_time = 60.0     # 任务执行时间 (单位: 秒)
        self.max_simulation_time = 3600.0        # 最大模拟时间 (单位: 秒)  1小时
        self.max_tasks_in_obs = 10               # 代理能观察到的最大任务数量
        self.crane_initial_positions = [1, self.num_bays]       # 场桥初始位置
        # 新增：为测试/评估设置步数上限，防止策略长时间运行
        self.max_episode_steps = 5000

        # --- 静态任务数据支持 ---
        self.static_tasks = static_tasks                        # 静态任务列表
        self.static_task_index = 0                              # 当前静态任务索引

        # --- 奖励设计 ---
        # 新的奖励机制：基于队列时间积分的惩罚
        self.collision_penalty = -500.0          # 代理做出无效动作（如导致碰撞）的惩罚
        self.reward_scale_factor = 100.0         # 奖励归一化缩放因子，用于控制奖励数值范围
        
        # 移除所有阶段性正奖励，采用纯惩罚机制
        # self.wait_penalty = -10.0              # 移除：等待惩罚
        # self.task_acceptance_reward = 50.0     # 移除：接受任务奖励
        # self.arrival_reward = 200.0            # 移除：到达位置奖励  
        # self.task_completion_reward = 750.0    # 移除：完成任务奖励

        # --- 动作空间和状态空间 ---
        self.action_space = spaces.Discrete(self.max_tasks_in_obs + 1)
        self.observation_space = spaces.Dict({
            "crane_status": spaces.Box(low=0, high=np.inf, shape=(self.num_cranes, 2), dtype=np.float32),
            "task_list": spaces.Box(low=0, high=np.inf, shape=(self.max_tasks_in_obs, 3), dtype=np.float32),
            "action_mask": spaces.Box(low=0, high=1, shape=(self.action_space.n,), dtype=np.int8),
            "crane_to_command": spaces.Box(low=0, high=1, shape=(self.num_cranes,), dtype=np.int8),
        })

        # --- 内部状态，初始化reset() ---
        self.current_time = 0.0                                 # 当前模拟时间
        self.crane_to_command: Optional[int] = None             # 当前要命令的场桥ID
        self.cranes: List[Crane] = []                           # 所有场桥
        self.task_queue: List[Task] = []                        # 任务队列
        self.completed_tasks: List[Task] = []                   # 已完成任务列表
        self.event_queue: List[Event] = []                      # 事件队列
        self.history: Dict[int, List[Tuple[float, int]]] = {}   # 场桥轨迹记录 (id -> [(time, pos)])
        self.task_generation_stopped = False
        self.static_task_index = 0                              # 重置静态任务索引
        # 新增：步数计数器
        self.episode_steps = 0
        
        # --- 性能指标收集 ---
        self.total_task_wait_time = 0.0                         # 任务总等待时间
        self.crane_move_count = 0                               # 场桥移动次数
        self.total_crane_move_time = 0.0                        # 场桥移动总时间
        self.total_crane_wait_time = 0.0                        # 场桥等待总时间

        # 场桥状态映射字典
        self._crane_status_map = {status: i for i, status in enumerate(CraneStatus)}
        # 渲染模式
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        # 为等待动作推进的微小时间步长（秒）
        self.wait_time_step = 1.0
        # 固定等待后重新决策的时长（秒）
        self.decision_wait_seconds = 30.0
        # 势函数塑形参数（不改变最优策略）：
        # 使用形如 R' = R + beta * (gamma * F(s') - F(s)) 的势函数塑形项。
        # 其中 F(s) 定义为“状态潜在值”，越大越好；我们用用户给定的代价 G(s)=avg_wait_norm+queue_len_norm，令 F(s)=-G(s)。
        self.shaping_gamma = 0.99
        self.shaping_beta = 1.0
        self.potential_weights = (0.5, 0.5)  # (w1, w2) for avg_wait_norm and queue_len_norm
        self.prev_potential = None  # 在首次决策点初始化

    def _generate_new_task(self):
        """生成一个新任务并将其放入任务队列，同时安排下一次任务生成事件。"""
        
        # 如果使用静态任务数据
        if self.static_tasks is not None:
            return self._generate_static_task()
        
        # 原有的随机任务生成逻辑
        # 1. 计算下一个任务的到达时间 todo：考虑任务生成间隔的预测
        next_task_arrival_time = self.current_time + random.expovariate(
            1.0 / self.mean_task_interval
        )

        # 2. 如果在模拟时间内且任务生成未停止，则生成事件
        if next_task_arrival_time < self.max_simulation_time and not self.task_generation_stopped:
            heapq.heappush(
                self.event_queue,
                Event(time=next_task_arrival_time, type=EventType.TASK_GENERATION, data={}),
            )
        elif next_task_arrival_time >= self.max_simulation_time:
            # 如果超过最大模拟时间，停止任务生成
            self.task_generation_stopped = True

        # 3. 创建新任务
        task_id = len(self.completed_tasks) + len(self.task_queue)
        location = random.randint(1, self.num_bays)
        available_time = self.current_time
        execution_time = random.expovariate(1.0 / self.mean_task_execution_time)

        new_task = Task(
            id=task_id,
            location=location,
            available_time=available_time,
            execution_time=execution_time,
            creation_time=self.current_time,
        )

        # 4. 将新任务添加到待处理队列
        self.task_queue.append(new_task)

    def _generate_static_task(self):
        """从静态任务列表中生成任务"""
        # 检查是否还有静态任务
        if self.static_task_index >= len(self.static_tasks):
            # 所有静态任务都已生成，停止任务生成
            self.task_generation_stopped = True
            return

        # 获取下一个静态任务数据
        static_task_data = self.static_tasks[self.static_task_index]
        
        # 计算下一个任务的到达时间
        next_task_arrival_time = static_task_data['available_time']
        
        # 如果任务到达时间还未到，安排任务生成事件
        if next_task_arrival_time > self.current_time:
            heapq.heappush(
                self.event_queue,
                Event(time=next_task_arrival_time, type=EventType.TASK_GENERATION, data={}),
            )
            return

        # 创建静态任务
        task_id = len(self.completed_tasks) + len(self.task_queue)
        new_task = Task(
            id=task_id,
            location=static_task_data['location'],
            available_time=static_task_data['available_time'],
            execution_time=static_task_data['execution_time'],
            creation_time=self.current_time,
        )

        # 将任务添加到队列
        self.task_queue.append(new_task)
        self.static_task_index += 1


        # 如果还有更多静态任务，安排下一个任务生成事件
        if self.static_task_index < len(self.static_tasks):
            next_static_task = self.static_tasks[self.static_task_index]
            next_arrival_time = next_static_task['available_time']
            
            if next_arrival_time < self.max_simulation_time:
                heapq.heappush(
                    self.event_queue,
                    Event(time=next_arrival_time, type=EventType.TASK_GENERATION, data={}),
                )
            else:
                self.task_generation_stopped = True

    def _is_task_feasible_for_crane(self, crane_id: int, task: Task) -> bool:
        """基于当前碰撞规则判断某个任务对指定场桥是否可执行。"""
        commanding_crane = self.cranes[crane_id]
        other_crane = self.cranes[1 - crane_id]
        # 动态障碍区间
        if other_crane.status == CraneStatus.MOVING:
            other_crane_barrier_min = min(other_crane.location, other_crane.current_task.location)
            other_crane_barrier_max = max(other_crane.location, other_crane.current_task.location)
        else:
            other_crane_barrier_min = other_crane.location
            other_crane_barrier_max = other_crane.location
        task_loc = task.location
        # 应用与 _get_action_mask 相同的边界规则（已放宽为严格不等式）
        if crane_id == 0:
            return not (max(commanding_crane.location, task_loc) > other_crane_barrier_min)
        else:
            return not (min(commanding_crane.location, task_loc) < other_crane_barrier_max)

    def _get_visible_tasks(self) -> List[Task]:
        """返回当前可见任务集合。
        优先包含对当前待指令场桥可执行的任务，避免仅显示不可执行任务导致持续等待。
        """
        # 先按 available_time 排序所有任务
        sorted_tasks = sorted(self.task_queue, key=lambda t: t.available_time)
        if self.crane_to_command is None:
            # 没有待指令场桥时，保持旧行为
            return sorted_tasks[:self.max_tasks_in_obs]
        # 优先挑选对当前场桥可执行的任务
        feasible = []
        infeasible = []
        for t in sorted_tasks:
            if self._is_task_feasible_for_crane(self.crane_to_command, t):
                feasible.append(t)
            else:
                infeasible.append(t)
        # 组装可见列表：先放可执行，再补不可执行，保证长度与稳定性
        visible = []
        visible.extend(feasible[:self.max_tasks_in_obs])
        if len(visible) < self.max_tasks_in_obs:
            need = self.max_tasks_in_obs - len(visible)
            visible.extend(infeasible[:need])
        return visible

    def _get_observation(self, crane_to_command_id: int) -> Dict:
        """将当前环境状态转换为 agent 需要的观测值 (numpy array)。"""
        # 1. 场桥状态
        crane_status_list = []
        for crane in self.cranes:
            crane_status_list.append([crane.location, self._crane_status_map[crane.status]])
        
        # 补齐到固定长度
        while len(crane_status_list) < self.num_cranes:
            crane_status_list.append([0, 0])

        # 2. 任务列表（使用可见任务，available_time 顺序）
        visible_tasks = self._get_visible_tasks()
        task_list_obs = []
        for task in visible_tasks:
            task_list_obs.append(
                [task.location, task.available_time, task.execution_time]
            )
        
        # 用零填充以达到固定长度
        while len(task_list_obs) < self.max_tasks_in_obs:
            task_list_obs.append([0, 0, 0])

        # 3. 动作掩码
        # 当 crane_to_command 为空时，至少允许“等待”以推进时间
        if crane_to_command_id is None or not (0 <= crane_to_command_id < self.num_cranes):
            action_mask = np.zeros(self.action_space.n, dtype=np.int8)
            action_mask[-1] = 1
        else:
            action_mask = self._get_action_mask(crane_to_command_id)

        # 4. 待指令的场桥ID (One-hot)
        crane_to_command_obs = [0] * self.num_cranes
        if crane_to_command_id is not None and 0 <= crane_to_command_id < self.num_cranes:
            crane_to_command_obs[crane_to_command_id] = 1

        # 5. 组合成字典, 并转换为 numpy array 以符合 gym 的格式要求
        observation = {
            "crane_status": np.array(crane_status_list, dtype=np.float32),
            "task_list": np.array(task_list_obs, dtype=np.float32),
            "action_mask": np.array(action_mask, dtype=np.int8),
            "crane_to_command": np.array(crane_to_command_obs, dtype=np.int8),
        }
        return observation

    def _get_action_mask(self, crane_id: int) -> np.ndarray:
        mask = np.zeros(self.action_space.n, dtype=np.int8)
        # 当没有待指令的场桥时，保持至少有“等待”动作可选
        if self.crane_to_command is None:
            mask[-1] = 1
            return mask

        commanding_crane = self.cranes[crane_id]
        other_crane = self.cranes[1 - crane_id]

        # --- 更严格的防碰撞逻辑 ---
        # 将另一台场桥视为动态障碍物，其整个路径（如果是移动状态）都是禁区
        if other_crane.status == CraneStatus.MOVING:
            # 如果另一台场桥正在移动，它的整个路径都是障碍
            other_crane_barrier_min = min(other_crane.location, other_crane.current_task.location)
            other_crane_barrier_max = max(other_crane.location, other_crane.current_task.location)
        else:
            # 如果另一台场桥是静止的，它的位置就是障碍点
            other_crane_barrier_min = other_crane.location
            other_crane_barrier_max = other_crane.location

        visible_tasks = self._get_visible_tasks()
        for i, task in enumerate(visible_tasks):
            is_collision = False
            task_loc = task.location

            # 规则：0号场桥必须始终在1号场桥的左边（或同一位置）
            if crane_id == 0:  # 我们正在指令左边的0号场桥
                # 它的目标位置，以及它路径上的最右点，都不能越过1号场桥的活动范围的最左点
                # 允许“等于”边界（两桥同位置时可以向外分离）
                if max(commanding_crane.location, task_loc) > other_crane_barrier_min:
                    is_collision = True
            else:  # 我们正在指令右边的1号场桥
                # 它的目标位置，以及它路径上的最左点，都不能越过0号场桥的活动范围的最右点
                # 允许“等于”边界（两桥同位置时可以向外分离）
                if min(commanding_crane.location, task_loc) < other_crane_barrier_max:
                    is_collision = True
            
            if not is_collision:
                mask[i] = 1

        # 始终允许等待动作，以保留探索能力
        # 如果存在至少一个可执行任务，则不允许“等待”，强制执行任务以避免无限等待
        # 如果存在至少一个可执行任务，则不允许“等待”，否则允许“等待”
        if np.any(mask[:self.max_tasks_in_obs] == 1):
            mask[-1] = 0
        else:
            mask[-1] = 1
        
        return mask

    def action_mask(self) -> np.ndarray:
        """
        为当前待指令的场桥生成动作掩码。
        此方法由 ActionMasker 包装器需要。
        """
        if self.crane_to_command is None:
            # 如果没有正在指令的场桥，返回仅允许“等待”的掩码，避免策略无动作可选
            mask = np.zeros(self.action_space.n, dtype=np.int8)
            mask[-1] = 1
            return mask
        return self._get_action_mask(self.crane_to_command)

    def _get_info(self):
        """返回环境的额外信息，包括性能指标。"""
        avg_wait_norm, queue_len_norm = self._compute_state_metrics()
        return {
            "simulation_time": self.current_time,
            "pending_tasks_count": len(self.task_queue),
            "completed_tasks_count": len(self.completed_tasks),
            "crane_to_command": self.crane_to_command,
            "total_task_wait_time": self.total_task_wait_time,
            "crane_move_count": self.crane_move_count,
            "total_crane_move_time": self.total_crane_move_time,
            "total_crane_wait_time": self.total_crane_wait_time,
            # 势函数相关的监控指标（便于TensorBoard查看）：
            "avg_wait_norm": float(avg_wait_norm),
            "queue_len_norm": float(queue_len_norm),
            "state_potential": float(self._compute_state_potential())
        }

    def _compute_state_metrics(self) -> Tuple[float, float]:
        """
        计算状态代价的两个组成：
        - avg_wait_norm: 当前队列中任务的平均等待（秒）经归一化到[0,1]
        - queue_len_norm: 队列长度相对于可见最大任务数归一化到[0,1]
        """
        # 平均等待（仅统计已到达的任务）
        waits = [max(0.0, self.current_time - t.available_time) for t in self.task_queue if t.available_time <= self.current_time]
        avg_wait = (sum(waits) / len(waits)) if len(waits) > 0 else 0.0
        # 归一化尺度：取两倍平均执行时长，避免过敏感
        norm_wait_scale = max(self.mean_task_execution_time * 2.0, 1e-6)
        avg_wait_norm = min(avg_wait / norm_wait_scale, 1.0)

        # 队列长度归一化
        queue_len_norm = min(len(self.task_queue) / float(max(self.max_tasks_in_obs, 1)), 1.0)
        return avg_wait_norm, queue_len_norm

    def _compute_state_potential(self) -> float:
        """
        势函数 F(s)：越大越好，用于势函数塑形项。
        用户希望的评分是代价 G(s)=avg_wait_norm+queue_len_norm（越大越差）。
        为保持最优策略不变，采用 R' = R + beta * (gamma * F(s') - F(s))，因此取 F(s) = -G(s)。
        """
        w1, w2 = self.potential_weights
        avg_wait_norm, queue_len_norm = self._compute_state_metrics()
        state_cost = w1 * avg_wait_norm + w2 * queue_len_norm
        return -state_cost

    def _compute_next_decision_time(self, crane_id: int) -> float:
        future_event_times = [e.time for e in self.event_queue if e.time > self.current_time]
        next_event_time = min(future_event_times) if future_event_times else None
        future_available_times = [t.available_time for t in self.task_queue if t.available_time > self.current_time]
        next_task_time = min(future_available_times) if future_available_times else None
        candidates = [t for t in [next_event_time, next_task_time] if t is not None]
        if not candidates:
            return self.current_time + self.wait_time_step
        # 确保下一决策时间严格大于当前时间，避免“等待”后时间不推进导致无限等待
        candidate = min(candidates)
        if candidate <= self.current_time:
            return self.current_time + self.wait_time_step
        return candidate

    def _apply_action(self, action: int) -> float:
        """根据代理的动作，更新环境状态并返回即时奖励。"""
        crane_id = self.crane_to_command
        crane = self.cranes[crane_id]
        reward = 0

        # 检查动作是否有效
        action_mask = self._get_action_mask(crane_id)
        if action_mask[action] == 0:
            # 惩罚无效动作并终止
            # 碰撞惩罚也需要相应缩放以保持一致性
            return self.collision_penalty / self.reward_scale_factor

        # 动作是“等待”
        wait_action_index = self.max_tasks_in_obs
        if action == wait_action_index:
            # 记录当前状态，以便在图表中显示
            self.history[crane.id].append((self.current_time, crane.location))
            crane.status = CraneStatus.IDLE

            # 固定原地等待5秒，然后安排下一次决策事件
            next_decision_time = self.current_time + self.decision_wait_seconds
            delta_wait = max(0.0, next_decision_time - self.current_time)
            self.total_crane_wait_time += delta_wait
            # print(f"[WAIT] t={self.current_time:.2f}s crane={crane_id} wait={self.decision_wait_seconds:.1f}s -> next_decision={next_decision_time:.2f}s, pending_tasks={len(self.task_queue)}")

            # 如果已经没有未来事件/任务，且没有待处理任务且两台场桥都空闲（且任务生成已停止），不再安排新的决策事件，交由事件循环终止
            future_event_times_non_decision = [
                e.time for e in self.event_queue
                if e.time > self.current_time and e.type != EventType.DECISION_REQUEST
            ]
            future_available_times = [t.available_time for t in self.task_queue if t.available_time > self.current_time]
            all_cranes_idle = all(c.status == CraneStatus.IDLE for c in self.cranes)
            no_pending_tasks = len(self.task_queue) == 0
            generation_fully_stopped = (
                self.task_generation_stopped or (self.static_tasks is not None and self.static_task_index >= len(self.static_tasks))
            )

            if not future_event_times_non_decision and not future_available_times and all_cranes_idle and no_pending_tasks and generation_fully_stopped:
                # 不推送新的 DECISION_REQUEST，让 _advance_simulation 检查并正常终止
                pass
            else:
                heapq.heappush(
                    self.event_queue,
                    Event(time=next_decision_time, type=EventType.DECISION_REQUEST, data={"crane_id": crane_id}),
                )

        else:
            # 将动作索引映射到同序的可见任务，定位真实队列索引
            visible_tasks = self._get_visible_tasks()
            if action >= len(visible_tasks):
                return self.collision_penalty / self.reward_scale_factor
            selected_task = visible_tasks[action]
            # 通过唯一 id 定位在 self.task_queue 中的索引
            idx_in_queue = next(i for i, t in enumerate(self.task_queue) if t.id == selected_task.id)
            task_to_assign = self.task_queue.pop(idx_in_queue)

            # 记录移动开始时的位置
            self.history[crane.id].append((self.current_time, crane.location))

            crane.status = CraneStatus.MOVING
            crane.current_task = task_to_assign

            travel_distance = abs(crane.location - task_to_assign.location)
            travel_time = travel_distance / self.crane_speed
            arrival_time = self.current_time + travel_time
            
            # 记录场桥移动指标
            if travel_distance > 0:
                self.crane_move_count += 1
                self.total_crane_move_time += travel_time

            heapq.heappush(
                self.event_queue,
                Event(time=arrival_time, type=EventType.CRANE_ARRIVAL, data={"crane_id": crane_id}),
            )

        self.crane_to_command = None
        return reward

    def reset(self, seed=None, options=None) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)

        self.current_time = 0.0
        self.next_task_id = 0
        self.task_generation_stopped = False  # 重置任务生成标志
        # 重置静态任务索引
        self.static_task_index = 0
        
        # 重置性能指标
        self.total_task_wait_time = 0.0
        self.crane_move_count = 0
        self.total_crane_move_time = 0.0
        self.total_crane_wait_time = 0.0
        
        # 初始化累积奖励（用于队列时间积分惩罚）
        self.accumulated_reward = 0.0
        self.prev_time = 0.0  # 记录上一次事件的时间
        # 重置步数
        self.episode_steps = 0

        self.cranes = [
            Crane(id=0, location=self.crane_initial_positions[0], status=CraneStatus.IDLE),
            Crane(id=1, location=self.crane_initial_positions[1], status=CraneStatus.IDLE)
        ]

        self.task_queue: List[Task] = []
        self.completed_tasks: List[Task] = []

        self.history = {i: [(0, self.cranes[i].location)] for i in range(self.num_cranes)}

        self.event_queue: List[Event] = []
        first_task_time = random.uniform(0, self.mean_task_interval * 2)
        heapq.heappush(self.event_queue, Event(time=first_task_time, type=EventType.TASK_GENERATION, data={}))

        # 初始时，两台场桥都请求决策
        heapq.heappush(self.event_queue, Event(time=0.0, type=EventType.DECISION_REQUEST, data={"crane_id": 0}))
        heapq.heappush(self.event_queue, Event(time=0.0, type=EventType.DECISION_REQUEST, data={"crane_id": 1}))

        # 在reset后，我们直接处理事件直到第一个决策点
        (
            _,  # crane_to_command is in info
            observation,
            _,  # reward is 0
            _,  # terminated is false
            _,  # truncated is false
            info,
        ) = self._advance_simulation()
        return observation, info

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """执行一个动作，并让环境前进到下一个需要决策的时间点。"""
        # 1. 接收动作并对环境做出即时变更
        reward = self._apply_action(action)
        # 增加步数（仅统计，不做截断）
        self.episode_steps += 1

        # 2. 推进仿真，直到下一个决策点或仿真结束
        (
            _,
            observation,
            event_reward,
            terminated,
            truncated,
            info,
        ) = self._advance_simulation()

        total_reward = reward + event_reward

        # 3. 步数截断：防止策略长时间停留在等待循环
        should_truncate = self.episode_steps >= self.max_episode_steps
        if should_truncate:
            info["termination_reason"] = "max_episode_steps_reached"
            truncated = True

        return (
            observation,
            total_reward,
            terminated,
            truncated,
            info,
        )

    def _advance_simulation(self) -> Tuple[int, Dict, float, bool, bool, Dict]:
        """事件循环，处理事件直到需要下一个决策或仿真结束。"""
        # 使用实例变量来累积奖励，实现队列时间积分惩罚
        
        # 事件循环，直到需要下一个决策或队列为空
        while self.event_queue:
            # 1. 从事件队列中取出下一个最近的事件
            event = heapq.heappop(self.event_queue)
            
            # 更新当前时间（移除原按队列长度积分的惩罚，改用势函数塑形降低方差）
            self.current_time = event.time
            self.prev_time = event.time

            # 3. 达到最大模拟时长后：停止任务生成，但继续处理现有事件/任务直到清空积压
            if self.current_time >= self.max_simulation_time:
                self.task_generation_stopped = True
                # 不截断，继续按事件类型处理，允许完成已到达的任务

            # 4. 根据事件类型处理事件
            if event.type == EventType.CRANE_ARRIVAL:
                crane_id = event.data["crane_id"]
                crane = self.cranes[crane_id]
                crane.status = CraneStatus.BUSY
                crane.location = crane.current_task.location
                self.history[crane.id].append((self.current_time, crane.location))
                completion_time = self.current_time + crane.current_task.execution_time
                crane.free_at = completion_time
                heapq.heappush(
                    self.event_queue,
                    Event(time=completion_time, type=EventType.TASK_COMPLETION, data={"crane_id": crane_id}),
                )
            elif event.type == EventType.TASK_COMPLETION:
                crane_id = event.data["crane_id"]
                crane = self.cranes[crane_id]
                completed_task = crane.current_task
                task_wait_time = self.current_time - completed_task.creation_time
                self.total_task_wait_time += task_wait_time
                self.completed_tasks.append(completed_task)
                crane.status = CraneStatus.IDLE
                crane.current_task = None
                crane.free_at = self.current_time
                self.history[crane.id].append((self.current_time, crane.location))
                heapq.heappush(
                    self.event_queue,
                    Event(time=self.current_time, type=EventType.DECISION_REQUEST, data={"crane_id": crane_id}),
                )
                for other_crane in self.cranes:
                    if other_crane.id != crane.id and other_crane.status == CraneStatus.IDLE:
                        heapq.heappush(
                            self.event_queue,
                            Event(time=self.current_time, type=EventType.DECISION_REQUEST, data={"crane_id": other_crane.id})
                        )
            elif event.type == EventType.TASK_GENERATION:
                self._generate_new_task()
                for crane in self.cranes:
                    if crane.status == CraneStatus.IDLE:
                        heapq.heappush(
                            self.event_queue,
                            Event(time=self.current_time, type=EventType.DECISION_REQUEST, data={"crane_id": crane.id})
                        )
            elif event.type == EventType.DECISION_REQUEST:
                crane_id_to_command = event.data["crane_id"]
                if self.cranes[crane_id_to_command].status == CraneStatus.IDLE:
                    # 如果当前场桥无可执行任务（除等待外），而另一台场桥可执行任务，则优先切换到另一台场桥，避免无限等待
                    # 先为当前场桥设置待指令，以便正确计算动作掩码
                    self.crane_to_command = crane_id_to_command
                    current_mask = self._get_action_mask(crane_id_to_command)
                    has_executable = np.any(current_mask[:-1] == 1)
                    if not has_executable:
                        other_id = 1 - crane_id_to_command
                        if self.cranes[other_id].status == CraneStatus.IDLE:
                            # 切换到另一台场桥并检查其动作掩码
                            self.crane_to_command = other_id
                            other_mask = self._get_action_mask(other_id)
                            other_has_exec = np.any(other_mask[:-1] == 1)
                            if other_has_exec:
                                observation = self._get_observation(other_id)
                                # 计算势函数塑形增量：beta * (gamma * F(s') - F(s))
                                new_potential = self._compute_state_potential()
                                if self.prev_potential is None:
                                    shaping = 0.0
                                else:
                                    shaping = self.shaping_beta * (self.shaping_gamma * new_potential - self.prev_potential)
                                self.prev_potential = new_potential

                                info = self._get_info()
                                info["reward_shaping"] = float(shaping)
                                reward_to_return = self.accumulated_reward + shaping
                                self.accumulated_reward = 0.0
                                return other_id, observation, reward_to_return, False, False, info
                    # 默认返回当前场桥的决策
                    self.crane_to_command = crane_id_to_command
                    observation = self._get_observation(crane_id_to_command)
                    # 计算势函数塑形增量：beta * (gamma * F(s') - F(s))
                    new_potential = self._compute_state_potential()
                    if self.prev_potential is None:
                        shaping = 0.0
                    else:
                        shaping = self.shaping_beta * (self.shaping_gamma * new_potential - self.prev_potential)
                    self.prev_potential = new_potential

                    info = self._get_info()
                    info["reward_shaping"] = float(shaping)
                    reward_to_return = self.accumulated_reward + shaping
                    self.accumulated_reward = 0.0
                    return crane_id_to_command, observation, reward_to_return, False, False, info

        # 如果事件队列为空，检查是否所有任务都已完成
        all_cranes_idle = all(crane.status == CraneStatus.IDLE for crane in self.cranes)
        no_pending_tasks = len(self.task_queue) == 0
        
        if all_cranes_idle and no_pending_tasks:
            final_obs = self._get_observation(0)
            final_info = self._get_info()
            # 若在达到时间上限后清空积压，给出更准确的终止原因
            if self.current_time >= self.max_simulation_time:
                final_info["termination_reason"] = "all_tasks_completed_after_time_limit"
            else:
                final_info["termination_reason"] = "all_tasks_completed"
            # 终止时也结算一次势函数塑形增量
            new_potential = self._compute_state_potential()
            if self.prev_potential is None:
                shaping = 0.0
            else:
                shaping = self.shaping_beta * (self.shaping_gamma * new_potential - self.prev_potential)
            final_info["reward_shaping"] = float(shaping)
            return 0, final_obs, self.accumulated_reward + shaping, True, False, final_info
        else:
            final_obs = self._get_observation(0)
            final_info = self._get_info()
            final_info["termination_reason"] = "event_queue_empty_with_pending_tasks"
            # 终止时也结算一次势函数塑形增量
            new_potential = self._compute_state_potential()
            if self.prev_potential is None:
                shaping = 0.0
            else:
                shaping = self.shaping_beta * (self.shaping_gamma * new_potential - self.prev_potential)
            final_info["reward_shaping"] = float(shaping)
            return 0, final_obs, self.accumulated_reward + shaping, True, False, final_info

    def plot_crane_trajectories(self, save_path="crane_trajectory.png"):
        """
        渲染环境状态，生成并保存一张包含两个场桥行走轨迹的图表。
        """
        if self.render_mode != "human":
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        colors = ['blue', 'red']
        linestyles = ['--', '-']
        labels = ['ARMG1', 'ARMG2']

        for crane_id in range(self.num_cranes):
            history = self.history[crane_id]
            if not history:
                continue

            # 将历史数据解压成时间和位置列表
            times, positions = zip(*history)

            # 绘制轨迹
            ax.plot(times, positions, color=colors[crane_id], linestyle=linestyles[crane_id], label=labels[crane_id])

        ax.set_xlabel("Time")
        ax.set_ylabel("Bay")
        ax.set_title("Crane Trajectory")
        ax.legend()
        ax.grid(True)

        # 设置y轴范围
        ax.set_ylim(0, self.num_bays)

        plt.savefig(save_path)
        plt.close(fig)


    def close(self):
        pass