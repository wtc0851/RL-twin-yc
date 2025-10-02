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

    def __init__(self, render_mode=None):
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

        # --- 奖励设计 ---
        self.wait_penalty = -10.0                # 代理选择“等待”动作的惩罚
        self.collision_penalty = -500.0          # 代理做出无效动作（如导致碰撞）的惩罚
        self.task_acceptance_reward = 50.0       # 奖励1: 接受任务
        self.arrival_reward = 200.0              # 奖励2: 到达位置
        self.task_completion_reward = 750.0      # 奖励3: 完成任务 (总奖励 50+200+750=1000)

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

        # 场桥状态映射字典
        self._crane_status_map = {status: i for i, status in enumerate(CraneStatus)}
        # 渲染模式
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _generate_new_task(self):
        """生成一个新任务并将其放入任务队列，同时安排下一次任务生成事件。"""
        # 1. 计算下一个任务的到达时间 todo：考虑任务生成间隔的预测
        next_task_arrival_time = self.current_time + random.expovariate(
            1.0 / self.mean_task_interval
        )

        # 2. 如果在模拟时间内，则生成事件
        if next_task_arrival_time < self.max_simulation_time:
            heapq.heappush(
                self.event_queue,
                Event(time=next_task_arrival_time, type=EventType.TASK_GENERATION, data={}),
            )

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
        print(
            f"[{self.current_time:.2f}s] 新任务 {task_id} 到达位置 {location}."
        )

    def _get_observation(self, crane_to_command_id: int) -> Dict:
        """将当前环境状态转换为 agent 需要的观测值 (numpy array)。"""
        # 1. 场桥状态
        crane_status_list = []
        for crane in self.cranes:
            crane_status_list.append([crane.location, self._crane_status_map[crane.status]])
        
        # 补齐到固定长度
        while len(crane_status_list) < self.num_cranes:
            crane_status_list.append([0, 0])

        # 2. 任务列表 (按创建时间排序)
        self.task_queue.sort(key=lambda t: t.creation_time)
        task_list_obs = []
        for task in self.task_queue[: self.max_tasks_in_obs]:
            task_list_obs.append(
                [task.location, task.available_time, task.execution_time]
            )
        
        # 用零填充以达到固定长度
        while len(task_list_obs) < self.max_tasks_in_obs:
            task_list_obs.append([0, 0, 0])

        # 3. 动作掩码
        action_mask = self._get_action_mask(crane_to_command_id)

        # 4. 待指令的场桥ID (One-hot)
        crane_to_command_obs = [0] * self.num_cranes
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
        if self.crane_to_command is None:
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

        sorted_tasks = sorted(self.task_queue, key=lambda t: t.available_time)[:self.max_tasks_in_obs]
        for i, task in enumerate(sorted_tasks):
            is_collision = False
            task_loc = task.location

            # 规则：0号场桥必须始终在1号场桥的左边（或同一位置）
            if crane_id == 0:  # 我们正在指令左边的0号场桥
                # 它的目标位置，以及它路径上的最右点，都不能越过1号场桥的活动范围的最左点
                if max(commanding_crane.location, task_loc) >= other_crane_barrier_min:
                    is_collision = True
            else:  # 我们正在指令右边的1号场桥
                # 它的目标位置，以及它路径上的最左点，都不能越过0号场桥的活动范围的最右点
                if min(commanding_crane.location, task_loc) <= other_crane_barrier_max:
                    is_collision = True
            
            is_available = task.available_time <= self.current_time

            if not is_collision and is_available:
                mask[i] = 1

        mask[-1] = 1
        return mask

    def action_mask(self) -> np.ndarray:
        """
        为当前待指令的场桥生成动作掩码。
        此方法由 ActionMasker 包装器需要。
        """
        if self.crane_to_command is None:
            # 如果没有正在指令的场桥，理论上不应该调用此方法。
            # 为安全起见，返回一个禁止所有动作的掩码。
            return np.zeros(self.action_space.n, dtype=np.int8)
        return self._get_action_mask(self.crane_to_command)

    def _get_info(self):
        return {
            "simulation_time": self.current_time,
            "pending_tasks_count": len(self.task_queue),
            "completed_tasks_count": len(self.completed_tasks),
            "crane_to_command": self.crane_to_command
        }

    def reset(self, seed=None, options=None) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)

        self.current_time = 0.0
        self.next_task_id = 0

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

        # 2. 推进仿真，直到下一个决策点或仿真结束
        # 这个函数封装了事件循环的复杂性
        (
            _,  # next_crane_to_command is now in info
            observation,
            event_reward,
            terminated,
            truncated,
            info,
        ) = self._advance_simulation()

        # 3. 组合奖励并返回
        total_reward = reward + event_reward

        return (
            observation,
            total_reward,
            terminated,
            truncated,
            info,
        )

    def _apply_action(self, action: int) -> float:
        """根据代理的动作，更新环境状态并返回即时奖励。"""
        crane_id = self.crane_to_command
        crane = self.cranes[crane_id]
        reward = 0

        # 检查动作是否有效
        action_mask = self._get_action_mask(crane_id)
        if action_mask[action] == 0:
            # 惩罚无效动作并终止
            return self.collision_penalty

        # 动作是“等待”
        wait_action_index = self.max_tasks_in_obs
        if action == wait_action_index:
            # 即使是等待，也记录当前状态，以便在图表中显示
            self.history[crane.id].append((self.current_time, crane.location))
            crane.status = CraneStatus.IDLE
            reward = self.wait_penalty
            print(f"[{self.current_time:.2f}s] 场桥 {crane_id} 选择等待")
        # 动作是“分配任务”
        else:
            task_to_assign = self.task_queue.pop(action)

            # 记录移动开始时的位置
            self.history[crane.id].append((self.current_time, crane.location))

            crane.status = CraneStatus.MOVING
            crane.current_task = task_to_assign

            travel_distance = abs(crane.location - task_to_assign.location)
            travel_time = travel_distance / self.crane_speed
            arrival_time = self.current_time + travel_time
            
            # --- 分阶段奖励 1: 接受任务 ---
            reward += self.task_acceptance_reward

            heapq.heappush(
                self.event_queue,
                Event(time=arrival_time, type=EventType.CRANE_ARRIVAL, data={"crane_id": crane_id}),
            )
            print(
                f"[{self.current_time:.2f}s] 场桥 {crane_id} 分配了任务 {task_to_assign.id} 在位置 {task_to_assign.location}. ETA: {arrival_time:.2f}s. 奖励: +{self.task_acceptance_reward:.2f}"
            )

        self.crane_to_command = None
        return reward

    def _advance_simulation(self) -> Tuple[int, Dict, float, bool, bool, Dict]:
        """事件循环，处理事件直到需要下一个决策或仿真结束。"""
        accumulated_reward = 0

        # 事件循环，直到需要下一个决策或队列为空
        while self.event_queue:
            # 1. 从事件队列中取出下一个最近的事件
            event = heapq.heappop(self.event_queue)
            self.current_time = event.time

            # 2. 检查是否超时
            if self.current_time >= self.max_simulation_time:
                print(f"[{self.current_time:.2f}s] 仿真时间已到.")
                # 在超时的情况下，我们需要为某个agent提供一个最终的观测值
                # 这里我们简单地选择0号crane，并提供一个全零的观测
                final_obs = self._get_observation(0)
                return 0, final_obs, accumulated_reward, False, True, {"termination_reason": "timeout"}

            # 3. 根据事件类型处理事件
            # --------------------------
            # --- 场桥到达事件 ---
            if event.type == EventType.CRANE_ARRIVAL:
                crane_id = event.data["crane_id"]
                crane = self.cranes[crane_id]
                crane.status = CraneStatus.BUSY
                crane.location = crane.current_task.location

                # --- 分阶段奖励 2: 到达位置 ---
                accumulated_reward += self.arrival_reward

                # 记录到达时的位置
                self.history[crane.id].append((self.current_time, crane.location))

                completion_time = self.current_time + crane.current_task.execution_time
                crane.free_at = completion_time

                heapq.heappush(
                    self.event_queue,
                    Event(time=completion_time, type=EventType.TASK_COMPLETION, data={"crane_id": crane_id}),
                )
                print(
                    f"[{self.current_time:.2f}s] 场桥 {crane_id} 到达位置 {crane.location} 并开始工作. 奖励: +{self.arrival_reward:.2f}"
                )

            # --------------------------
            # --- 任务完成事件 ---
            elif event.type == EventType.TASK_COMPLETION:
                crane_id = event.data["crane_id"]
                crane = self.cranes[crane_id]
                completed_task = crane.current_task

                # --- 分阶段奖励 3: 完成任务 (减去等待时间的惩罚) ---
                task_wait_time = self.current_time - completed_task.creation_time
                reward = self.task_completion_reward - task_wait_time
                accumulated_reward += reward

                print(
                    f"[{self.current_time:.2f}s] 场桥 {crane_id} 完成任务 {completed_task.id}. 等待时间: {task_wait_time:.2f}s. 奖励: {reward:.2f}"
                )

                # 更新状态
                self.completed_tasks.append(completed_task)
                crane.status = CraneStatus.IDLE
                crane.current_task = None
                crane.free_at = self.current_time

                # 记录任务完成时的位置（位置不变，但时间更新）
                self.history[crane.id].append((self.current_time, crane.location))

                # 为这个刚空闲的crane请求决策
                heapq.heappush(
                    self.event_queue,
                    Event(time=self.current_time, type=EventType.DECISION_REQUEST, data={"crane_id": crane_id}),
                )

                # 检查是否有其他空闲的crane，并为它们请求决策（因为环境变了）
                for other_crane in self.cranes:
                    if other_crane.id != crane.id and other_crane.status == CraneStatus.IDLE:
                        heapq.heappush(
                            self.event_queue,
                            Event(time=self.current_time, type=EventType.DECISION_REQUEST, data={"crane_id": other_crane.id})
                        )

            # --------------------------
            # --- 新任务生成事件 ---
            elif event.type == EventType.TASK_GENERATION:
                self._generate_new_task()
                # 检查是否有空闲的crane，并为它们请求决策
                for crane in self.cranes:
                    if crane.status == CraneStatus.IDLE:
                        heapq.heappush(
                            self.event_queue,
                            Event(time=self.current_time, type=EventType.DECISION_REQUEST, data={"crane_id": crane.id})
                        )

            # --------------------------
            # --- 决策请求事件 ---
            elif event.type == EventType.DECISION_REQUEST:
                crane_id_to_command = event.data["crane_id"]
                # 检查这个决策请求是否仍然有效 (crane是否还是IDLE)
                if self.cranes[crane_id_to_command].status == CraneStatus.IDLE:
                    self.crane_to_command = crane_id_to_command
                    observation = self._get_observation(crane_id_to_command)
                    info = self._get_info()
                    # 找到决策点，返回，暂停事件循环
                    return crane_id_to_command, observation, accumulated_reward, False, False, info

        # 如果事件队列为空，说明仿真结束
        print(f"[{self.current_time:.2f}s] Event queue is empty. Simulation finished.")
        final_obs = self._get_observation(0)
        return 0, final_obs, accumulated_reward, True, False, {"termination_reason": "event_queue_empty"}

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