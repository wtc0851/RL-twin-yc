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

        # --- 环境核心参数 (Hardcoded) ---
        self.num_bays = 50
        self.num_cranes = 2
        self.crane_speed = 2.5
        self.mean_task_interval = 30.0
        self.mean_task_execution_time = 60.0
        self.max_simulation_time = 3600.0
        self.max_tasks_in_obs = 10
        
        # --- 奖励设计 (分阶段奖励) ---
        self.wait_penalty = -10.0                # 智能体选择“等待”动作的惩罚
        self.collision_penalty = -500.0          # 智能体做出无效动作（如导致碰撞）的惩罚
        self.task_acceptance_reward = 50.0       # 奖励1: 接受任务
        self.arrival_reward = 200.0              # 奖励2: 到达位置
        self.task_completion_reward = 750.0      # 奖励3: 完成任务 (总奖励 50+200+750=1000)

        # --- 其他 ---
        self.crane_initial_positions = [1, self.num_bays]

        # Observation and action spaces
        self.action_space = spaces.Discrete(self.max_tasks_in_obs + 1)
        self.observation_space = spaces.Dict({
            "crane_status": spaces.Box(low=0, high=np.inf, shape=(self.num_cranes, 2), dtype=np.float32),
            "task_list": spaces.Box(low=0, high=np.inf, shape=(self.max_tasks_in_obs, 3), dtype=np.float32),
            "action_mask": spaces.Box(low=0, high=1, shape=(self.action_space.n,), dtype=np.int8),
            "crane_to_command": spaces.Box(low=0, high=1, shape=(self.num_cranes,), dtype=np.int8),
        })

        # --- 内部状态 (在 reset() 中初始化) ---
        self.current_time = 0.0
        self.crane_to_command: Optional[int] = None
        self.cranes: List[Crane] = []
        self.task_queue: List[Task] = []
        self.completed_tasks: List[Task] = []
        self.event_queue: List[Event] = []
        self.history: Dict[int, List[Tuple[float, int]]] = {}

        self._crane_status_map = {status: i for i, status in enumerate(CraneStatus)}
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _generate_new_task(self):
        """生成一个新任务并将其放入任务队列，同时安排下一次任务生成事件。"""
        # 1. 计算下一个任务的到达时间
        next_task_arrival_time = self.current_time + random.expovariate(
            1.0 / self.mean_task_interval
        )

        # 2. 如果在模拟时间内，则安排下一次生成事件
        if next_task_arrival_time < self.max_simulation_time:
            heapq.heappush(
                self.event_queue,
                Event(time=next_task_arrival_time, type=EventType.TASK_GENERATION, data={}),
            )

        # 3. 创建新任务
        task_id = len(self.completed_tasks) + len(self.task_queue)
        location = random.randint(0, self.num_bays - 1)
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
            f"[{self.current_time:.2f}s] New task {task_id} generated at location {location}."
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

    def _get_action_mask(self, crane_id: int) -> List[int]:
        mask = np.zeros(self.action_space.n, dtype=np.int8)
        if self.crane_to_command is None:
            return mask

        commanding_crane = self.cranes[self.crane_to_command]
        other_crane = self.cranes[1 - self.crane_to_command]

        sorted_tasks = sorted(self.task_queue, key=lambda t: t.available_time)[:self.max_tasks_in_obs]
        for i, task in enumerate(sorted_tasks):
            is_collision = False
            if commanding_crane.location < other_crane.location:
                if task.location > other_crane.location:
                    is_collision = True
            elif commanding_crane.location > other_crane.location:
                if task.location < other_crane.location:
                    is_collision = True
            
            is_available = task.available_time <= self.current_time

            if not is_collision and is_available:
                mask[i] = 1

        mask[-1] = 1
        return mask

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
            print(f"[{self.current_time:.2f}s] Crane {crane_id} chooses to wait.")
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
                f"[{self.current_time:.2f}s] Crane {crane_id} assigned to task {task_to_assign.id} at {task_to_assign.location}. ETA: {arrival_time:.2f}s. Reward: +{self.task_acceptance_reward:.2f}"
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
                print(f"[{self.current_time:.2f}s] Simulation time limit reached.")
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
                    f"[{self.current_time:.2f}s] Crane {crane_id} arrived at {crane.location} and starts working. Reward: +{self.arrival_reward:.2f}"
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
                    f"[{self.current_time:.2f}s] Crane {crane_id} completed task {completed_task.id}. Wait time: {task_wait_time:.2f}s. Reward: {reward:.2f}"
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
        print(f"Trajectory plot saved to {save_path}")
        plt.close(fig)


    def close(self):
        pass