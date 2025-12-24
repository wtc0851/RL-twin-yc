"""
奖惩机制说明 (Reward Mechanism):
本环境采用 **稠密奖励 (Dense Reward)** 机制，即 **逐步惩罚 (Step-based Penalty)**。
- 每一步 (step) 都会计算惩罚值。
- 单步惩罚 = (当前已到达但未完成的任务数量) * (距离上一步的时间差) * (-1.0)。
- 最终奖励 = 单步惩罚 / 缩放因子。
- 异常结束时会有额外的大额惩罚 (-1000)。
- 目标：通过在每一步尽量减少积压任务数量，从而最小化总的加权等待时间。
"""
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
    CRANE_ARRIVAL = 2  # 场桥到达目标贝位事件
    TASK_COMPLETION = 3  # 场桥完成任务事件
    DECISION_REQUEST = 4  # 请求代理做出决策的事件


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

    def __init__(self, render_mode=None, static_tasks=None, **kwargs):
        super().__init__()

        # --- 环境核心参数---
        self.num_bays = 50  # 贝位数量
        self.num_cranes = 2  # 场桥数量
        self.crane_speed = 2  # 场桥移动速度 (单位: 贝位/秒)
        self.safe_distance = 2  # 场桥安全距离 (单位: 贝位)


        self.task_interval = 180.0  # 任务生成窗口大小 (单位: 秒)
        self.task_num_per_window = 9  # 每个窗口内平均生成的任务数量 (泊松分布的lambda)
        self.mean_task_arrieve_internel = self.task_interval / self.task_num_per_window  # 平均任务到达间隔 (单位: 秒)


        self.mean_task_execution_time = 45.0  # 任务耗时 (单位: 秒)
        self.max_simulation_time = 3600.0  # 最大模拟时间 (单位: 秒)  1小时


        self.max_tasks_in_obs = 30  # 代理能观察到的最大任务数量
        self.crane_initial_positions = [1, self.num_bays]  # 场桥初始位置


        # --- 奖励设计 ---
        self.reward_scale_factor = 10.0  # 奖励归一化缩放因子，用于控制奖励数值范围
        self.step_penalty_factor = -1.0  # 单步惩罚系数
        self.previous_time = 0.0  # 记录上一步的时间，用于计算时间差

        # --- 动作空间和状态空间 ---
        self.action_space = spaces.Discrete(self.max_tasks_in_obs + 1)
        self.observation_space = spaces.Dict({
            "crane_status": spaces.Box(low=0, high=np.inf, shape=(self.num_cranes, 2), dtype=np.float32),
            "task_list": spaces.Box(low=0, high=np.inf, shape=(self.max_tasks_in_obs, 3), dtype=np.float32),
            "action_mask": spaces.Box(low=0, high=1, shape=(self.action_space.n,), dtype=np.int8),
            "crane_to_command": spaces.Box(low=0, high=1, shape=(self.num_cranes,), dtype=np.int8),
        })



        # --- 内部状态，初始化reset() ---
        self.current_time = 0.0  # 当前模拟时间
        self.crane_to_command: Optional[int] = None  # 当前要命令的场桥ID
        self.cranes: List[Crane] = []  # 所有场桥
        self.task_queue: List[Task] = []  # 任务队列
        self.completed_tasks: List[Task] = []  # 已完成任务列表
        self.event_queue: List[Event] = []  # 事件队列
        self.history: Dict[int, List[Tuple[float, int]]] = {}  # 场桥轨迹记录 (id -> [(time, pos)])
        self.task_generation_stopped = False
        self.static_task_index = 0  # 重置静态任务索引
        # 新增：缓存最近一次观测下的可见任务快照，确保掩码与动作索引一致
        self._last_visible_tasks: List[Task] = []
        # 新增：步数计数器
        self.episode_steps = 0
        # 新增：步数上限，防止策略长时间运行
        self.max_episode_steps = 5000

        # --- 性能指标收集 ---
        self.total_task_wait_time = 0.0             # 任务总等待时间（available_time 至开工时间的等待）
        self.crane_move_count = 0                   # 场桥移动次数
        self.total_crane_move_time = 0.0            # 场桥移动总时间
        self.total_crane_wait_time = 0.0            # 场桥等待任务总时间（仅统计提前到位等待）
        self.total_crane_idle_time = 0.0            # 场桥闲置总时间（等待动作推进的时间）



        # 场桥状态映射字典
        self._crane_status_map = {status: i for i, status in enumerate(CraneStatus)}
        # 渲染模式
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        # 静态任务支持
        self.use_static_tasks = False
        self.static_tasks: List[Task] = []
        if static_tasks is not None:
            try:
                self.static_tasks = sorted(list(static_tasks), key=lambda t: t.available_time)
                self.use_static_tasks = True
            except Exception:
                self.static_tasks = []
                self.use_static_tasks = False
        # 等待时间步长（当没有更近的事件/任务时用于推进时间），单位：秒
        self.wait_time_step = 180.0

    def _generate_new_tasks(self):
        """在当前窗口生成一批任务，并安排下一窗口事件。"""
        window_start = self.current_time
        if getattr(self, "use_static_tasks", False):
            window_end = window_start + self.task_interval
            while self.static_task_index < len(self.static_tasks):
                t = self.static_tasks[self.static_task_index]
                if t.available_time >= window_end:
                    break
                if window_start <= t.available_time < window_end:
                    self.task_queue.append(t)
                    self.static_task_index += 1
                else:
                    if t.available_time < window_start:
                        self.static_task_index += 1
                    else:
                        break
            next_window_time = window_start + self.task_interval
            if next_window_time <= self.max_simulation_time and not self.task_generation_stopped:
                heapq.heappush(
                    self.event_queue,
                    Event(time=next_window_time, type=EventType.TASK_GENERATION, data={}),
                )
            else:
                self.task_generation_stopped = True
        else:
            # 改为泊松分布生成任务
            num_tasks_in_window = np.random.poisson(self.task_num_per_window)
            offsets = sorted(random.uniform(0.0, self.task_interval) for _ in range(num_tasks_in_window))
            for off in offsets:
                arrival_time = window_start + off
                if arrival_time >= self.max_simulation_time:
                    self.task_generation_stopped = True
                    break
                task_id = len(self.completed_tasks) + len(self.task_queue)
                location = random.randint(1, self.num_bays)
                # 修改为固定耗时
                execution_time = self.mean_task_execution_time
                new_task = Task(
                    id=task_id,
                    init_time=window_start,
                    location=location,
                    available_time=arrival_time,
                    execution_time=execution_time,
                )
                self.task_queue.append(new_task)
            next_window_time = window_start + self.task_interval
            if next_window_time <= self.max_simulation_time and not self.task_generation_stopped:
                heapq.heappush(
                    self.event_queue,
                    Event(time=next_window_time, type=EventType.TASK_GENERATION, data={}),
                )
            else:
                self.task_generation_stopped = True


    def _is_task_feasible_for_crane(self, crane_id: int, task: Task) -> bool:
        """基于当前碰撞规则判断某个任务对指定场桥是否可执行。"""
        commanding_crane = self.cranes[crane_id]
        other_crane = self.cranes[1 - crane_id]
        task_loc = task.location

        # 另一台场桥的移动路径区间（静止则为一个点）
        if other_crane.status == CraneStatus.MOVING and other_crane.current_task:
            other_path_min = min(other_crane.location, other_crane.current_task.location)
            other_path_max = max(other_crane.location, other_crane.current_task.location)
        else:
            other_path_min = other_crane.location
            other_path_max = other_crane.location

        # 当前场桥执行该任务的移动路径区间
        cmd_path_min = min(commanding_crane.location, task_loc)
        cmd_path_max = max(commanding_crane.location, task_loc)

        # 安全规则：0号的最右位置 + 安全距离 <= 1号的最左位置
        if crane_id == 0:
            return cmd_path_max + self.safe_distance <= other_path_min
        else:
            return other_path_max + self.safe_distance <= cmd_path_min


    def _get_visible_tasks(self) -> List[Task]:
        """
        返回当前可见任务集合。
        优先包含对当前待指令场桥可执行的任务，避免仅显示不可执行任务导致持续等待。
        """
        # 先按 init_time 排序所有任务
        sorted_tasks = sorted(self.task_queue, key=lambda t: t.init_time)
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
        # 缓存本次可见任务快照，掩码与动作索引以此为准
        self._last_visible_tasks = list(visible_tasks)
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
        # 当没有待指令的场桥时，至少允许“闲置”动作
        if self.crane_to_command is None:
            mask[-1] = 1
            return mask

        # 与观测保持完全一致：使用最近一次观测的可见任务快照（无快照则回退到当前可见）
        visible_tasks = self._last_visible_tasks if self._last_visible_tasks else self._get_visible_tasks()
        for i, task in enumerate(visible_tasks[:self.max_tasks_in_obs]):
            if self._is_task_feasible_for_crane(crane_id, task):
                mask[i] = 1

        # 若存在可执行任务则关闭“闲置”，否则只允许“闲置”
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
        return {
            "simulation_time": self.current_time,
            "step_count": self.episode_steps,
            "pending_tasks_count": len(self.task_queue),
            "completed_tasks_count": len(self.completed_tasks),
            "crane_to_command": self.crane_to_command,
            "total_task_wait_time": self.total_task_wait_time,
            "crane_move_count": self.crane_move_count,
            "total_crane_move_time": self.total_crane_move_time,
            "total_crane_wait_time": self.total_crane_wait_time,
            "total_crane_idle_time": self.total_crane_idle_time,
        }

    def _calculate_step_penalty(self) -> float:
        """
        计算单步惩罚：(剩余已到达未完成任务数量) * (时间差) * 惩罚系数
        """
        # 计算时间差
        time_diff = self.current_time - self.previous_time
        
        # 统计已到达但未完成的任务数量
        overdue_tasks_count = 0
        for task in self.task_queue:
            if task.available_time <= self.current_time:
                overdue_tasks_count += 1
        
        # 计算惩罚值
        penalty = overdue_tasks_count * time_diff * self.step_penalty_factor
        return penalty


    def _apply_action(self, action: int) -> float:
        """根据代理的动作，更新环境状态并返回即时奖励。"""


        crane_id = self.crane_to_command
        crane = self.cranes[crane_id]
        reward = 0.0

        # 动作是“闲置”
        wait_action_index = self.max_tasks_in_obs
        if action == wait_action_index:
            # 记录当前状态，以便在图表中显示
            self.history[crane.id].append((self.current_time, crane.location))
            crane.status = CraneStatus.IDLE
            # 闲置动作不再主动推送决策事件，避免同刻事件堆积

            next_task_init_time = 0
            for event in self.event_queue:
                if event.type == EventType.TASK_GENERATION:
                    next_task_init_time = event.time
                    break
            if next_task_init_time != 0:

                step_duration = next_task_init_time - self.current_time
                self.total_crane_idle_time += step_duration
                print(f"next_task_init_time: {next_task_init_time}, step_duration: {step_duration}, current_time: {self.current_time}")


            # 这里要统计闲置的时间
        else:
            # 使用最近一次观测的可见任务快照，避免索引错位
            visible_tasks = self._last_visible_tasks if self._last_visible_tasks else self._get_visible_tasks()
            # 索引越界兜底：视为“等待”
            if action >= len(visible_tasks):
                self.history[crane.id].append((self.current_time, crane.location))
                crane.status = CraneStatus.IDLE
                self.crane_to_command = None
                return reward

            selected_task = visible_tasks[action]
            # 执行前二次校验：若已不可行（例如另一台场桥状态变更），则兜底为“等待”
            if not self._is_task_feasible_for_crane(crane_id, selected_task):
                self.history[crane.id].append((self.current_time, crane.location))
                crane.status = CraneStatus.IDLE
                self.crane_to_command = None
                return reward

            # 通过唯一 id 定位在 self.task_queue 中的索引
            idx_in_queue = next((i for i, t in enumerate(self.task_queue) if t.id == selected_task.id), None)
            # 若任务已被其他事件移除，兜底为“等待”
            if idx_in_queue is None:
                self.history[crane.id].append((self.current_time, crane.location))
                crane.status = CraneStatus.IDLE
                self.crane_to_command = None
                return reward

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
        # 重置静态索引
        self.static_task_index = 0
        # 重置快照
        self._last_visible_tasks = []

        # 重置性能指标
        self.total_task_wait_time = 0.0
        self.crane_move_count = 0
        self.total_crane_move_time = 0.0
        self.total_crane_wait_time = 0.0
        self.total_crane_idle_time = 0.0

        # 重置步数
        self.episode_steps = 0
        
        # 重置时间相关变量
        self.current_time = 0.0
        self.previous_time = 0.0

        self.cranes = [
            Crane(id=0, location=self.crane_initial_positions[0], status=CraneStatus.IDLE),
            Crane(id=1, location=self.crane_initial_positions[1], status=CraneStatus.IDLE)
        ]

        self.task_queue: List[Task] = []
        self.completed_tasks: List[Task] = []

        self.history = {i: [(0, self.cranes[i].location)] for i in range(self.num_cranes)}

        self.event_queue: List[Event] = []
        # 首个任务窗口在 t=0 触发
        heapq.heappush(self.event_queue, Event(time=0.0, type=EventType.TASK_GENERATION, data={}))

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
        # 增加步数（统计用）
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

        # 3. 计算单步惩罚（基于时间差和已到达未完成任务数）
        step_penalty = self._calculate_step_penalty()

        # 4. 更新上一步时间戳
        self.previous_time = self.current_time

        # 5. 组合奖励并进行缩放
        total_reward = (reward + event_reward + step_penalty) / self.reward_scale_factor

        # 舍弃“步数截断”逻辑，避免重复决策导致非预期截断
        truncated = False

        return (
            observation,
            total_reward,
            terminated,
            truncated,
            info,
        )

    def _advance_simulation(self) -> Tuple[int, Dict, float, bool, bool, Dict]:
        """事件循环，处理事件直到需要下一个决策或仿真结束。"""

        # 事件循环，直到需要下一个决策或队列为空
        while self.event_queue:
            # 1. 从事件队列中取出下一个最近的事件
            event = heapq.heappop(self.event_queue)

            # 更新当前时间
            self.current_time = event.time

            # 3. 达到最大模拟时长后：停止任务生成，但继续处理现有事件/任务直到清空积压
            if self.current_time >= self.max_simulation_time:
                self.task_generation_stopped = True
                # 不截断，继续按事件类型处理，允许完成已到达的任务

            # 4. 根据事件类型处理事件
            if event.type == EventType.CRANE_ARRIVAL:
                crane_id = event.data["crane_id"]
                crane = self.cranes[crane_id]
                target_loc = crane.current_task.location

                # 到达前最终守卫：检查与另一台场桥的最终位置是否冲突
                other_crane = self.cranes[1 - crane_id]
                other_crane_pos = other_crane.location

                # 如果另一台场桥也在移动，必须用它的目标位置作为检查基准
                if other_crane.status == CraneStatus.MOVING and other_crane.current_task:
                    other_crane_pos = other_crane.current_task.location

                # 根据场桥ID确定左右位置
                crane0_pos = target_loc if crane_id == 0 else other_crane_pos
                crane1_pos = other_crane_pos if crane_id == 0 else target_loc

                # 最终检查：0号场桥的最终位置必须在1号的左边（含安全距离）
                if crane0_pos + self.safe_distance > crane1_pos:
                    # 检测到冲突，取消本次到达
                    self.history[crane.id].append((self.current_time, crane.location))
                    crane.status = CraneStatus.IDLE
                    failed_task = crane.current_task
                    crane.current_task = None
                    self.task_queue.append(failed_task)
                    heapq.heappush(
                        self.event_queue,
                        Event(time=self.current_time, type=EventType.DECISION_REQUEST, data={"crane_id": crane_id}),
                    )
                    continue  # 跳过此事件，处理下一个

                # 到达目标位置（通过顺序守卫校验后）
                crane.location = target_loc
                self.history[crane.id].append((self.current_time, crane.location))

                # 若任务尚未可开始，提前等待至 available_time
                start_time = max(self.current_time, crane.current_task.available_time)
                early_wait = max(0.0, start_time - self.current_time)
                if early_wait > 0:
                    self.total_crane_wait_time += early_wait
                # 任务等待时间（从可开始到实际开工）
                task_wait = max(0.0, start_time - crane.current_task.available_time)
                self.total_task_wait_time += task_wait

                # 正式开始作业到完成
                crane.status = CraneStatus.BUSY
                completion_time = start_time + crane.current_task.execution_time
                crane.free_at = completion_time
                heapq.heappush(
                    self.event_queue,
                    Event(time=completion_time, type=EventType.TASK_COMPLETION, data={"crane_id": crane_id}),
                )
            elif event.type == EventType.TASK_COMPLETION:
                crane_id = event.data["crane_id"]
                crane = self.cranes[crane_id]
                completed_task = crane.current_task
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
                        # 仅当该空闲场桥存在可执行任务时才推送决策请求
                        has_executable = any(
                            self._is_task_feasible_for_crane(other_crane.id, t) for t in self.task_queue
                        )
                        if has_executable:
                            heapq.heappush(
                                self.event_queue,
                                Event(time=self.current_time, type=EventType.DECISION_REQUEST,
                                      data={"crane_id": other_crane.id})
                            )
            elif event.type == EventType.TASK_GENERATION:
                # 在当前窗口批量生成任务
                self._generate_new_tasks()
                # 为所有空闲场桥推送决策事件
                for crane in self.cranes:
                    if crane.status == CraneStatus.IDLE:
                        heapq.heappush(
                            self.event_queue,
                            Event(time=self.current_time+1, type=EventType.DECISION_REQUEST, data={"crane_id": crane.id})
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
                                info = self._get_info()
                                return other_id, observation, 0.0, False, False, info
                    # 默认返回当前场桥的决策
                    self.crane_to_command = crane_id_to_command
                    observation = self._get_observation(crane_id_to_command)
                    info = self._get_info()
                    return crane_id_to_command, observation, 0.0, False, False, info

        # 如果事件队列为空，检查是否所有任务都已完成
        all_cranes_idle = all(crane.status == CraneStatus.IDLE for crane in self.cranes)
        no_pending_tasks = len(self.task_queue) == 0

        final_obs = self._get_observation(0)
        final_info = self._get_info()

        # 若在达到时间上限后清空积压，给出更准确的终止原因
        if all_cranes_idle and no_pending_tasks and self.current_time >= self.max_simulation_time:
            final_info["termination_reason"] = "all_tasks_completed_after_time_limit"
            print('仿真正常结束')
            return 0, final_obs, 0.0, True, False, final_info
        else:
            final_info["termination_reason"] = "no_future_generation_event"
            # print(f"仿真异常结束，原因：{final_info['termination_reason']}")
            return 0, final_obs, -1000, True, False, final_info

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

    def set_static_tasks(self, tasks: List[Task]) -> None:
        """在运行时设置静态任务并启用静态模式。"""
        self.static_tasks = sorted(list(tasks), key=lambda t: t.available_time)
        self.static_task_index = 0
        self.use_static_tasks = True