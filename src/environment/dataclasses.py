from enum import Enum
from dataclasses import dataclass, field
from typing import Optional


class CraneStatus(Enum):
    """
    龙门吊（场桥）的状态枚举
    """
    IDLE = "idle"  # 空闲，可以接受新任务
    MOVING = "moving"  # 正在移动到目标位置
    BUSY = "busy"  # 正在作业（吊起或放下集装箱）


@dataclass
class Task:
    """
    任务的数据结构
    """
    id: int  # 任务的唯一标识符
    location: int  # 任务需要操作的贝位（1-50）
    init_time: float  # 任务被创建（生成）的时间点（模拟分钟）
    available_time: float  # 任务可以开始处理的最早时间（模拟分钟）
    execution_time: float  # 任务执行需要花费的时间（模拟分钟）


@dataclass
class Crane:
    """
    龙门吊（场桥）的数据结构
    """
    id: int  # 场桥的唯一标识符 (例如 0 或 1)
    location: int  # 场桥当前所在的贝位
    status: CraneStatus  # 场桥当前的状态 (空闲, 移动中, 或作业中)
    current_task: Optional[Task] = None  # 场桥当前正在处理的任务
    free_at: float = 0.0  # 场桥预计完全空闲的时间点（完成移动和作业后）