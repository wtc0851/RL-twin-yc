"""
生成静态任务数据，任务到达过程遵循泊松过程。

任务字段：
- `init_time`：任务到达时间所在窗口的起始时间（用于排序/分析）。
- `available_time`：任务可开始的精确时间（遵循泊松过程）。
- `location`：整数贝位索引 [1, num_bays]。
- `execution_time`：作业时长，服从指数分布（均值 `mean_exec_time`）。

使用示例：
  # 在 Python 脚本中直接调用
  from tests.generate_static_tasks_env import generate_static_tasks
  tasks = generate_static_tasks(seed=1, out="my_tasks.json")
"""

from __future__ import annotations
import json
import random
import math
from pathlib import Path
from typing import List, Dict, Any, Optional


def generate_tasks_poisson(
    num_bays: int = 50,
    task_interval: float = 180.0,
    tasks_per_window: int = 5,
    mean_exec_time: float = 45.0,
    max_sim_time: float = 3600.0,
    arrival_rate: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    按泊松过程生成任务到达。

    - 若未指定到达率 `arrival_rate`，则根据窗口参数推导：
      arrival_rate = tasks_per_window / task_interval。
    - 为保持与 env 的排序逻辑一致，`init_time` 设为所在窗口起始时间。
    """
    tasks: List[Dict[str, Any]] = []
    task_id = 0
    current_time = 0.0
    rate = arrival_rate if arrival_rate is not None else (tasks_per_window / task_interval)

    while current_time < max_sim_time:
        # 采样下一次到达间隔（指数分布）
        delta = random.expovariate(rate)
        arrival_time = current_time + delta
        if arrival_time >= max_sim_time:
            break
        location = random.randint(1, num_bays)
        # 修改为固定耗时
        execution_time = mean_exec_time
        window_start = math.floor(arrival_time / task_interval) * task_interval
        tasks.append({
            "id": task_id,
            "init_time": window_start,
            "location": location,
            "available_time": arrival_time,
            "execution_time": execution_time,
        })
        task_id += 1
        current_time = arrival_time

    return tasks


def _round_task_fields(tasks: List[Dict[str, Any]], ndigits: int = 6) -> List[Dict[str, Any]]:
    rounded: List[Dict[str, Any]] = []
    for t in tasks:
        rounded.append({
            "id": int(t["id"]),
            "init_time": round(float(t["init_time"]), ndigits),
            "location": int(t["location"]),
            "available_time": round(float(t["available_time"]), ndigits),
            "execution_time": round(float(t["execution_time"]), ndigits),
        })
    return rounded


def write_json(tasks: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(_round_task_fields(tasks), f, ensure_ascii=False, indent=2)


def write_py(tasks: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rounded = _round_task_fields(tasks)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# 自动生成的静态任务，任务到达遵循泊松过程\n")
        f.write("# 字段: id, init_time, location, available_time, execution_time\n\n")
        f.write("STATIC_TASKS = ")
        json.dump(rounded, f, ensure_ascii=False, indent=2)
        f.write("\n")


def generate_static_tasks(
    seed: int = 42,
    num_bays: int = 50,
    task_interval: float = 180.0,
    tasks_per_window: int = 5,
    mean_exec_time: float = 45.0,
    max_sim_time: float = 3600.0,
    arrival_rate: Optional[float] = None,
    output_format: str = "json",
    out: Optional[str | Path] = None,
) -> List[Dict[str, Any]]:
    """
    以函数参数形式生成基于泊松过程的静态任务数据。

    - 若提供 `out` 路径，将按 `output_format` 写入文件（'json' 或 'py'）。
    - 始终返回任务列表（由 Python 字典组成）。
    """
    random.seed(seed)

    tasks = generate_tasks_poisson(
        num_bays=num_bays,
        task_interval=task_interval,
        tasks_per_window=tasks_per_window,
        mean_exec_time=mean_exec_time,
        max_sim_time=max_sim_time,
        arrival_rate=arrival_rate,
    )

    if out is not None:
        out_path = Path(out)
        if output_format == "json":
            write_json(tasks, out_path)
        elif output_format == "py":
            write_py(tasks, out_path)
        else:
            raise ValueError("输出格式必须为 'json' 或 'py'")
        print(f"已生成 {len(tasks)} 个任务 -> {out_path}")

    return tasks


def main() -> None:
    """默认调用：生成 JSON 文件到同目录。"""
    default_out = Path(__file__).with_name("static_tasks_env.json")
    generate_static_tasks(
        out=str(default_out),
        output_format="json",
        tasks_per_window=9,
        seed=4,
    )


if __name__ == "__main__":
    main()