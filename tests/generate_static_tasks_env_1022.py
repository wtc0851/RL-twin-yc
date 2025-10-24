#!/usr/bin/env python3
"""
生成与 src/environment/env_1022.py 一致的静态任务数据。

生成机制：
- “窗口”机制：固定长度 `task_interval` 的时间窗；每个窗口生成 `tasks_per_window` 个任务。
- “泊松”机制：按给定平均到达率（任务/秒）以泊松过程生成任务到达。

任务字段：
- `init_time`：窗口起始时间（用于排序/分析；泊松机制下为到达时间所在窗口起点）。
- `available_time`：任务可开始时间。
- `location`：整数贝位索引 [1, num_bays]。
- `execution_time`：作业时长，指数分布（均值 `mean_exec_time`）。

使用示例：
  python tests/generate_static_tasks_env_1022.py \
    --随机种子 1 --输出格式 json --输出文件 tests/static_tasks_env1022.json
  python tests/generate_static_tasks_env_1022.py \
    --随机种子 1 --输出格式 py --输出文件 tests/static_tasks_env1022.py
  python tests/generate_static_tasks_env_1022.py \
    --生成机制 泊松 --平均到达率 0.03 --输出格式 json
"""

from __future__ import annotations
import argparse
import json
import random
import math
from pathlib import Path
from typing import List, Dict, Any, Optional


def generate_tasks(
    num_bays: int = 50,
    task_interval: float = 180.0,
    tasks_per_window: int = 5,
    mean_exec_time: float = 60.0,
    max_sim_time: float = 3600.0,
) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    current_time = 0.0
    task_id = 0

    while current_time < max_sim_time:
        window_start = current_time
        # Generate offsets uniformly in [0, task_interval)
        offsets = sorted(random.uniform(0.0, task_interval) for _ in range(tasks_per_window))
        for off in offsets:
            available_time = window_start + off
            if available_time >= max_sim_time:
                break
            location = random.randint(1, num_bays)
            # Exp(mean = mean_exec_time) => rate = 1/mean
            execution_time = random.expovariate(1.0 / mean_exec_time)
            tasks.append({
                "id": task_id,
                "init_time": window_start,
                "location": location,
                "available_time": available_time,
                "execution_time": execution_time,
            })
            task_id += 1
        # Advance to next window
        current_time = window_start + task_interval

    return tasks


def generate_tasks_poisson(
    num_bays: int = 50,
    task_interval: float = 180.0,
    tasks_per_window: int = 5,
    mean_exec_time: float = 60.0,
    max_sim_time: float = 3600.0,
    arrival_rate: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """按泊松过程生成任务到达。
    - 若未指定到达率，则根据窗口参数推导：arrival_rate = tasks_per_window / task_interval。
    - 为保持与 env_1022 的排序逻辑一致，init_time 设为所在窗口起始时间。
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
        execution_time = random.expovariate(1.0 / mean_exec_time)
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
        f.write("# 自动生成的静态任务，匹配 env_1022 的生成器\n")
        f.write("# 字段: id, init_time, location, available_time, execution_time\n\n")
        f.write("STATIC_TASKS = ")
        json.dump(rounded, f, ensure_ascii=False, indent=2)
        f.write("\n")


def generate_static_tasks_env_1022(
    seed: int = 42,
    num_bays: int = 50,
    task_interval: float = 180.0,
    tasks_per_window: int = 5,
    mean_exec_time: float = 60.0,
    max_sim_time: float = 3600.0,
    mechanism: str = "窗口",
    arrival_rate: Optional[float] = None,
    output_format: str = "json",
    out: Optional[str | Path] = None,
) -> List[Dict[str, Any]]:
    """以函数参数形式生成静态任务数据。

    - 使用 `mechanism` 选择生成机制（"窗口"/"泊松"）。
    - 若提供 `out`，将按 `output_format` 写入文件（json/py）。
    - 始终返回任务列表（Python 字典组成）。
    """
    random.seed(seed)

    if mechanism in ("窗口", "window"):
        tasks = generate_tasks(
            num_bays=num_bays,
            task_interval=task_interval,
            tasks_per_window=tasks_per_window,
            mean_exec_time=mean_exec_time,
            max_sim_time=max_sim_time,
        )
    elif mechanism in ("泊松", "poisson"):
        tasks = generate_tasks_poisson(
            num_bays=num_bays,
            task_interval=task_interval,
            tasks_per_window=tasks_per_window,
            mean_exec_time=mean_exec_time,
            max_sim_time=max_sim_time,
            arrival_rate=arrival_rate,
        )
    else:
        raise ValueError(f"未知生成机制: {mechanism}")

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
    """默认调用：使用函数参数生成 JSON 文件到同目录。"""
    default_out = Path(__file__).with_name("static_tasks_env1022.json")
    generate_static_tasks_env_1022(out=str(default_out), output_format="json")


if __name__ == "__main__":
    main()