import os
import sys
import time
import re
import importlib
import io
from contextlib import redirect_stdout

# 简单的解析函数（正则匹配标准输出中的指标）
def parse_metrics(output_text: str) -> dict:
    def find_float(pattern):
        m = re.search(pattern, output_text)
        return float(m.group(1)) if m else None
    def find_int(pattern):
        m = re.search(pattern, output_text)
        return int(m.group(1)) if m else None
    return {
        "total_task_wait_time_s": find_float(r"任务总等待时间:\s*([0-9]+(?:\.[0-9]+)?)s"),
        "crane_move_count": find_int(r"场桥移动次数:\s*([0-9]+)"),
        "total_simulation_time_s": find_float(r"总仿真时间:\s*([0-9]+(?:\.[0-9]+)?)s"),
        "completed_tasks_count": find_int(r"已完成任务数:\s*([0-9]+)"),
        "total_reward": find_float(r"总奖励:\s*([\-0-9]+(?:\.[0-9]+)?)"),
        "agent_loop_time_s": find_float(r"算法运行时间:\s*([0-9]+(?:\.[0-9]+)?)s"),
        "model_prep_time_s": find_float(r"模型准备时间:\s*([0-9]+(?:\.[0-9]+)?)s"),
    }

# 写入 Excel（若失败则回退到 CSV）
def write_excel(rows: list[dict], output_path: str):
    try:
        import openpyxl
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Results"
        headers = [
            "algorithm",
            "use_static",
            "total_task_wait_time_s",
            "crane_move_count",
            "algorithm_runtime_s",
            "model_prep_time_s",
            "total_simulation_time_s",
            "completed_tasks_count",
            "total_reward",
        ]
        ws.append(headers)
        for row in rows:
            ws.append([
                row.get("algorithm"),
                row.get("use_static"),
                row.get("total_task_wait_time_s"),
                row.get("crane_move_count"),
                row.get("algorithm_runtime_s"),
                row.get("model_prep_time_s"),
                row.get("total_simulation_time_s"),
                row.get("completed_tasks_count"),
                row.get("total_reward"),
            ])
        wb.save(output_path)
        print(f"Excel 已生成: {output_path}")
    except Exception as e:
        import csv
        csv_path = os.path.splitext(output_path)[0] + ".csv"
        headers = [
            "algorithm",
            "use_static",
            "total_task_wait_time_s",
            "crane_move_count",
            "algorithm_runtime_s",
            "model_prep_time_s",
            "total_simulation_time_s",
            "completed_tasks_count",
            "total_reward",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k) for k in headers})
        print(f"openpyxl 不可用或写入失败({e})，已生成 CSV: {csv_path}")

if __name__ == "__main__":
    # 解析命令行参数
    use_static = True
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_results.xlsx")
    for arg in sys.argv[1:]:
        if arg.startswith("--use_static="):
            val = arg.split("=", 1)[1].strip().lower()
            use_static = val in {"1", "true", "yes", "y"}
        elif arg.startswith("--output="):
            output_path = os.path.abspath(arg.split("=", 1)[1].strip())

    # 确保 tests 目录在导入路径中
    repo_root = os.path.dirname(os.path.abspath(__file__))
    tests_dir = os.path.join(repo_root, "tests")
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)

    modules = [
        ("随机策略", "test_random"),
        ("最近任务", "test_nearest"),
        ("最大等待时间", "test_biggest_wait_time"),
        ("贪心集束搜索", "test_greedy_search"),
        ("AStar", "test_astar"),
        ("模拟退火", "test_SA"),
        ("插入启发式", "test_insertion_heuristic"),
        ("PPO智能体", "test_agent"),
    ]

    results = []
    for alg_name, mod_name in modules:
        print(f"\n=== 运行 {alg_name}（模块 {mod_name}），use_static={use_static} ===")
        mod = importlib.import_module(mod_name)
        if not hasattr(mod, "run"):
            print(f"运行 {alg_name} 失败：模块不含 run() 方法")
            continue
        buf = io.StringIO()
        start = time.perf_counter()
        try:
            with redirect_stdout(buf):
                mod.run(use_static_data=use_static, enable_render=False, save_plot=False, output_metrics=True)
        except Exception as ex:
            print(f"运行 {alg_name} 失败：{ex}")
            continue
        end = time.perf_counter()

        output = buf.getvalue()
        metrics = parse_metrics(output)
        measured_total = round(end - start, 6)
        agent_loop = metrics.get("agent_loop_time_s")
        prep_time = metrics.get("model_prep_time_s")
        if agent_loop is not None:
            metrics["algorithm_runtime_s"] = agent_loop
        elif prep_time is not None:
            adjusted = measured_total - prep_time
            metrics["algorithm_runtime_s"] = round(adjusted if adjusted > 0 else 0.0, 6)
        else:
            metrics["algorithm_runtime_s"] = measured_total
        metrics["algorithm"] = alg_name
        metrics["use_static"] = use_static
        results.append(metrics)
        print(f"完成：{alg_name}，总等待时间={metrics.get('total_task_wait_time_s')}, 移动次数={metrics.get('crane_move_count')}, 运行时长={metrics.get('algorithm_runtime_s')}s")

    write_excel(results, output_path)