# -*- coding: utf-8 -*-
"""汇总 result/baselines 下的基线实验结果到 CSV。

运行方式（在项目根目录）：
    python summarize_baselines.py

输出文件：
    result/baselines/baselines_summary.csv

文件名约定：
    baselines_<graph_type>_<n_nodes>_k<sensor_budget>.json
    baselines_<graph_type>_<n_nodes>_k<sensor_budget>_delay.json

其中带 ``_delay`` 的文件表示使用了带连边延迟分布的距离矩阵；
不带 ``_delay`` 的文件表示只使用跳数距离。
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "result" / "baselines"
OUTPUT_FILE = INPUT_DIR / "baselines_summary.csv"

FILE_PATTERN = re.compile(
    # 图类型允许包含下划线，例如 IEEE_14Bus、WDN_Net1。
    # 通过末尾的“节点数_k预算”反向确定字段边界。
    r"^baselines_(?P<graph_type>.+)_(?P<n_nodes>\d+)_k(?P<sensor_budget>\d+)"
    r"(?P<delay>_delay)?\.json$"
)


def sample_std(values: list[float]) -> float:
    """计算样本标准差；只有一个样本时返回 0。"""
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return (sum((value - avg) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def as_float(value) -> float:
    return float(value) if value is not None else float("nan")


def build_rows() -> list[dict]:
    rows = []
    for result_file in sorted(INPUT_DIR.glob("baselines_*.json")):
        match = FILE_PATTERN.match(result_file.name)
        if match is None:
            print(f"跳过无法解析文件名的结果：{result_file.name}")
            continue

        info = match.groupdict()
        graph_type = info["graph_type"]
        n_nodes = int(info["n_nodes"])
        sensor_budget = int(info["sensor_budget"])
        has_delay_distribution = bool(info["delay"])

        with result_file.open("r", encoding="utf-8") as handle:
            result = json.load(handle)

        # 当前结果结构为 {"PPOAgent": {method: {ave_rank, ave_time, ...}}}。
        # 同时兼容直接以方法名为第一层键的结构。
        method_results = result.get("PPOAgent", result)
        if not isinstance(method_results, dict):
            print(f"跳过结果结构异常的文件：{result_file.name}")
            continue

        for method, metrics in sorted(method_results.items()):
            if not isinstance(metrics, dict):
                continue

            rank_values = [as_float(value) for value in metrics.get("rank", [])]
            time_values = [as_float(value) for value in metrics.get("time", [])]

            # 优先使用文件中记录的平均值；若不存在，则由明细重新计算。
            ave_rank = metrics.get("ave_rank")
            if ave_rank is None and rank_values:
                ave_rank = mean(rank_values)

            ave_time = metrics.get("ave_time")
            if ave_time is None and time_values:
                ave_time = mean(time_values)

            rows.append({
                "graph_type": graph_type,
                "n_nodes": n_nodes,
                "sensor_budget": sensor_budget,
                "delay_distribution": (
                    "with_delay_distribution"
                    if has_delay_distribution
                    else "no_delay_hop_only"
                ),
                "has_delay_distribution": has_delay_distribution,
                "method": method,
                "ave_rank": as_float(ave_rank),
                "ave_time_seconds": as_float(ave_time),
                "rank_std": sample_std(rank_values) if rank_values else float("nan"),
                "time_std_seconds": sample_std(time_values) if time_values else float("nan"),
                "num_graphs": len(rank_values),
                "source_file": result_file.name,
            })

    rows.sort(key=lambda row: (
        row["graph_type"],
        row["n_nodes"],
        row["has_delay_distribution"],
        row["method"],
    ))
    return rows


def write_csv(rows: list[dict]) -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "graph_type",
        "n_nodes",
        "sensor_budget",
        "delay_distribution",
        "has_delay_distribution",
        "method",
        "ave_rank",
        "ave_time_seconds",
        "rank_std",
        "time_std_seconds",
        "num_graphs",
        "source_file",
    ]

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"找不到结果目录：{INPUT_DIR}")

    summary_rows = build_rows()
    write_csv(summary_rows)
    print(f"已汇总 {len(summary_rows)} 条方法结果")
    print(f"CSV 文件：{OUTPUT_FILE}")
