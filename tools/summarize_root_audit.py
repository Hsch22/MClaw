#!/usr/bin/env python
from __future__ import annotations

"""汇总多 task root-audit 输出。"""

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize root-audit outputs across tasks.")
    parser.add_argument("--root", required=True, help="Root output directory containing per-task audit dirs.")
    parser.add_argument(
        "--tasks",
        default="textcraft,babyai,maze,weather",
        help="Comma-separated task directory names.",
    )
    parser.add_argument(
        "--methods",
        default="action,hidden_state,output_grad,logprob,logit_distribution",
        help="Comma-separated methods to summarize.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    root = Path(args.root)
    tasks = [item.strip() for item in str(args.tasks).split(",") if item.strip()]
    methods = [item.strip() for item in str(args.methods).split(",") if item.strip()]

    metrics: dict[str, dict[str, dict[str, float | int]]] = {}
    for task in tasks:
        jsonl_path = root / task / "root_cluster_audit.jsonl"
        reports = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        task_metrics: dict[str, dict[str, float | int]] = {}
        for method in methods:
            successful = [
                report["methods"][method]
                for report in reports
                if method in report.get("methods", {}) and "error" not in report["methods"][method]
            ]
            if not successful:
                continue
            actual_clusters = [len(result.get("clusters", [])) for result in successful]
            max_clusters = [
                max((int(cluster.get("size", 0)) for cluster in result.get("clusters", [])), default=0)
                for result in successful
            ]
            selected = [len(result.get("selected_indices", [])) for result in successful]
            mean_regret = [
                float(result.get("utility", {}).get("mean_representative_regret", 0.0))
                for result in successful
            ]
            task_metrics[method] = {
                "n_success": len(successful),
                "mean_actual_clusters": mean(actual_clusters),
                "mean_max_cluster": mean(max_clusters),
                "mean_selected": mean(selected),
                "mean_representative_regret": mean(mean_regret),
            }
        metrics[task] = task_metrics

    (root / "cross_task_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines: list[str] = []
    lines.append("# Cross-Task Summary")
    lines.append("")
    lines.append(f"- output_root: `{root}`")
    lines.append(f"- tasks: `{', '.join(tasks)}`")
    lines.append(f"- methods: `{', '.join(methods)}`")
    lines.append("")
    for task in tasks:
        lines.append(f"## {task}")
        lines.append("")
        lines.append("| method | n_success | mean_actual_clusters | mean_max_cluster | mean_selected | mean_regret |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for method in methods:
            item = metrics.get(task, {}).get(method)
            if item is None:
                continue
            lines.append(
                f"| {method} | {int(item['n_success'])} | "
                f"{float(item['mean_actual_clusters']):.2f} | "
                f"{float(item['mean_max_cluster']):.2f} | "
                f"{float(item['mean_selected']):.2f} | "
                f"{float(item['mean_representative_regret']):.6f} |"
            )
        lines.append("")

    (root / "cross_task_summary.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(root / "cross_task_summary.md")
    print(root / "cross_task_metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
