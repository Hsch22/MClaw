#!/usr/bin/env python
from __future__ import annotations

"""汇总 hidden-state 变体消融结果。"""

import argparse
import json
from pathlib import Path
from statistics import mean


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize hidden-state ablation outputs.")
    parser.add_argument("--root", required=True, help="Ablation output root.")
    parser.add_argument(
        "--tasks",
        default="textcraft,babyai,maze,weather",
        help="Comma-separated task names.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    root = Path(args.root)
    tasks = [item.strip() for item in str(args.tasks).split(",") if item.strip()]

    variants: dict[str, dict[str, object]] = {}
    for variant_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        metrics_path = variant_dir / "cross_task_metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        first_task = next((metrics.get(task) for task in tasks if task in metrics), None)
        if not first_task or "hidden_state" not in first_task:
            continue

        scores = []
        for task in tasks:
            item = metrics.get(task, {}).get("hidden_state")
            if item is None:
                continue
            scores.append(float(item["mean_max_cluster"]))

        command_file = variant_dir / tasks[0] / "audit.command.txt"
        overrides = ""
        if command_file.exists():
            for line in command_file.read_text(encoding="utf-8").splitlines():
                if line.startswith("extra_overrides="):
                    overrides = line.split("=", 1)[1]
                    break

        variants[variant_dir.name] = {
            "overrides": overrides,
            "avg_mean_max_cluster": mean(scores) if scores else 0.0,
            "per_task": {
                task: metrics.get(task, {}).get("hidden_state")
                for task in tasks
                if metrics.get(task, {}).get("hidden_state") is not None
            },
        }

    (root / "variant_metrics.json").write_text(
        json.dumps(variants, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines: list[str] = []
    lines.append("# Hidden-State Ablation Summary")
    lines.append("")
    lines.append(f"- output_root: `{root}`")
    lines.append("")
    lines.append("| variant | avg_mean_max_cluster | textcraft | babyai | maze | weather | overrides |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- |")

    def _task_metric(variant: dict[str, object], task: str) -> str:
        per_task = variant.get("per_task", {})
        if not isinstance(per_task, dict) or task not in per_task:
            return "-"
        item = per_task[task]
        if not isinstance(item, dict):
            return "-"
        return f"{float(item['mean_max_cluster']):.2f}"

    sorted_variants = sorted(
        variants.items(),
        key=lambda item: float(item[1]["avg_mean_max_cluster"]),
    )
    for variant_name, variant in sorted_variants:
        overrides = str(variant.get("overrides", ""))
        lines.append(
            "| "
            + f"{variant_name} | "
            + f"{float(variant['avg_mean_max_cluster']):.2f} | "
            + f"{_task_metric(variant, 'textcraft')} | "
            + f"{_task_metric(variant, 'babyai')} | "
            + f"{_task_metric(variant, 'maze')} | "
            + f"{_task_metric(variant, 'weather')} | "
            + f"`{overrides}` |"
        )

    (root / "variant_summary.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(root / "variant_summary.md")
    print(root / "variant_metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
