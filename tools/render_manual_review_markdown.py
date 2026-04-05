#!/usr/bin/env python
from __future__ import annotations

"""将 root_cluster_audit.jsonl 渲染成更适合人工打分的 Markdown。"""

import argparse
from collections.abc import Mapping, Sequence
import html
import json
from pathlib import Path
import re
from typing import Any

PRIMARY_METHODS = ["hidden_state", "output_grad", "logprob"]
SECONDARY_METHODS = ["action", "logit_distribution"]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render manual-review markdown from root cluster audit JSONL.")
    parser.add_argument("--input", required=True, help="Path to root_cluster_audit.jsonl")
    parser.add_argument("--output", required=True, help="Output markdown path")
    parser.add_argument(
        "--methods",
        default="hidden_state,output_grad,logprob,action,logit_distribution",
        help="Comma-separated method order to render",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=12,
        help="Max number of items to render; <=0 means render all successful items",
    )
    parser.add_argument(
        "--top-clusters",
        type=int,
        default=4,
        help="How many largest clusters to show before collapsing the rest",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional title override",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    input_path = Path(args.input)
    output_path = Path(args.output)
    methods = [item.strip() for item in str(args.methods).split(",") if item.strip()]

    reports = load_reports(input_path)
    successful = [report for report in reports if not report.get("root_error")]
    successful.sort(key=lambda report: interestingness(report, methods), reverse=True)
    if int(args.max_items) > 0:
        successful = successful[: int(args.max_items)]

    title = args.title or f"Manual Review: {input_path.parent.name}"
    markdown = render_markdown(
        title=title,
        reports=successful,
        methods=methods,
        top_clusters=max(int(args.top_clusters), 1),
        source_path=input_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    print(f"[manual_review] wrote {output_path}")
    return 0


def load_reports(path: Path) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            value = json.loads(raw)
            if isinstance(value, Mapping):
                reports.append(dict(value))
    return reports


def interestingness(report: Mapping[str, Any], methods: Sequence[str]) -> float:
    method_results = [
        report.get("methods", {}).get(method)
        for method in methods
        if isinstance(report.get("methods", {}).get(method), Mapping)
        and "error" not in report.get("methods", {}).get(method, {})
    ]
    if not method_results:
        return -1.0

    largest_sizes = [max((int(cluster.get("size", 0)) for cluster in result.get("clusters", [])), default=0) for result in method_results]
    actual_clusters = [len(result.get("clusters", [])) for result in method_results]
    selected_sets = {
        tuple(int(index) for index in result.get("selected_indices", []))
        for result in method_results
    }
    representative_sets = {
        tuple(int(index) for index in result.get("representative_indices", []))
        for result in method_results
    }
    score = 0.0
    score += 5.0 * (max(largest_sizes) - min(largest_sizes))
    score += 3.0 * (max(actual_clusters) - min(actual_clusters))
    score += float(max(largest_sizes))
    score += 2.0 * max(len(selected_sets) - 1, 0)
    score += 1.0 * max(len(representative_sets) - 1, 0)
    return score


def render_markdown(
    *,
    title: str,
    reports: Sequence[Mapping[str, Any]],
    methods: Sequence[str],
    top_clusters: int,
    source_path: Path,
) -> str:
    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"- source: `{source_path}`")
    lines.append(f"- rendered_items: `{len(reports)}`")
    lines.append(f"- method_order: `{', '.join(methods)}`")
    lines.append("")
    lines.append("## 打分标准")
    lines.append("")
    lines.append("- `语义纯度(1-5)`: 同簇动作是否大体表达同一意图")
    lines.append("- `错误合并(0/1)`: 明显不同语义被并到了同一簇")
    lines.append("- `过度拆分(0/1)`: 同一语义被拆得过散")
    lines.append("- `代表动作合适(0/1)`: representative 是否真能代表该簇")
    lines.append("")

    for report in reports:
        lines.extend(render_report(report=report, methods=methods, top_clusters=top_clusters))

    return "\n".join(lines).rstrip() + "\n"


def render_report(
    *,
    report: Mapping[str, Any],
    methods: Sequence[str],
    top_clusters: int,
) -> list[str]:
    lines: list[str] = []
    item_index = report.get("item_index")
    item_id = report.get("item_id")
    score = interestingness(report, methods)
    lines.append(f"## State {item_index} | item_id `{item_id}` | interest `{score:.1f}`")
    lines.append("")
    lines.append(f"- n_candidates: `{report.get('n_candidates')}`")
    lines.append(f"- requested_root_budget: `{report.get('requested_root_budget')}`")
    lines.append(f"- requested_root_clusters: `{report.get('requested_root_clusters')}`")
    lines.append(f"- requested_n_envs: `{report.get('requested_n_envs')}`")
    lines.append("")

    state_text = normalize_text(report.get("state_text", ""))
    if state_text:
        lines.append("**State**")
        lines.append("")
        lines.append(f"> {truncate_for_quote(state_text, 220)}")
        lines.append("")
        lines.append("<details><summary>完整 state_text</summary>")
        lines.append("")
        lines.append("```text")
        lines.append(state_text)
        lines.append("```")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    method_results = report.get("methods", {})
    present_methods = [
        method
        for method in methods
        if isinstance(method_results.get(method), Mapping)
    ]
    lines.append("**Quick Compare**")
    lines.append("")
    lines.append("| method | actual_clusters | largest_cluster | selected | representatives |")
    lines.append("| --- | ---: | ---: | --- | --- |")
    for method in present_methods:
        result = method_results[method]
        if "error" in result:
            lines.append(f"| {method} | error | error | error | error |")
            continue
        clusters = list(result.get("clusters", []))
        largest = max((int(cluster.get("size", 0)) for cluster in clusters), default=0)
        lines.append(
            "| "
            + " | ".join(
                [
                    method,
                    str(len(clusters)),
                    str(largest),
                    format_index_list(result.get("selected_indices", [])),
                    format_index_list(result.get("representative_indices", [])),
                ]
            )
            + " |"
        )
    lines.append("")

    groups = build_method_groups(present_methods)
    for label, group in groups:
        if not group:
            continue
        lines.append(f"### {label}")
        lines.append("")
        lines.extend(render_method_group(report=report, methods=group, top_clusters=top_clusters))
        lines.append("")

    lines.append("### Method Score")
    lines.append("")
    lines.append("| method | 语义纯度(1-5) | 错误合并(0/1) | 过度拆分(0/1) | 代表动作合适(0/1) | 备注 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
    for method in present_methods:
        lines.append(f"| {method} |  |  |  |  |  |")
    lines.append("")
    return lines


def build_method_groups(present_methods: Sequence[str]) -> list[tuple[str, list[str]]]:
    primary = [method for method in PRIMARY_METHODS if method in present_methods]
    secondary = [method for method in SECONDARY_METHODS if method in present_methods]
    remaining = [method for method in present_methods if method not in primary and method not in secondary]
    groups: list[tuple[str, list[str]]] = []
    if primary:
        groups.append(("Primary Methods", primary))
    if secondary or remaining:
        label = "Secondary Methods" if secondary else "Additional Methods"
        groups.append((label, secondary + remaining))
    return groups


def render_method_group(
    *,
    report: Mapping[str, Any],
    methods: Sequence[str],
    top_clusters: int,
) -> list[str]:
    method_results = report.get("methods", {})
    lines: list[str] = []
    lines.append("<table>")
    lines.append("<tr>")
    for method in methods:
        lines.append(f"<th>{html.escape(method)}</th>")
    lines.append("</tr>")
    lines.append("<tr>")
    for method in methods:
        result = method_results.get(method, {})
        lines.append('<td valign="top">')
        lines.extend(render_method_cell(result=result, top_clusters=top_clusters))
        lines.append("</td>")
    lines.append("</tr>")
    lines.append("</table>")
    return lines


def render_method_cell(*, result: Mapping[str, Any], top_clusters: int) -> list[str]:
    lines: list[str] = []
    if "error" in result:
        lines.append(f"<p><code>error</code>: {escape_inline(result.get('error', ''))}</p>")
        return lines

    clusters = sorted(
        (dict(cluster) for cluster in result.get("clusters", [])),
        key=lambda cluster: (-int(cluster.get("size", 0)), int(cluster.get("cluster_id", 0))),
    )
    largest = max((int(cluster.get("size", 0)) for cluster in clusters), default=0)
    lines.append("<p><strong>Overview</strong></p>")
    lines.append("<ul>")
    lines.append(f"<li>actual_clusters: <code>{len(clusters)}</code></li>")
    lines.append(f"<li>largest_cluster: <code>{largest}</code></li>")
    lines.append(f"<li>selected: <code>{escape_inline(format_index_list(result.get('selected_indices', [])))}</code></li>")
    lines.append("</ul>")

    visible = clusters[:top_clusters]
    hidden = clusters[top_clusters:]
    lines.append("<p><strong>Largest Clusters</strong></p>")
    for cluster in visible:
        lines.extend(render_cluster_card(cluster))
    if hidden:
        lines.append(f"<details><summary>其余 {len(hidden)} 个簇</summary>")
        for cluster in hidden:
            lines.extend(render_cluster_card(cluster))
        lines.append("</details>")
    return lines


def render_cluster_card(cluster: Mapping[str, Any]) -> list[str]:
    members = list(cluster.get("members", []))
    representative_index = int(cluster.get("representative_index", -1))
    selected_index = cluster.get("selected_index")
    selected_repr = f"#{int(selected_index)}" if selected_index is not None else "-"
    member_ids = " ".join(f"#{int(member.get('candidate_index', -1))}" for member in members)
    representative_text = next(
        (
            normalize_text(member.get("action_text", ""))
            for member in members
            if int(member.get("candidate_index", -1)) == representative_index
        ),
        "",
    )
    examples = build_examples(members, representative_index)

    lines: list[str] = []
    lines.append(
        "<p><strong>"
        f"C{int(cluster.get('cluster_id', -1))}"
        "</strong> | size <code>"
        f"{int(cluster.get('size', 0))}"
        "</code> | rep <code>"
        f"#{representative_index}"
        "</code> | sel <code>"
        f"{selected_repr}"
        "</code></p>"
    )
    lines.append(f"<p><strong>rep:</strong> {escape_inline(truncate_for_quote(representative_text, 120))}</p>")
    lines.append(f"<p><strong>members:</strong> <code>{escape_inline(member_ids)}</code></p>")
    if examples:
        lines.append("<p><strong>examples:</strong><br>")
        for example in examples:
            lines.append(example + "<br>")
        lines.append("</p>")
    return lines


def build_examples(members: Sequence[Mapping[str, Any]], representative_index: int) -> list[str]:
    examples: list[str] = []
    for member in members:
        candidate_index = int(member.get("candidate_index", -1))
        if candidate_index == representative_index:
            continue
        preview = truncate_for_quote(normalize_text(member.get("action_text", "")), 100)
        examples.append(f"<code>#{candidate_index}</code> {escape_inline(preview)}")
        if len(examples) >= 3:
            break
    if not examples and members:
        preview = truncate_for_quote(normalize_text(members[0].get("action_text", "")), 100)
        examples.append(f"<code>#{int(members[0].get('candidate_index', -1))}</code> {escape_inline(preview)}")
    return examples


def format_index_list(values: Sequence[Any]) -> str:
    return " ".join(f"#{int(value)}" for value in values)


def normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def truncate_for_quote(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)].rstrip() + "..."


def escape_inline(text: Any) -> str:
    return html.escape(str(text), quote=False)


if __name__ == "__main__":
    raise SystemExit(main())
