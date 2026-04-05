#!/usr/bin/env python
from __future__ import annotations

"""对首轮根结点 rollout 候选动作做多方法聚类审计。"""

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import random
from statistics import mean
from typing import Any

from mclaw.core import BranchSelector
from mclaw.critic import QCritic, QHead
from mclaw.trainer.main import (
    _build_actor_module,
    _build_clusterer,
    _build_env_client_factory,
    _build_inference_engine,
    _build_q_head,
    _build_rollout_handler_factory,
    _build_tokenizer,
    load_config,
)
from mclaw.trainer.mclaw_trainer import _load_prompt_items_from_file
from mclaw.utils import EmbeddingMatrixCache


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit root-level clustering on a fixed candidate pool.")
    parser.add_argument("--config", required=True, help="MClaw YAML config path.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write JSONL / Markdown reports.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of dataset items to audit.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Dataset offset before taking `limit` items.",
    )
    parser.add_argument(
        "--methods",
        default="action,hidden_state,logprob,output_grad,logit_distribution",
        help="Comma-separated clustering methods to compare on the same candidate pool.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional override for trainer.seed.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort immediately if one method fails on one sample.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Extra dotlist overrides, same format as mclaw.trainer.main.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    _ensure_local_no_proxy()
    config = load_config(args.config, args.overrides)
    if args.seed is not None:
        config.trainer.seed = int(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _set_seed(int(config.trainer.seed))

    tokenizer = None
    actor_module = None
    q_head = None
    q_critic = None
    inference_engine = None
    rollout = None
    embedding_matrix_cache = None

    try:
        tokenizer = _build_tokenizer(config)
        actor_module = _build_actor_module(config)
        q_head = _build_q_head(config, actor_module)
        q_critic = QCritic(
            actor_module_fsdp=actor_module,
            q_head=q_head,
            tokenizer=tokenizer,
            config=config.q_critic,
        )
        inference_engine = _build_inference_engine(config)
        env_client_factory = _build_env_client_factory(config)
        branch_selector = BranchSelector()
        rollout_handler_factory = _build_rollout_handler_factory(
            config=config,
            tokenizer=tokenizer,
        )
        embedding_matrix_cache = EmbeddingMatrixCache(actor_module)

        from mclaw.core import TreeRollout

        # 占位 clusterer 仅用于构造 rollout；真正的多方法聚类在固定 candidate pool 上逐个重放。
        rollout = TreeRollout(
            inference_engine=inference_engine,
            actor_module_fsdp=actor_module,
            q_critic=q_critic,
            clusterer=_build_clusterer(config),
            tokenizer=tokenizer,
            config=config.tree_rollout,
            env_client_factory=env_client_factory,
            branch_selector=branch_selector,
            handler_factory=rollout_handler_factory,
        )

        methods = [item.strip() for item in args.methods.split(",") if item.strip()]
        items = _load_prompt_items_from_file(Path(config.data.train_file))
        start = max(int(args.offset), 0)
        end = start + max(int(args.limit), 0)
        selected_items = items[start:end]

        if not selected_items:
            raise ValueError(
                "no prompt items selected; check --offset/--limit against data.train_file"
            )

        reports: list[dict[str, Any]] = []
        method_errors: dict[str, int] = defaultdict(int)

        for relative_index, prompt_item in enumerate(selected_items):
            absolute_index = start + relative_index
            normalized_item = _normalize_prompt_item(prompt_item, config.environment.reset_kwargs)
            try:
                report = build_root_report(
                    rollout=rollout,
                    prompt_item=normalized_item,
                    methods=methods,
                    config=config,
                    branch_selector=branch_selector,
                    embedding_matrix_cache=embedding_matrix_cache,
                    fail_fast=bool(args.fail_fast),
                    item_index=absolute_index,
                )
            except Exception as exc:  # pragma: no cover - runtime audit should keep going
                if args.fail_fast:
                    raise
                report = {
                    "item_index": absolute_index,
                    "item_id": _resolve_field(normalized_item, "item_id"),
                    "root_error": f"{type(exc).__name__}: {exc}",
                    "methods": {},
                }
            reports.append(report)

            for method_name, method_result in report["methods"].items():
                if "error" in method_result:
                    method_errors[method_name] += 1

        summary = build_summary(
            reports=reports,
            config_path=args.config,
            methods=methods,
            data_file=config.data.train_file,
            output_dir=output_dir,
            overrides=list(args.overrides),
            seed=int(config.trainer.seed),
        )

        jsonl_path = output_dir / "root_cluster_audit.jsonl"
        summary_path = output_dir / "summary.json"
        markdown_path = output_dir / "summary.md"

        with jsonl_path.open("w", encoding="utf-8") as handle:
            for report in reports:
                handle.write(json.dumps(report, ensure_ascii=False) + "\n")

        summary["method_errors"] = dict(method_errors)
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)

        markdown_path.write_text(render_markdown_summary(summary, reports), encoding="utf-8")

        print(f"[root_audit] wrote {jsonl_path}")
        print(f"[root_audit] wrote {summary_path}")
        print(f"[root_audit] wrote {markdown_path}")
        return 0
    finally:
        _cleanup_audit_resources(
            inference_engine=inference_engine,
            rollout=rollout,
            embedding_matrix_cache=embedding_matrix_cache,
            q_critic=q_critic,
            q_head=q_head,
            actor_module=actor_module,
            tokenizer=tokenizer,
        )


def _ensure_local_no_proxy() -> None:
    localhost_entries = ["127.0.0.1", "localhost"]
    for key in ("NO_PROXY", "no_proxy"):
        current = os.environ.get(key, "")
        parts = [item.strip() for item in current.split(",") if item.strip()]
        updated = list(parts)
        for entry in localhost_entries:
            if entry not in updated:
                updated.append(entry)
        os.environ[key] = ",".join(updated)


def build_root_report(
    *,
    rollout: Any,
    prompt_item: Any,
    methods: Sequence[str],
    config: Any,
    branch_selector: BranchSelector,
    embedding_matrix_cache: EmbeddingMatrixCache,
    fail_fast: bool,
    item_index: int,
) -> dict[str, Any]:
    env_pool = rollout._initialize_env_pool(prompt_item)
    try:
        root = rollout._build_root_node(prompt_item)
        if not root.state_tokens and env_pool:
            obs = env_pool[0].observe()
            if obs:
                obs_text = str(obs)
                root.state_text = obs_text
                root.state_tokens = list(rollout.tokenizer.encode(obs_text, add_special_tokens=False))
        root.metadata["rollout_handler"] = rollout._build_root_handler(prompt_item, root)

        candidates = list(rollout._expand_root_candidates(root))
        model_outputs = dict(root.metadata.get("candidate_model_outputs", {}))
        base_candidates = [_serialize_candidate(node, index=index) for index, node in enumerate(candidates)]

        report: dict[str, Any] = {
            "item_index": item_index,
            "item_id": _resolve_field(prompt_item, "item_id"),
            "root_node_id": root.node_id,
            "state_text": root.state_text,
            "state_token_count": len(root.state_tokens),
            "n_candidates": len(candidates),
            "requested_root_budget": int(config.tree_rollout.root_budget),
            "requested_root_clusters": int(config.tree_rollout.root_clusters),
            "requested_n_envs": int(config.tree_rollout.n_envs),
            "candidates": base_candidates,
            "methods": {},
        }

        if not candidates:
            return report

        for method in methods:
            try:
                report["methods"][method] = run_single_method_audit(
                    method=method,
                    config=config,
                    candidates=candidates,
                    model_outputs=model_outputs,
                    branch_selector=branch_selector,
                    embedding_matrix_cache=embedding_matrix_cache,
                )
            except Exception as exc:  # pragma: no cover - runtime audit should keep going
                if fail_fast:
                    raise
                report["methods"][method] = {
                    "error": f"{type(exc).__name__}: {exc}",
                }

        return report
    finally:
        for env_client in env_pool:
            close = getattr(env_client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass


def run_single_method_audit(
    *,
    method: str,
    config: Any,
    candidates: Sequence[Any],
    model_outputs: Mapping[str, Any],
    branch_selector: BranchSelector,
    embedding_matrix_cache: EmbeddingMatrixCache,
) -> dict[str, Any]:
    method_config = deepcopy(config)
    method_config.clustering.method = str(method)
    clusterer = _build_clusterer(method_config)

    method_model_outputs = dict(model_outputs)
    if method == "output_grad" and "embedding_matrix" not in method_model_outputs:
        method_model_outputs["embedding_matrix"] = embedding_matrix_cache.get_embedding_matrix()

    cluster_result = clusterer.cluster_candidates(
        nodes=candidates,
        n_clusters=int(method_config.tree_rollout.root_clusters),
        model_outputs=method_model_outputs,
    )

    n_select = min(
        int(method_config.tree_rollout.n_envs),
        len(candidates),
        len(cluster_result.representative_indices),
    )
    selected_indices = branch_selector.select_root_representatives(
        candidates=candidates,
        representative_indices=cluster_result.representative_indices,
        cluster_labels=cluster_result.labels,
        n_select=n_select,
    )

    clusters = build_cluster_cards(
        candidates=candidates,
        labels=cluster_result.labels,
        representative_indices=cluster_result.representative_indices,
        selected_indices=selected_indices,
    )
    utility = compute_cluster_utility(
        candidates=candidates,
        clusters=clusters,
    )

    return {
        "labels": list(cluster_result.labels),
        "representative_indices": list(cluster_result.representative_indices),
        "selected_indices": list(selected_indices),
        "cluster_metadata": _to_python(cluster_result.metadata),
        "clusters": clusters,
        "utility": utility,
    }


def build_cluster_cards(
    *,
    candidates: Sequence[Any],
    labels: Sequence[int],
    representative_indices: Sequence[int],
    selected_indices: Sequence[int],
) -> list[dict[str, Any]]:
    groups: dict[int, list[int]] = defaultdict(list)
    for index, cluster_id in enumerate(labels):
        groups[int(cluster_id)].append(index)

    representative_set = set(int(index) for index in representative_indices)
    selected_set = set(int(index) for index in selected_indices)

    cards: list[dict[str, Any]] = []
    for cluster_id in sorted(groups):
        member_indices = groups[cluster_id]
        representative_index = next(
            (index for index in member_indices if index in representative_set),
            member_indices[0],
        )
        selected_index = next(
            (index for index in member_indices if index in selected_set),
            None,
        )
        members = [
            {
                "candidate_index": index,
                "action_text": getattr(candidates[index], "action_text", ""),
                "action_tokens": list(getattr(candidates[index], "action_tokens", [])),
                "log_prob": _safe_float(getattr(candidates[index], "log_prob", None)),
                "q_value": _safe_float(getattr(candidates[index], "q_value", None)),
                "is_representative": index == representative_index,
                "is_selected": index in selected_set,
            }
            for index in member_indices
        ]
        cards.append(
            {
                "cluster_id": int(cluster_id),
                "size": len(member_indices),
                "representative_index": int(representative_index),
                "selected_index": int(selected_index) if selected_index is not None else None,
                "members": members,
            }
        )
    return cards


def compute_cluster_utility(
    *,
    candidates: Sequence[Any],
    clusters: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    regrets: list[float] = []
    best_cluster_regret: float | None = None
    global_best_index: int | None = None
    global_best_q = _max_q(candidates)

    if global_best_q is not None:
        global_best_index = _argmax_q(candidates)

    for cluster in clusters:
        members = cluster.get("members", [])
        representative_index = int(cluster["representative_index"])
        cluster_qs = [member["q_value"] for member in members if member.get("q_value") is not None]
        representative_q = _safe_float(getattr(candidates[representative_index], "q_value", None))
        if cluster_qs and representative_q is not None:
            regret = max(cluster_qs) - representative_q
            regrets.append(float(regret))
            if global_best_index is not None and any(
                int(member["candidate_index"]) == global_best_index for member in members
            ):
                best_cluster_regret = float(regret)

    return {
        "mean_representative_regret": mean(regrets) if regrets else 0.0,
        "max_representative_regret": max(regrets) if regrets else 0.0,
        "global_best_q": global_best_q,
        "global_best_index": global_best_index,
        "global_best_cluster_regret": best_cluster_regret,
    }


def build_summary(
    *,
    reports: Sequence[Mapping[str, Any]],
    config_path: str,
    methods: Sequence[str],
    data_file: str,
    output_dir: Path,
    overrides: Sequence[str],
    seed: int,
) -> dict[str, Any]:
    root_errors = [
        {
            "item_index": report.get("item_index"),
            "item_id": report.get("item_id"),
            "error": report.get("root_error"),
        }
        for report in reports
        if report.get("root_error")
    ]
    method_summary: dict[str, dict[str, Any]] = {}
    for method in methods:
        successful = [
            report["methods"][method]
            for report in reports
            if method in report.get("methods", {}) and "error" not in report["methods"][method]
        ]
        if not successful:
            method_summary[method] = {"n_success": 0}
            continue

        actual_clusters = [
            int(result["cluster_metadata"].get("n_clusters", 0))
            for result in successful
        ]
        mean_regrets = [
            float(result["utility"]["mean_representative_regret"])
            for result in successful
        ]
        max_regrets = [
            float(result["utility"]["max_representative_regret"])
            for result in successful
        ]
        method_summary[method] = {
            "n_success": len(successful),
            "mean_actual_clusters": mean(actual_clusters),
            "mean_representative_regret": mean(mean_regrets),
            "worst_representative_regret": max(max_regrets),
        }

    return {
        "config_path": config_path,
        "data_file": data_file,
        "output_dir": str(output_dir),
        "seed": seed,
        "n_items": len(reports),
        "n_root_errors": len(root_errors),
        "root_errors": root_errors,
        "methods": list(methods),
        "overrides": list(overrides),
        "method_summary": method_summary,
    }


def render_markdown_summary(summary: Mapping[str, Any], reports: Sequence[Mapping[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# Root Cluster Audit")
    lines.append("")
    lines.append(f"- data_file: `{summary['data_file']}`")
    lines.append(f"- n_items: `{summary['n_items']}`")
    lines.append(f"- n_root_errors: `{summary.get('n_root_errors', 0)}`")
    lines.append(f"- methods: `{', '.join(summary['methods'])}`")
    lines.append(f"- seed: `{summary['seed']}`")
    if summary.get("overrides"):
        lines.append(f"- overrides: `{ ' '.join(summary['overrides']) }`")
    lines.append("")
    if summary.get("root_errors"):
        lines.append("## Root Errors")
        lines.append("")
        for item in summary["root_errors"]:
            lines.append(
                f"- item `{item.get('item_index')}` / item_id `{item.get('item_id')}`: "
                f"`{item.get('error')}`"
            )
        lines.append("")

    lines.append("## Method Summary")
    lines.append("")
    for method, result in summary["method_summary"].items():
        lines.append(f"### `{method}`")
        if result.get("n_success", 0) == 0:
            lines.append("- failed on all audited items")
            lines.append("")
            continue
        lines.append(f"- n_success: `{result['n_success']}`")
        lines.append(f"- mean_actual_clusters: `{result['mean_actual_clusters']:.3f}`")
        lines.append(f"- mean_representative_regret: `{result['mean_representative_regret']:.6f}`")
        lines.append(f"- worst_representative_regret: `{result['worst_representative_regret']:.6f}`")
        lines.append("")

    lines.append("## Per-Item Cards")
    lines.append("")
    for report in reports:
        lines.append(f"### item `{report['item_index']}` / item_id `{report['item_id']}`")
        lines.append("")
        if report.get("root_error"):
            lines.append(f"- root_error: `{report['root_error']}`")
            lines.append("")
            continue
        state_text = str(report.get("state_text", "")).strip()
        if state_text:
            lines.append("```text")
            lines.append(state_text[:4000])
            lines.append("```")
            lines.append("")
        lines.append(f"- n_candidates: `{report['n_candidates']}`")
        lines.append(f"- requested_root_clusters: `{report['requested_root_clusters']}`")
        lines.append("")
        for method, result in report["methods"].items():
            lines.append(f"#### `{method}`")
            if "error" in result:
                lines.append(f"- error: `{result['error']}`")
                lines.append("")
                continue
            lines.append(
                f"- actual_clusters: `{result['cluster_metadata'].get('n_clusters')}`, "
                f"representatives: `{result['representative_indices']}`, "
                f"selected: `{result['selected_indices']}`"
            )
            lines.append(
                f"- mean_representative_regret: `{float(result['utility']['mean_representative_regret']):.6f}`, "
                f"max_representative_regret: `{float(result['utility']['max_representative_regret']):.6f}`"
            )
            for cluster in result["clusters"]:
                lines.append(
                    f"- cluster `{cluster['cluster_id']}` size=`{cluster['size']}` "
                    f"rep=`{cluster['representative_index']}` selected=`{cluster['selected_index']}`"
                )
                for member in cluster["members"]:
                    flags = []
                    if member["is_representative"]:
                        flags.append("R")
                    if member["is_selected"]:
                        flags.append("S")
                    prefix = f"[{','.join(flags)}] " if flags else ""
                    q_value = member["q_value"]
                    log_prob = member["log_prob"]
                    lines.append(
                        f"  {prefix}#{member['candidate_index']} "
                        f"q={_format_optional_float(q_value)} "
                        f"logp={_format_optional_float(log_prob)} "
                        f"{_one_line(member['action_text'])}"
                    )
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _serialize_candidate(node: Any, *, index: int) -> dict[str, Any]:
    return {
        "candidate_index": index,
        "action_text": getattr(node, "action_text", ""),
        "action_tokens": list(getattr(node, "action_tokens", [])),
        "log_prob": _safe_float(getattr(node, "log_prob", None)),
        "q_value": _safe_float(getattr(node, "q_value", None)),
    }


def _normalize_prompt_item(item: Any, reset_kwargs: Mapping[str, Any] | None) -> Any:
    if not isinstance(item, Mapping):
        return item
    normalized = dict(item)
    if "env_reset_kwargs" not in normalized and reset_kwargs:
        normalized["env_reset_kwargs"] = dict(reset_kwargs)
    return normalized


def _resolve_field(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ModuleNotFoundError:
        pass

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_python(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _to_python(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_to_python(item) for item in value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _to_python(tolist())
        except Exception:
            return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            return value
    return value


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _max_q(candidates: Sequence[Any]) -> float | None:
    values = [_safe_float(getattr(node, "q_value", None)) for node in candidates]
    finite = [value for value in values if value is not None]
    return max(finite) if finite else None


def _argmax_q(candidates: Sequence[Any]) -> int | None:
    best_index: int | None = None
    best_value: float | None = None
    for index, node in enumerate(candidates):
        q_value = _safe_float(getattr(node, "q_value", None))
        if q_value is None:
            continue
        if best_value is None or q_value > best_value:
            best_value = q_value
            best_index = index
    return best_index


def _format_optional_float(value: float | None) -> str:
    return "None" if value is None else f"{value:.6f}"


def _one_line(text: str) -> str:
    text = str(text).replace("\n", " ").strip()
    return text[:300]


def _cleanup_audit_resources(
    *,
    inference_engine: Any,
    rollout: Any,
    embedding_matrix_cache: Any,
    q_critic: Any,
    q_head: Any,
    actor_module: Any,
    tokenizer: Any,
) -> None:
    del rollout
    del embedding_matrix_cache
    del q_critic
    del q_head
    del actor_module
    del tokenizer

    if inference_engine is not None:
        shutdown = getattr(inference_engine, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception:
                pass
    del inference_engine

    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
