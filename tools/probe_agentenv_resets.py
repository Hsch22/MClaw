#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import requests


TASK_CONFIG = {
    "babyai": {
        "create_path": "/create",
        "reset_path": "/reset",
        "reset_id_key": "id",
        "reset_item_key": "data_idx",
    },
    "maze": {
        "create_path": "/maze/create",
        "reset_path": "/maze/reset",
        "reset_id_key": "id",
        "reset_item_key": "game",
    },
    "weather": {
        "create_path": "/create",
        "reset_path": "/reset",
        "create_payload": {"id": 0},
        "reset_id_key": "env_idx",
        "reset_item_key": "id",
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe AgentEnv create/reset behavior and log JSONL results.")
    parser.add_argument("--task", choices=sorted(TASK_CONFIG), required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    _ensure_local_no_proxy()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = TASK_CONFIG[args.task]
    items = json.loads(Path(args.data_file).read_text(encoding="utf-8"))
    selected = items[max(args.offset, 0) : max(args.offset, 0) + max(args.limit, 0)]

    with output_path.open("w", encoding="utf-8") as handle:
        for row_index, item in enumerate(selected, start=max(args.offset, 0)):
            item_id = str(item["item_id"])
            numeric_id = _coerce_item_id(item_id)
            record = {
                "row_index": row_index,
                "item_id": item_id,
                "numeric_id": numeric_id,
            }
            try:
                create_payload = dict(config.get("create_payload", {}))
                create_response = requests.post(
                    args.base_url.rstrip("/") + config["create_path"],
                    json=create_payload or None,
                    timeout=args.timeout,
                )
                record["create_status"] = create_response.status_code
                record["create_text"] = create_response.text[:500]
                create_response.raise_for_status()
                create_body = create_response.json()
                record["create_body"] = create_body
                env_id = create_body["id"] if isinstance(create_body, dict) else create_body
                reset_payload = {
                    config["reset_id_key"]: env_id,
                    config["reset_item_key"]: numeric_id,
                }
                reset_response = requests.post(
                    args.base_url.rstrip("/") + config["reset_path"],
                    json=reset_payload,
                    timeout=args.timeout,
                )
                record["reset_status"] = reset_response.status_code
                record["reset_text"] = reset_response.text[:1000]
                if reset_response.headers.get("content-type", "").startswith("application/json"):
                    try:
                        record["reset_body"] = reset_response.json()
                    except json.JSONDecodeError:
                        pass
                record["ok"] = reset_response.status_code == 200 and _looks_like_success(
                    args.task, record.get("reset_body"), reset_response.text
                )
            except Exception as exc:  # pragma: no cover - operational helper
                record["ok"] = False
                record["error"] = f"{type(exc).__name__}: {exc}"

            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(json.dumps(record, ensure_ascii=False))
    return 0


def _coerce_item_id(item_id: Any) -> int:
    if isinstance(item_id, int):
        return item_id
    s = str(item_id).strip()
    match = re.search(r"_(\d+)$", s)
    if match:
        return int(match.group(1))
    return int(s)


def _looks_like_success(task: str, body: Any, text: str) -> bool:
    if task == "weather":
        return isinstance(text, str) and bool(text.strip())
    if not isinstance(body, dict):
        return False
    return "observation" in body and "error" not in body


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


if __name__ == "__main__":
    raise SystemExit(main())
