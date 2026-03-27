from __future__ import annotations

"""MClaw 训练入口接口。"""

import argparse
from typing import Sequence

from mclaw.config import DEFAULT_CONFIG_PATH, MClawTrainerConfig

from .mclaw_trainer import MClawTrainer


def build_arg_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="MClaw training entrypoint")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Hydra/YAML 配置文件路径。",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="可选的 checkpoint 路径。",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="额外的命令行覆盖项，例如 mclaw.tree_rollout.max_rounds=8。",
    )
    return parser


def load_config(config_path: str, overrides: Sequence[str] | None = None) -> MClawTrainerConfig:
    """加载配置文件并应用命令行覆盖。"""
    raise NotImplementedError("TODO: 实现配置加载逻辑。")


def build_trainer(config: MClawTrainerConfig) -> MClawTrainer:
    """根据配置组装训练器和外部后端适配器。"""
    raise NotImplementedError("TODO: 实现训练器构造逻辑。")


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口。"""
    raise NotImplementedError("TODO: 实现命令行主入口。")


if __name__ == "__main__":
    raise SystemExit(main())
