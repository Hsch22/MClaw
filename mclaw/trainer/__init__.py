"""训练模块导出。"""

from .contracts import ActorBackendProtocol, LoggerProtocol, ReferencePolicyProtocol
from .main import build_arg_parser, build_trainer, load_config, main
from .mclaw_trainer import MClawTrainer

__all__ = [
    "ActorBackendProtocol",
    "LoggerProtocol",
    "MClawTrainer",
    "ReferencePolicyProtocol",
    "build_arg_parser",
    "build_trainer",
    "load_config",
    "main",
]
