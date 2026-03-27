"""Critic 模块导出。"""

from .advantage import AdvantageComputationResult, compute_tree_advantage, estimate_state_value
from .q_critic import QCritic, QCriticOutput
from .q_head import QHead

__all__ = [
    "AdvantageComputationResult",
    "QCritic",
    "QCriticOutput",
    "QHead",
    "compute_tree_advantage",
    "estimate_state_value",
]
