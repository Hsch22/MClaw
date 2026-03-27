from __future__ import annotations

"""Q-head 模块接口。"""

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

    class _FallbackModule:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

    class _UnavailableLayer:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ModuleNotFoundError("torch is required to instantiate QHead.")

    class _FallbackNN:
        Module = _FallbackModule
        Sequential = _UnavailableLayer
        Linear = _UnavailableLayer
        GELU = _UnavailableLayer

    nn = _FallbackNN()  # type: ignore[assignment]


class QHead(nn.Module):
    """接在 frozen backbone 之后的轻量 Q-head。"""

    def __init__(self, hidden_dim: int, intermediate_dim: int = 1024) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        # 先把网络层级占好位，具体前向逻辑后续补齐。
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, 1),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """将 action hidden state 映射成标量 Q 值。"""
        raise NotImplementedError("TODO: 实现 QHead.forward。")
