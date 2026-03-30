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
        self.output_proj = nn.Linear(intermediate_dim, 1)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, intermediate_dim),
            nn.GELU(),
            self.output_proj,
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """将最后一层置零，避免初始 Q 值抖动过大。"""
        if torch is None:
            raise ModuleNotFoundError("torch is required to initialize QHead.")
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """将 action hidden state 映射成标量 Q 值。"""
        if torch is None:
            raise ModuleNotFoundError("torch is required to run QHead.")
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError("hidden_states must be a torch.Tensor")
        if hidden_states.ndim != 2:
            raise ValueError(
                "hidden_states must be 2D, "
                f"got shape {tuple(hidden_states.shape)}"
            )
        if hidden_states.size(-1) != self.hidden_dim:
            raise ValueError(
                f"expected hidden_states dim {self.hidden_dim}, "
                f"got {hidden_states.size(-1)}"
            )
        return self.mlp(hidden_states).squeeze(-1)
