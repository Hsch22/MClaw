from __future__ import annotations

"""对接 verl RolloutHandler 语义的本地轨迹拼装器。"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from mclaw.core import TokenizerProtocol, TrajectoryRecord


_FORMAT_CONFIG: dict[str, dict[str, str]] = {
    "qwen": {
        "assistant_prefix": "\n<|im_start|>assistant\n",
        "assistant_suffix": "<|im_end|>",
        "user_prefix": "\n<|im_start|>user\n",
        "user_suffix": "<|im_end|>",
    }
}


@dataclass(slots=True)
class RolloutMessage:
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(slots=True)
class VerlRolloutHandler:
    """实现 RolloutHandlerProtocol 的本地 handler。"""

    tokenizer: TokenizerProtocol
    messages: list[RolloutMessage] = field(default_factory=list)
    task_name: str = ""
    item_id: Any | None = None
    score: float = 0.0
    done: bool = False
    input_ids: list[int] = field(default_factory=list)
    attention_mask: list[int] = field(default_factory=list)
    position_ids: list[int] = field(default_factory=list)
    loss_mask: list[int] = field(default_factory=list)
    prompt_ids: list[int] = field(default_factory=list)
    max_response_len: int = 8192
    max_model_len: int = 32768
    chat_format: str = "qwen"
    _assistant_spans: list[tuple[int, int]] = field(default_factory=list, repr=False)
    _step_advantages: list[float] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if self.chat_format not in _FORMAT_CONFIG:
            raise ValueError(f"Unsupported chat format: {self.chat_format}")

        self.messages = [self._coerce_message(message) for message in self.messages]
        if not self.prompt_ids:
            self.prompt_ids = list(self.get_generation_prompt(self.tokenizer))

        if not self.input_ids:
            self.input_ids = list(self.prompt_ids)
        if not self.attention_mask:
            self.attention_mask = [1] * len(self.input_ids)
        if not self.position_ids:
            self.position_ids = list(range(len(self.input_ids)))
        if not self.loss_mask:
            self.loss_mask = [0] * len(self.input_ids)

        self._validate_lengths()

    def add_user_message(self, observation: Any, token_ids: Sequence[int] | None = None) -> None:
        content, resolved_token_ids = self._normalize_content_and_tokens(observation, token_ids)
        self.messages.append(RolloutMessage(role="user", content=content))
        append_token_ids, append_loss_mask = self._build_chat_append(
            role="user",
            content_token_ids=resolved_token_ids,
        )
        self._append_tokens(append_token_ids, append_loss_mask)

    def add_assistant_message(self, action: Any, token_ids: Sequence[int] | None = None) -> None:
        content, resolved_token_ids = self._normalize_content_and_tokens(action, token_ids)
        self.messages.append(RolloutMessage(role="assistant", content=content))
        append_token_ids, append_loss_mask = self._build_chat_append(
            role="assistant",
            content_token_ids=resolved_token_ids,
        )
        start = len(self.input_ids)
        self._append_tokens(append_token_ids, append_loss_mask)
        end = len(self.input_ids)
        self._assistant_spans.append((start, end))

    def record_step_advantage(self, advantage: float) -> None:
        self._step_advantages.append(float(advantage))

    def mark_done(self, done: bool) -> None:
        self.done = bool(done)

    def get_generation_prompt(self, tokenizer: TokenizerProtocol) -> Sequence[int]:
        conversations = [message.to_dict() for message in self.messages]
        apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
        if callable(apply_chat_template):
            return apply_chat_template(
                conversations,
                add_generation_prompt=True,
                tokenize=True,
            )
        return list(self.input_ids)

    def truncate_output_ids(self) -> None:
        self.input_ids = self.input_ids[: self.max_model_len]
        self.attention_mask = self.attention_mask[: self.max_model_len]
        self.position_ids = self.position_ids[: self.max_model_len]
        self.loss_mask = self.loss_mask[: self.max_model_len]
        self._assistant_spans = [
            (start, min(end, len(self.input_ids)))
            for start, end in self._assistant_spans
            if start < len(self.input_ids)
        ]
        self._validate_lengths()

    def build_trajectory_record(self) -> TrajectoryRecord:
        response_ids = self.input_ids[len(self.prompt_ids):][: self.max_response_len]
        # In the verl multi-turn rollout format, `loss_mask` is exactly the assistant-token
        # mask that participates in PPO loss. That is the same semantic contract MClaw uses
        # for `TrajectoryRecord.response_mask`.
        response_mask = self.loss_mask[: len(self.input_ids)]
        response_token_weights = [1.0 if mask else 0.0 for mask in response_mask]
        advantages = [0.0] * len(self.input_ids)
        returns = [0.0] * len(self.input_ids)
        state_values = [0.0] * len(self.input_ids)

        for (start, end), advantage in zip(self._assistant_spans, self._step_advantages):
            for token_index in range(start, min(end, len(advantages))):
                if response_mask[token_index]:
                    advantages[token_index] = float(advantage)

        return TrajectoryRecord(
            input_ids=list(self.input_ids),
            responses=list(response_ids),
            attention_mask=list(self.attention_mask),
            position_ids=list(self.position_ids),
            response_mask=list(response_mask),
            response_token_weights=response_token_weights,
            advantages=advantages,
            returns=returns,
            state_values=state_values,
            old_log_probs=[0.0] * len(self.input_ids),
            ref_log_probs=[0.0] * len(self.input_ids),
            metadata={
                "prompt_length": len(self.prompt_ids),
                "messages": [message.to_dict() for message in self.messages],
                "task_name": self.task_name,
                "item_id": self.item_id,
                "score": float(self.score),
                "done": bool(self.done),
                "chat_format": self.chat_format,
            },
        )

    def _build_chat_append(
        self,
        *,
        role: str,
        content_token_ids: Sequence[int],
    ) -> tuple[list[int], list[int]]:
        format_config = _FORMAT_CONFIG[self.chat_format]
        if role == "assistant":
            prefix_ids = self.tokenizer.encode(
                format_config["assistant_prefix"],
                add_special_tokens=False,
            )
            suffix_ids = self.tokenizer.encode(
                format_config["assistant_suffix"],
                add_special_tokens=False,
            )
            prefix_loss_mask_value = 0
            content_loss_mask_value = 1
            suffix_loss_mask_value = 1
        elif role == "user":
            prefix_ids = self.tokenizer.encode(
                format_config["user_prefix"],
                add_special_tokens=False,
            )
            suffix_ids = self.tokenizer.encode(
                format_config["user_suffix"],
                add_special_tokens=False,
            )
            prefix_loss_mask_value = 0
            content_loss_mask_value = 0
            suffix_loss_mask_value = 0
        else:
            raise ValueError(f"Unsupported role: {role}")

        if self._ends_with_tokens(prefix_ids):
            append_token_ids = list(content_token_ids) + list(suffix_ids)
            append_loss_mask = (
                [content_loss_mask_value] * len(content_token_ids)
                + [suffix_loss_mask_value] * len(suffix_ids)
            )
        elif self._ends_with_tokens(suffix_ids):
            append_token_ids = list(prefix_ids) + list(content_token_ids) + list(suffix_ids)
            append_loss_mask = (
                [prefix_loss_mask_value] * len(prefix_ids)
                + [content_loss_mask_value] * len(content_token_ids)
                + [suffix_loss_mask_value] * len(suffix_ids)
            )
        else:
            max_len = max(len(prefix_ids), len(suffix_ids))
            decoded_suffix = self.tokenizer.decode(
                self.input_ids[-max_len:],
                skip_special_tokens=False,
            )
            raise ValueError(
                "Unsupported end of message format while appending chat message: "
                f"{decoded_suffix!r}"
            )
        return append_token_ids, append_loss_mask

    def _append_tokens(self, token_ids: Sequence[int], loss_mask: Sequence[int]) -> None:
        if len(token_ids) != len(loss_mask):
            raise ValueError("token_ids and loss_mask must have the same length")
        if not token_ids:
            return

        start_position = self.position_ids[-1] + 1 if self.position_ids else 0
        self.input_ids.extend(int(token_id) for token_id in token_ids)
        self.attention_mask.extend([1] * len(token_ids))
        self.position_ids.extend(range(start_position, start_position + len(token_ids)))
        self.loss_mask.extend(int(mask) for mask in loss_mask)
        self._validate_lengths()

    def _normalize_content_and_tokens(
        self,
        content: Any,
        token_ids: Sequence[int] | None,
    ) -> tuple[str, list[int]]:
        if token_ids is not None:
            resolved_token_ids = [int(token_id) for token_id in token_ids]
            if isinstance(content, str):
                return content, resolved_token_ids
            return self.tokenizer.decode(resolved_token_ids, skip_special_tokens=False), resolved_token_ids

        if isinstance(content, str):
            return content, self.tokenizer.encode(content, add_special_tokens=False)

        if isinstance(content, Sequence) and not isinstance(content, (str, bytes, bytearray)):
            resolved_token_ids = [int(token_id) for token_id in content]
            return self.tokenizer.decode(resolved_token_ids, skip_special_tokens=False), resolved_token_ids

        text = str(content)
        return text, self.tokenizer.encode(text, add_special_tokens=False)

    def _coerce_message(self, value: Any) -> RolloutMessage:
        if isinstance(value, RolloutMessage):
            return value
        if isinstance(value, Mapping):
            role = str(value.get("role", "user"))
            content = str(value.get("content", ""))
            return RolloutMessage(role=role, content=content)
        raise TypeError(f"Unsupported message type: {type(value)!r}")

    def _validate_lengths(self) -> None:
        if not (
            len(self.input_ids)
            == len(self.attention_mask)
            == len(self.position_ids)
            == len(self.loss_mask)
        ):
            raise ValueError(
                "Rollout handler internal arrays must have identical lengths: "
                f"input_ids={len(self.input_ids)}, attention_mask={len(self.attention_mask)}, "
                f"position_ids={len(self.position_ids)}, loss_mask={len(self.loss_mask)}"
            )

    def _ends_with_tokens(self, suffix: Sequence[int]) -> bool:
        if not suffix:
            return False
        if len(self.input_ids) < len(suffix):
            return False
        return self.input_ids[-len(suffix):] == list(suffix)


__all__ = ["RolloutMessage", "VerlRolloutHandler"]
