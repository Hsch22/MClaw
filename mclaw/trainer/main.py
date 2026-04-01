from __future__ import annotations

"""MClaw 训练入口接口。"""

import argparse
from datetime import datetime
import inspect
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

from mclaw.adapters import (
    AgentEnvClientAdapter,
    DataProtoAdapter,
    StandardLogger,
    VerlActorBackend,
    VerlInferenceEngine,
    VerlReferencePolicy,
    VerlRolloutHandler,
    build_tracker,
)
from mclaw.clustering import (
    HiddenStateClusterer,
    LogProbClusterer,
    LogitDistributionClusterer,
    OutputGradClusterer,
)
from mclaw.config import DEFAULT_CONFIG_PATH, MClawTrainerConfig, trainer_config_from_mapping
from mclaw.core import BranchSelector, TreeRollout
from mclaw.critic import QCritic, QHead

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
    OmegaConf = _import_omegaconf()

    yaml_config = OmegaConf.load(config_path)
    if overrides:
        yaml_config = OmegaConf.merge(
            yaml_config,
            OmegaConf.from_dotlist(list(overrides)),
        )

    container = OmegaConf.to_container(yaml_config, resolve=True)
    if not isinstance(container, dict):
        raise TypeError("top-level config must resolve to a mapping")

    plain_config = dict(container)
    mclaw_namespace = plain_config.pop("mclaw", {})
    if isinstance(mclaw_namespace, dict):
        plain_config.update(mclaw_namespace)

    return trainer_config_from_mapping(plain_config)


def build_trainer(config: MClawTrainerConfig) -> MClawTrainer:
    """根据配置组装训练器和外部后端适配器。"""
    _configure_python_logging(config)

    tokenizer = _build_tokenizer(config)
    actor_module = _build_actor_module(config)
    actor_module_fsdp = _wrap_model_with_fsdp_if_enabled(actor_module, config)
    q_head = _build_q_head(config, actor_module_fsdp)
    q_critic = QCritic(
        actor_module_fsdp=actor_module_fsdp,
        q_head=q_head,
        tokenizer=tokenizer,
        config=config.q_critic,
    )

    clusterer = _build_clusterer(config)
    dataproto_adapter = DataProtoAdapter(
        pad_token_id=_resolve_pad_token_id(tokenizer),
        default_meta_info=_build_dataproto_meta_info(config),
    )
    actor_backend = _build_actor_backend(
        config=config,
        actor_module=actor_module_fsdp,
        dataproto_adapter=dataproto_adapter,
    )
    ref_policy = _build_ref_policy(
        config=config,
        dataproto_adapter=dataproto_adapter,
    )
    inference_engine = _build_inference_engine(config)
    env_client_factory = _build_env_client_factory(config)
    branch_selector = BranchSelector()
    rollout_sharding_manager = _build_rollout_sharding_manager(
        config=config,
        actor_module_fsdp=actor_module_fsdp,
        inference_engine=inference_engine,
    )
    training_sharding_manager = _build_training_sharding_manager(
        config=config,
        rollout_sharding_manager=rollout_sharding_manager,
    )
    logger = _build_logger(config)
    rollout_handler_factory = _build_rollout_handler_factory(
        config=config,
        tokenizer=tokenizer,
    )

    tree_rollout = TreeRollout(
        inference_engine=inference_engine,
        actor_module_fsdp=actor_module_fsdp,
        q_critic=q_critic,
        clusterer=clusterer,
        tokenizer=tokenizer,
        config=config.tree_rollout,
        env_client_factory=env_client_factory,
        branch_selector=branch_selector,
        handler_factory=rollout_handler_factory,
    )

    return MClawTrainer(
        config=config,
        tree_rollout=tree_rollout,
        actor=actor_backend,
        ref_policy=ref_policy,
        q_critic=q_critic,
        logger=logger,
        rollout_sharding_manager=rollout_sharding_manager,
        training_sharding_manager=training_sharding_manager,
        tokenizer=tokenizer,
        inference_engine=inference_engine,
        clusterer=clusterer,
        env_client_factory=env_client_factory,
        branch_selector=branch_selector,
        rollout_handler_factory=rollout_handler_factory,
        dataproto_adapter=dataproto_adapter,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口。"""
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = load_config(args.config, args.overrides)
    if args.resume:
        config.trainer.resume_from = str(args.resume)

    trainer = build_trainer(config)
    try:
        trainer.fit()
    finally:
        logger = getattr(trainer, "logger", None)
        close = getattr(logger, "close", None)
        if callable(close):
            close()
    return 0


def _build_tokenizer(config: MClawTrainerConfig) -> Any:
    AutoTokenizer = _import_auto_tokenizer()

    model_path = _resolve_model_path(config)
    tokenizer_path = config.model.tokenizer_path or model_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=bool(config.model.trust_remote_code),
        revision=config.model.revision,
    )

    if getattr(tokenizer, "pad_token_id", None) is None and bool(config.model.pad_token_as_eos):
        eos_token = getattr(tokenizer, "eos_token", None)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token is not None:
            tokenizer.pad_token = eos_token
        elif eos_token_id is not None:
            tokenizer.pad_token_id = eos_token_id
    return tokenizer


def _build_actor_module(config: MClawTrainerConfig) -> Any:
    AutoModelForCausalLM = _import_auto_model_for_causal_lm()
    torch = _import_torch()

    model_path = _resolve_model_path(config)
    torch_dtype = _resolve_torch_dtype(config.model.dtype, torch_module=torch)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=bool(config.model.trust_remote_code),
        revision=config.model.revision,
        torch_dtype=torch_dtype,
    )

    device = _resolve_device(config, torch_module=torch)
    if device is not None:
        model = model.to(device)
    return model


def _wrap_model_with_fsdp_if_enabled(model: Any, config: MClawTrainerConfig) -> Any:
    if not bool(config.distributed.enable_fsdp):
        return model

    torch = _import_torch()
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError(
            "distributed.enable_fsdp=True requires torch.distributed to be initialized; "
            "launch with torchrun or disable FSDP"
        )

    FSDP = _import_fsdp()
    fsdp_kwargs = _filter_kwargs_for_callable(FSDP, dict(config.distributed.fsdp_config))
    return FSDP(model, **fsdp_kwargs)


def _build_q_head(config: MClawTrainerConfig, actor_module: Any) -> QHead:
    torch = _import_torch()

    hidden_dim = int(config.q_critic.hidden_dim)
    actor_config = getattr(actor_module, "config", None)
    actor_hidden_dim = getattr(actor_config, "hidden_size", None)
    if actor_hidden_dim is not None and int(actor_hidden_dim) != hidden_dim:
        hidden_dim = int(actor_hidden_dim)

    q_head = QHead(
        hidden_dim=hidden_dim,
        intermediate_dim=int(config.q_critic.intermediate_dim),
    )
    device = _resolve_module_device(actor_module, torch_module=torch)
    return q_head.to(device)


def _build_clusterer(config: MClawTrainerConfig) -> Any:
    method = str(config.clustering.method).strip().lower()
    if method == "hidden_state":
        return HiddenStateClusterer(config.clustering)
    if method == "output_grad":
        return OutputGradClusterer(config.clustering)
    if method == "logit_distribution":
        return LogitDistributionClusterer(config.clustering)
    if method == "logprob":
        return LogProbClusterer(config.clustering)
    raise ValueError(f"unsupported clustering.method: {config.clustering.method}")


def _build_actor_backend(
    *,
    config: MClawTrainerConfig,
    actor_module: Any,
    dataproto_adapter: DataProtoAdapter,
) -> VerlActorBackend | None:
    backend_name = str(config.adapter.actor_backend).strip().lower()
    if not backend_name:
        return None
    if backend_name != "verl":
        raise NotImplementedError(f"unsupported actor backend adapter: {config.adapter.actor_backend}")

    DataParallelPPOActor, OmegaConf = _import_verl_actor_backend()
    torch = _import_torch()
    if not torch.cuda.is_available():
        raise RuntimeError("VerlActorBackend requires CUDA; no visible GPU was found")

    actor_cfg = _build_actor_backend_config(config)
    learning_rate = float(actor_cfg.get("optim", {}).get("lr", 1e-6))
    parameters = [parameter for parameter in actor_module.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
    dp_actor = DataParallelPPOActor(
        config=OmegaConf.create(actor_cfg),
        actor_module=actor_module,
        actor_optimizer=optimizer,
    )
    return VerlActorBackend(
        actor=dp_actor,
        adapter=dataproto_adapter,
        dataproto_meta_info=_build_dataproto_meta_info(config),
        aux_loss_config={
            "coef": float(config.aux_loss.coef),
            "use_same_advantage": bool(config.aux_loss.use_same_advantage),
            "weighting": str(config.aux_loss.weighting),
        },
    )


def _build_ref_policy(
    *,
    config: MClawTrainerConfig,
    dataproto_adapter: DataProtoAdapter,
) -> VerlReferencePolicy | None:
    if not bool(_build_actor_backend_config(config).get("use_kl_loss", False)):
        return None

    DataParallelPPOActor, OmegaConf = _import_verl_actor_backend()
    ref_model = _build_reference_model(config)
    ref_cfg = _build_ref_backend_config(config)
    ref_actor = DataParallelPPOActor(
        config=OmegaConf.create(ref_cfg),
        actor_module=ref_model,
        actor_optimizer=None,
    )
    return VerlReferencePolicy(
        ref_policy=ref_actor,
        adapter=dataproto_adapter,
        dataproto_meta_info=_build_ref_dataproto_meta_info(config),
    )


def _build_reference_model(config: MClawTrainerConfig) -> Any:
    ref_model = _build_actor_module(config)
    if hasattr(ref_model, "eval"):
        ref_model.eval()
    for parameter in ref_model.parameters():
        parameter.requires_grad = False
    return ref_model


def _build_inference_engine(config: MClawTrainerConfig) -> VerlInferenceEngine:
    engine_name = str(config.adapter.inference_engine).strip().lower()
    if engine_name not in {"verl_vllm", "vllm"}:
        raise NotImplementedError(
            f"unsupported inference engine adapter: {config.adapter.inference_engine}"
        )

    torch = _import_torch()
    if not torch.cuda.is_available():
        raise RuntimeError("VerlInferenceEngine/vLLM requires CUDA; no visible GPU was found")

    LLM = _import_llm()
    model_path = _resolve_model_path(config)
    tokenizer_path = config.model.tokenizer_path or model_path
    rollout_cfg = _as_mapping(config.actor_rollout_ref.get("rollout"))

    llm_kwargs = {
        "model": model_path,
        "tokenizer": tokenizer_path,
        "tensor_parallel_size": max(int(config.distributed.tensor_parallel_size), 1),
        "dtype": config.model.dtype,
    }
    for key in ("gpu_memory_utilization", "max_model_len", "max_num_batched_tokens", "max_num_seqs"):
        if key in rollout_cfg:
            llm_kwargs[key] = rollout_cfg[key]

    sampling_kwargs = {
        "max_tokens": int(rollout_cfg.get("max_tokens", config.data.max_response_length)),
        "temperature": float(rollout_cfg.get("temperature", 1.0)),
        "n": int(rollout_cfg.get("n", 1)),
        "logprobs": int(rollout_cfg.get("logprobs", 1)),
    }
    for key in ("top_p", "top_k", "presence_penalty", "frequency_penalty"):
        if key in rollout_cfg:
            sampling_kwargs[key] = rollout_cfg[key]

    llm = LLM(**llm_kwargs)
    return VerlInferenceEngine(llm=llm, sampling_kwargs=sampling_kwargs)


def _build_env_client_factory(config: MClawTrainerConfig) -> Any:
    if str(config.adapter.env_client).strip().lower() != "agentenv":
        raise NotImplementedError(f"unsupported env client adapter: {config.adapter.env_client}")
    if not config.adapter.task_name or not config.adapter.env_addr:
        raise ValueError("adapter.task_name and adapter.env_addr are required for env client setup")

    args = SimpleNamespace(
        task_name=config.adapter.task_name,
        env_addr=config.adapter.env_addr,
        max_retries=max(int(config.adapter.max_retries), 1),
    )

    def _factory() -> AgentEnvClientAdapter:
        return AgentEnvClientAdapter.from_agentgym_args(args)

    return _factory


def _build_rollout_sharding_manager(
    *,
    config: MClawTrainerConfig,
    actor_module_fsdp: Any,
    inference_engine: VerlInferenceEngine,
) -> Any | None:
    if not bool(config.distributed.use_rollout_sharding_manager):
        return None
    if not bool(config.distributed.enable_fsdp):
        return None

    FSDP = _import_fsdp()
    if not isinstance(actor_module_fsdp, FSDP):
        return None

    FSDPVLLMShardingManager = _import_rollout_sharding_manager()
    return FSDPVLLMShardingManager(
        module=actor_module_fsdp,
        inference_engine=inference_engine.llm,
        model_config=getattr(actor_module_fsdp, "config", None),
        full_params=bool(config.distributed.full_params),
    )


def _build_training_sharding_manager(
    *,
    config: MClawTrainerConfig,
    rollout_sharding_manager: Any | None,
) -> Any | None:
    if not bool(config.distributed.use_training_sharding_manager):
        return None
    return rollout_sharding_manager


def _build_logger(config: MClawTrainerConfig) -> StandardLogger:
    python_logger = logging.getLogger("mclaw")
    python_logger.setLevel(_resolve_log_level(config.logging.level))
    tracker = build_tracker(
        tracker_name=config.logging.tracker,
        project_name=config.logging.project_name,
        experiment_name=config.logging.experiment_name,
        default_local_dir=config.trainer.default_local_dir,
        path_pattern=config.logging.path_pattern,
        tracker_kwargs=config.logging.tracker_kwargs,
    )
    return StandardLogger(tracker=tracker, python_logger=python_logger)


def _configure_python_logging(config: MClawTrainerConfig) -> None:
    logger = logging.getLogger("mclaw")
    logger.setLevel(_resolve_log_level(config.logging.level))
    logger.propagate = False
    if logger.handlers:
        return

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    path_pattern = str(config.logging.path_pattern).strip()
    if path_pattern:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = Path(path_pattern.format(timestamp=timestamp))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def _build_dataproto_meta_info(config: MClawTrainerConfig) -> dict[str, Any]:
    actor_cfg = _build_actor_backend_config(config)
    rollout_cfg = _as_mapping(config.actor_rollout_ref.get("rollout"))
    return {
        "micro_batch_size": int(actor_cfg.get("ppo_micro_batch_size_per_gpu", 1)),
        "temperature": float(rollout_cfg.get("temperature", 1.0)),
        "use_dynamic_bsz": bool(actor_cfg.get("use_dynamic_bsz", False)),
    }


def _build_ref_dataproto_meta_info(config: MClawTrainerConfig) -> dict[str, Any]:
    ref_cfg = _as_mapping(config.actor_rollout_ref.get("ref"))
    rollout_cfg = _as_mapping(config.actor_rollout_ref.get("rollout"))
    actor_cfg = _build_actor_backend_config(config)
    return {
        "micro_batch_size": int(
            ref_cfg.get(
                "log_prob_micro_batch_size_per_gpu",
                actor_cfg.get("ppo_micro_batch_size_per_gpu", 1),
            )
        ),
        "temperature": float(rollout_cfg.get("temperature", 1.0)),
        "use_dynamic_bsz": bool(actor_cfg.get("use_dynamic_bsz", False)),
    }


def _build_rollout_handler_factory(
    *,
    config: MClawTrainerConfig,
    tokenizer: Any,
) -> Any | None:
    handler_name = str(config.adapter.rollout_handler).strip().lower()
    if not handler_name or handler_name == "none":
        return None
    if handler_name != "verl":
        raise NotImplementedError(
            f"unsupported rollout handler adapter: {config.adapter.rollout_handler}"
        )

    rollout_cfg = _as_mapping(config.actor_rollout_ref.get("rollout"))
    default_chat_format = _resolve_default_chat_format(config)
    max_response_len = int(rollout_cfg.get("max_tokens", config.data.max_response_length))
    max_model_len = int(
        rollout_cfg.get("max_model_len", getattr(tokenizer, "model_max_length", 32768) or 32768)
    )

    def _factory(prompt_item: Any, root: Any) -> VerlRolloutHandler | None:
        messages = _extract_prompt_messages(prompt_item)
        prompt_ids = _extract_prompt_token_ids(prompt_item, root)
        if not messages and not prompt_ids:
            return None
        # 若数据集只有 item_id，messages 为空但 root 已有来自环境观测的
        # state_tokens/state_text，则构造一条 user message 让 chat template
        # 能正常生成 generation prompt。
        if not messages and prompt_ids:
            obs_text = getattr(root, "state_text", None)
            if obs_text:
                from mclaw.adapters.rollout_handler import RolloutMessage
                messages = [RolloutMessage(role="user", content=str(obs_text))]
        return VerlRolloutHandler(
            tokenizer=tokenizer,
            messages=messages or [],
            prompt_ids=prompt_ids,
            task_name=str(config.adapter.task_name),
            item_id=_extract_prompt_item_value(prompt_item, "item_id", "index"),
            max_response_len=max_response_len,
            max_model_len=max_model_len,
            chat_format=str(
                _extract_prompt_item_value(prompt_item, "chat_format") or default_chat_format
            ),
        )

    return _factory


def _extract_prompt_messages(prompt_item: Any) -> list[Any]:
    messages = _extract_prompt_item_value(prompt_item, "messages", "prompt_messages")
    if isinstance(messages, list):
        return list(messages)
    if isinstance(messages, tuple):
        return list(messages)
    return []


def _extract_prompt_token_ids(prompt_item: Any, root: Any) -> list[int]:
    token_ids = _extract_prompt_item_value(prompt_item, "prompt_token_ids", "input_ids")
    if isinstance(token_ids, (list, tuple)):
        return [int(token_id) for token_id in token_ids]
    state_tokens = getattr(root, "state_tokens", None)
    if isinstance(state_tokens, list):
        return [int(token_id) for token_id in state_tokens]
    return []


def _extract_prompt_item_value(prompt_item: Any, *keys: str) -> Any:
    for key in keys:
        if isinstance(prompt_item, dict) and key in prompt_item:
            return prompt_item[key]
        if hasattr(prompt_item, key):
            return getattr(prompt_item, key)
    return None


def _resolve_default_chat_format(config: MClawTrainerConfig) -> str:
    model_path = _resolve_model_path(config).lower()
    if "qwen" in model_path:
        return "qwen"
    return "qwen"


def _build_actor_backend_config(config: MClawTrainerConfig) -> dict[str, Any]:
    actor_cfg = {
        "clip_ratio": 0.2,
        "entropy_coeff": 0.0,
        "grad_clip": 1.0,
        "kl_loss_coef": 0.001,
        "kl_loss_type": "low_var_kl",
        "ppo_mini_batch_size": 1,
        "ppo_micro_batch_size_per_gpu": 1,
        "ppo_max_token_len_per_gpu": 2048,
        "use_dynamic_bsz": False,
        "use_kl_loss": False,
        "ulysses_sequence_parallel_size": 1,
        "use_remove_padding": bool(config.model.use_remove_padding),
        "entropy_from_logits_with_chunking": False,
        "entropy_checkpointing": False,
        "fsdp_config": {"dtype": "bfloat16"},
        "optim": {
            "lr": 1e-6,
        },
    }
    actor_cfg.update(_as_mapping(config.actor_rollout_ref.get("actor")))
    if "optim" not in actor_cfg:
        actor_cfg["optim"] = {"lr": 1e-6}
    return actor_cfg


def _build_ref_backend_config(config: MClawTrainerConfig) -> dict[str, Any]:
    actor_cfg = _build_actor_backend_config(config)
    ref_cfg = dict(actor_cfg)
    ref_cfg.update(_as_mapping(config.actor_rollout_ref.get("ref")))
    ref_cfg["use_kl_loss"] = False
    return ref_cfg


def _resolve_model_path(config: MClawTrainerConfig) -> str:
    if config.model.model_path:
        return config.model.model_path
    if config.model.family:
        return config.model.family

    model_cfg = _as_mapping(config.actor_rollout_ref.get("model"))
    model_path = model_cfg.get("path")
    if isinstance(model_path, str) and model_path:
        return model_path
    raise ValueError("model.family or model.model_path must be set")


def _resolve_pad_token_id(tokenizer: Any) -> int:
    for attribute in ("pad_token_id", "eos_token_id", "bos_token_id"):
        value = getattr(tokenizer, attribute, None)
        if value is not None:
            return int(value)
    return 0


def _resolve_torch_dtype(dtype_name: str, *, torch_module: Any) -> Any:
    normalized = str(dtype_name).strip().lower()
    mapping = {
        "bf16": torch_module.bfloat16,
        "bfloat16": torch_module.bfloat16,
        "fp16": torch_module.float16,
        "float16": torch_module.float16,
        "fp32": torch_module.float32,
        "float32": torch_module.float32,
    }
    return mapping.get(normalized, torch_module.bfloat16)


def _resolve_device(config: MClawTrainerConfig, *, torch_module: Any) -> Any | None:
    requested = str(config.distributed.device).strip().lower()
    if requested.startswith("cuda") and torch_module.cuda.is_available():
        return torch_module.device(requested)
    if requested == "cpu":
        return torch_module.device("cpu")
    if requested and requested != "cuda":
        return torch_module.device(requested)
    if torch_module.cuda.is_available():
        return torch_module.device("cuda")
    return torch_module.device("cpu")


def _resolve_module_device(module: Any, *, torch_module: Any) -> Any:
    try:
        parameter = next(module.parameters())
    except (StopIteration, AttributeError, TypeError):
        if torch_module.cuda.is_available():
            return torch_module.device("cuda")
        return torch_module.device("cpu")
    return parameter.device


def _resolve_log_level(level_name: str) -> int:
    return getattr(logging, str(level_name).strip().upper(), logging.INFO)


def _filter_kwargs_for_callable(target: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):
        return kwargs

    accepted: dict[str, Any] = {}
    parameters = signature.parameters
    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    for key, value in kwargs.items():
        if accepts_var_kwargs or key in parameters:
            accepted[key] = value
    return accepted


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "items"):
        return {str(key): item for key, item in value.items()}
    return {}


def _import_omegaconf() -> Any:
    from omegaconf import OmegaConf

    return OmegaConf


def _import_torch() -> Any:
    import torch

    return torch


def _import_auto_tokenizer() -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer


def _import_auto_model_for_causal_lm() -> Any:
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM


def _import_fsdp() -> Any:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    return FSDP


def _import_verl_actor_backend() -> tuple[Any, Any]:
    from omegaconf import OmegaConf
    from verl.workers.agent_actor.dp_actor import DataParallelPPOActor

    return DataParallelPPOActor, OmegaConf


def _import_rollout_sharding_manager() -> Any:
    from verl.workers.sharding_manager.fsdp_vllm import FSDPVLLMShardingManager

    return FSDPVLLMShardingManager


def _import_llm() -> Any:
    try:
        from verl.third_party.vllm import LLM
    except ModuleNotFoundError:
        from vllm import LLM
    return LLM


if __name__ == "__main__":
    raise SystemExit(main())
