# src/llamafactory/model/load_cocoun_trajectory.py
# Loader for COCONUT Trajectory Model

import os
from typing import TYPE_CHECKING, Any, Dict, Optional, TypedDict
import torch

from transformers import (
    AutoConfig,
    Qwen2_5_VLForConditionalGeneration,
)

from ..extras import logging
from ..extras.misc import count_parameters, skip_check_imports, try_download_model_from_other_hub
from .adapter import init_adapter
from .model_utils.liger_kernel import apply_liger_kernel
from .model_utils.misc import register_autoclass
from .patcher import patch_config, patch_tokenizer, patch_processor, patch_model
from .cocoun_trajectory_model import Cocoun_trajectory_model

if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedTokenizer, ProcessorMixin
    from ..hparams import ModelArguments, FinetuningArguments

logger = logging.get_logger(__name__)


def _get_init_kwargs(model_args: "ModelArguments") -> Dict[str, Any]:
    r"""
    Gets arguments to load config/tokenizer/model.
    Note: including inplace operation of model_args.
    """
    skip_check_imports()
    model_args.model_name_or_path = try_download_model_from_other_hub(model_args)
    return {
        "trust_remote_code": model_args.trust_remote_code,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.hf_hub_token,
    }


def load_config(model_args: "ModelArguments") -> "PretrainedConfig":
    r"""
    Loads model config.
    """
    init_kwargs = _get_init_kwargs(model_args)
    return AutoConfig.from_pretrained(model_args.model_name_or_path, **init_kwargs)


def load_trajectory_unlearn_model(
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool = False,
    forget_concept: str = "Joe Biden",
    unlearning_loss_weight: float = 1.0,
    # Trajectory-specific参数
    forbidden_token_ids: Optional[list] = None,
    risk_threshold: float = 0.1,
    redirect_strength: float = 0.5,
) -> Qwen2_5_VLForConditionalGeneration:
    r"""
    Loads COCONUT Trajectory Model for trajectory-level unlearning.

    New features:
    - Logit-Lens Critic: Monitors risk at each latent step
    - Latent Redirector: Dynamically rewrites high-risk latent states
    - Trajectory Loss: Ensures safety of the entire reasoning trajectory
    """

    init_kwargs = _get_init_kwargs(model_args)
    config = load_config(model_args)
    patch_config(config, tokenizer, model_args, init_kwargs, is_trainable)

    # 1. 获取 COCONUT 特殊 token IDs
    try:
        latent_token_id = tokenizer.convert_tokens_to_ids("<latent>")
        start_latent_id = tokenizer.convert_tokens_to_ids("<latent_start>")
        end_latent_id = tokenizer.convert_tokens_to_ids("<latent_end>")

        if latent_token_id == tokenizer.unk_token_id:
            raise ValueError("<latent> token not found in tokenizer or id is unk_token_id")
        if start_latent_id == tokenizer.unk_token_id:
             raise ValueError("<latent_start> token not found in tokenizer or id is unk_token_id")
        if end_latent_id == tokenizer.unk_token_id:
             raise ValueError("<latent_end> token not found in tokenizer or id is unk_token_id")

        logger.info_rank0(
            f"[TRAJECTORY] COCONUT tokens: "
            f"latent={latent_token_id}, start={start_latent_id}, end={end_latent_id}"
        )
    except Exception as e:
        raise ValueError(
            f"Failed to get COCONUT token IDs from tokenizer. "
            f"Make sure CoconutDatasetProcessor has added these tokens. Error: {e}"
        )

    # 2. 加载Cocoun_trajectory_model
    load_kwargs = {k: v for k, v in init_kwargs.items() if k != "torch_dtype"}

    # 直接用 from_pretrained 加载
    model = Cocoun_trajectory_model.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=model_args.compute_dtype,
        **load_kwargs,
    )

    logger.info_rank0("Loaded pretrained Cocoun_trajectory_model")

    # 3. 计算forget concept vector
    forget_concept_vector = _compute_forget_concept_vector(tokenizer, model, forget_concept)

    # 4. 设置模型参数
    model.tokenizer = tokenizer
    model.latent_token_id = latent_token_id
    model.start_latent_id = start_latent_id
    model.end_latent_id = end_latent_id
    model.forget_concept_vector = forget_concept_vector
    model.unlearning_loss_weight = unlearning_loss_weight

    # 5. 设置Trajectory-specific参数
    model.forbidden_token_ids = forbidden_token_ids or []
    model.risk_threshold = risk_threshold
    model.redirect_strength = redirect_strength

    logger.info_rank0(
        f"[TRAJECTORY] Set parameters: "
        f"latent_token_id={latent_token_id}, "
        f"start_latent_id={start_latent_id}, "
        f"end_latent_id={end_latent_id}, "
        f"risk_threshold={risk_threshold}, "
        f"redirect_strength={redirect_strength}"
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 6. 应用其他补丁和初始化
    patch_model(model, tokenizer, model_args, is_trainable, add_valuehead=False)
    register_autoclass(config, model, tokenizer)

    # 7. 应用 Liger Kernel (如果需要)
    apply_liger_kernel(config, model_args, is_trainable, require_logits=(finetuning_args.stage not in ["pt", "sft"]))

    # 8. 初始化适配器 (Lora/QLoRA等)
    model = init_adapter(config, model, model_args, finetuning_args, is_trainable)

    # 9. 设置模型训练/评估模式和 dtype
    if not is_trainable:
        model.requires_grad_(False)
        for param in model.parameters():
            if param.data.dtype == torch.float32 and model_args.compute_dtype != torch.float32:
                param.data = param.data.to(model_args.compute_dtype)
        model.eval()
    else:
        model.train()

    trainable_params, all_param = count_parameters(model)
    if is_trainable:
        param_stats = "trainable params: {:,} || all params: {:,} || trainable%: {:.4f}".format(
            trainable_params, all_param, 100 * trainable_params / all_param
        )
    else:
        param_stats = f"all params: {all_param:,}"

    logger.info_rank0(param_stats)

    if model_args.print_param_status and int(os.getenv("LOCAL_RANK", "0")) == 0:
        for name, param in model.named_parameters():
            print(f"name: {name}, dtype: {param.dtype}, device: {param.device}, trainable: {param.requires_grad}")

    logger.info_rank0(
        f"[TRAJECTORY] Model loaded successfully for trajectory-level unlearning. "
        f"Forget concept: '{forget_concept}', "
        f"Unlearning loss weight: {unlearning_loss_weight}, "
        f"Risk threshold: {risk_threshold}, "
        f"Redirect strength: {redirect_strength}"
    )

    return model


def _compute_forget_concept_vector(tokenizer, model_instance, concept):
    """
    计算要遗忘的概念的向量表示
    """
    device = next(model_instance.parameters()).device
    inputs = tokenizer(concept, return_tensors="pt").to(device)
    with torch.no_grad():
        # 获取最后一层的输出作为真正的语义概念向量
        outputs = model_instance.base_model(input_ids=inputs.input_ids, output_hidden_states=True)
        # 取最后一层的最后一个 token 或者所有 token 的平均
        concept_vector = outputs.hidden_states[-1][:, -1, :].squeeze(0)
    return concept_vector
