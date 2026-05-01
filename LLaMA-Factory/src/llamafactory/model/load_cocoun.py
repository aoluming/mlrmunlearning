# src/models/loader_unlearn.py
# ... (imports remain similar, ensure Qwen2_5_VLForConditionalGeneration is imported) ...
import os
from typing import TYPE_CHECKING, Any, Dict, Optional, TypedDict
import torch

from transformers import (
    AutoConfig,
    AutoModelForCausalLM, # Potentially needed for register_autoclass or other general uses
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration, # The specific model we are extending
)

# Assume these are importable from a common utility module or directly from loader.py
# If not, you'll need to copy their definitions here.
from ..extras import logging
from ..extras.misc import count_parameters, skip_check_imports, try_download_model_from_other_hub
from .adapter import init_adapter
from .model_utils.liger_kernel import apply_liger_kernel
from .model_utils.misc import register_autoclass
from .patcher import patch_config, patch_tokenizer, patch_processor, patch_model
from .cocoun_model import Cocoun_model # Import our new class

if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedTokenizer, ProcessorMixin
    from ..hparams import ModelArguments, FinetuningArguments

logger = logging.get_logger(__name__)


class TokenizerModule(TypedDict):
    tokenizer: "PreTrainedTokenizer"
    processor: Optional["ProcessorMixin"]


def _get_init_kwargs(model_args: "ModelArguments") -> Dict[str, Any]:
    r"""
    Gets arguments to load config/tokenizer/model.
    Note: including inplace operation of model_args.
    """
    skip_check_imports() # This should be in extras.misc
    model_args.model_name_or_path = try_download_model_from_other_hub(model_args) # This should be in extras.misc
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

# Note: load_tokenizer is not strictly needed here for the model loading,
# as the tokenizer should be loaded *before* calling load_unlearn_model,
# and then passed as an argument. However, if there are specific processor
# related patches that need to be applied with a config, having load_tokenizer
# or its components available can be useful. For now, I'll omit the full
# load_tokenizer func as it should happen before this module is called.

# =========================================================
#  COCONUT Specific Loader
# =========================================================

def load_unlearn_model(
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool = False,
    forget_concept: str = "Joe Biden",
    unlearning_loss_weight: float = 1.0,
) -> Qwen2_5_VLForConditionalGeneration: # Return type is our custom model
    r"""
    Loads pretrained Qwen2_5_VLForConditionalGeneration model and wraps it for COCONUT unlearning tasks.
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
            f"[COCONUT] COCONUT tokens: "
            f"latent={latent_token_id}, start={start_latent_id}, end={end_latent_id}"
        )
    except Exception as e:
        raise ValueError(
            f"Failed to get COCONUT token IDs from tokenizer. "
            f"Make sure CoconutDatasetProcessor has added these tokens. Error: {e}"
        )

    # 2. 从配置文件创建我们的自定义模型实例
    # 注意：这里我们直接创建 CoconutQwen2_5_VLForConditionalGeneration 实例，
    # 而不是先加载 Qwen2_5_VLForConditionalGeneration 再包装。
    # 我们的类本身就是 Qwen2_5_VLForConditionalGeneration 的扩展。
    
    # 临时创建模型实例以获取输入嵌入层 (用于计算概念向量)
    # 这是一个小技巧，因为 model.get_input_embeddings() 需要模型实例存在。
    # 我们可以先创建一个不带权重的实例，计算完概念向量后，再加载完整权重。
    # 或者直接使用 Qwen2_5_VLForConditionalGeneration 来计算，因为嵌入层是共享的。
    load_kwargs = {k: v for k, v in init_kwargs.items() if k != "torch_dtype"}

    # 直接用 from_pretrained 加载
    model = Cocoun_model.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=model_args.compute_dtype,
        **load_kwargs,
    )

    logger.info_rank0("Loaded pretrained Cocoun_model")

    # 然后设置 COCONUT 特有的参数
    forget_concept_vector = _compute_forget_concept_vector(tokenizer, model, forget_concept)

    model.tokenizer = tokenizer
    model.latent_token_id = latent_token_id
    model.start_latent_id = start_latent_id
    model.end_latent_id = end_latent_id
    model.forget_concept_vector = forget_concept_vector
    model.unlearning_loss_weight = unlearning_loss_weight

    logger.info_rank0(f"Set COCONUT parameters: latent_token_id={latent_token_id}, "
                    f"start_latent_id={start_latent_id}, end_latent_id={end_latent_id}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


    # 6. 应用其他补丁和初始化 (从 loader.py 复制或导入)
    # patch_model 应该在加载权重之后，因为它可能修改模型结构
    patch_model(model, tokenizer, model_args, is_trainable, add_valuehead=False) # add_valuehead=False for unlearning task
    register_autoclass(config, model, tokenizer) # Ensure our custom model is registered for AutoModel

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
        f"[COCONUT] Model loaded successfully for unlearning. "
        f"Forget concept: '{forget_concept}', "
        f"Unlearning loss weight: {unlearning_loss_weight}"
    )

    return model


# def _compute_forget_concept_vector(
#     tokenizer: "PreTrainedTokenizer",
#     model_instance: Qwen2_5_VLForConditionalGeneration, # Expect an instance of the base model or our inherited model
#     concept: str,
# ) -> torch.Tensor:
#     r"""
#     计算要遗忘的概念的向量表示。
#     通过对概念中所有 token 的嵌入进行平均汇聚得到。
    
#     Args:
#         tokenizer: Tokenizer
#         model_instance: An instance of the Qwen2_5_VLForConditionalGeneration (or inherited) model
#         concept: 要遗忘的概念文本
    
#     Returns:
#         Concept vector of shape (hidden_dim,)
#     """
#     device = next(model_instance.parameters()).device # Get device from model parameters
    
#     # Tokenize 概念
#     concept_ids = tokenizer.encode(concept, add_special_tokens=False)
#     if not concept_ids:
#         logger.warning_rank0(f"Concept '{concept}' tokenized to empty list. Returning zero vector.")
#         return torch.zeros(model_instance.config.hidden_size, device=device, dtype=model_instance.dtype)

#     concept_ids_tensor = torch.tensor(concept_ids, device=device).unsqueeze(0)  # (1, seq_len)
    
#     # 获取嵌入
#     with torch.no_grad():
#         # Ensure we use the correct input embeddings from the model
#         embeddings = model_instance.get_input_embeddings()(concept_ids_tensor)  # (1, seq_len, hidden_dim)
#         concept_vector = embeddings.mean(dim=1).squeeze(0)  # (hidden_dim,)
    
#     logger.info_rank0(f"[COCONUT] Forget concept vector computed for '{concept}': shape={concept_vector.shape}, dtype={concept_vector.dtype}")
    
#     return concept_vector

def _compute_forget_concept_vector(tokenizer, model_instance, concept):
    device = next(model_instance.parameters()).device
    inputs = tokenizer(concept, return_tensors="pt").to(device)
    with torch.no_grad():
        # 获取最后一层的输出作为真正的语义概念向量
        outputs = model_instance.base_model(input_ids=inputs.input_ids, output_hidden_states=True)
        # 取最后一层的最后一个 token 或者所有 token 的平均
        concept_vector = outputs.hidden_states[-1][:, -1, :].squeeze(0) 
    return concept_vector