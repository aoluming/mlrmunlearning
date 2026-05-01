# src/models/coconut_model.py
# Copyright 2025 the LlamaFactory team.

"""
COCONUT Model for Machine Unlearning.
Inherits from Qwen2_5_VLForConditionalGeneration to integrate latent token handling and unlearning loss.
"""
import pdb
import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple

# 假设 Qwen2_5_VLForConditionalGeneration 及其相关类位于 transformers 库中
# 或者你将其定义复制到了本地文件，确保能正确导入
from transformers import Qwen2_5_VLForConditionalGeneration, PreTrainedTokenizer
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.nn import CrossEntropyLoss # Qwen2_5_VLForConditionalGeneration 内部使用了 CrossEntropyLoss

import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple

from transformers import Qwen2_5_VLForConditionalGeneration, PreTrainedTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.nn import CrossEntropyLoss

class Cocoun_model(Qwen2_5_VLForConditionalGeneration):
    def __init__(
        self,
        config,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        latent_token_id: Optional[int] = None,
        start_latent_id: Optional[int] = None,
        end_latent_id: Optional[int] = None,
        forget_concept_vector: Optional[torch.Tensor] = None,
        unlearning_loss_weight: float = 1.0,
        unlearn_lm_loss_weight: float = 20, # 新增: 控制梯度上升部分的权重
    ):
        super().__init__(config)

        # 这里的 token IDs 应该从 config 或外部传入，例如像你的 OCR 输出那样
        # [COCONUT] COCONUT tokens: latent-151665, start-151666, end-151667
        # 如果不是通过 config 传入，则需要手动设置默认值或确保参数传递正确
        self.latent_token_id = latent_token_id if latent_token_id is not None else getattr(config, 'latent_token_id', -1)
        self.start_latent_id = start_latent_id if start_latent_id is not None else getattr(config, 'start_latent_id', -1)
        self.end_latent_id = end_latent_id if end_latent_id is not None else getattr(config, 'end_latent_id', -1)
        
        self.tokenizer = tokenizer
        self.forget_concept_vector = forget_concept_vector
        self.unlearning_loss_weight = unlearning_loss_weight
        self.unlearn_lm_loss_weight = unlearn_lm_loss_weight

        self.enable_coconut = True

        if self.latent_token_id != -1:
            print(f"[COCONUT] Initialized Cocoun_model with latent_token_id={self.latent_token_id}, "
                  f"start_latent_id={self.start_latent_id}, end_latent_id={self.end_latent_id}")

    # def forward(
    #     self,
    #     input_ids: Optional[torch.LongTensor] = None,
    #     pixel_values: Optional[torch.FloatTensor] = None,
    #     image_grid_thw: Optional[Tuple[int, int, int]] = None,
    #     video_grid_thw: Optional[Tuple[int, int, int]] = None,
    #     second_per_grid_ts: Optional[torch.Tensor] = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     position_ids: Optional[torch.LongTensor] = None,
    #     past_key_values: Optional[list[torch.FloatTensor]] = None,
    #     inputs_embeds: Optional[torch.FloatTensor] = None,
    #     labels: Optional[torch.LongTensor] = None,
    #     use_cache: Optional[bool] = None,
    #     output_attentions: Optional[bool] = None,
    #     output_hidden_states: Optional[bool] = None,
    #     return_dict: Optional[bool] = None,
    #     cache_position: Optional[torch.LongTensor] = None,
    #     unlearn_mask: Optional[torch.BoolTensor] = None,
    #     **kwargs  # <<<<<<< 添加这一行来捕获其他参数比如 rope_deltas
    # ) -> CausalLMOutputWithPast:
    #     """
    #     Overrides the forward method of Qwen2_5_VLForConditionalGeneration.
    #     Integrates COCONUT's latent token processing, unlearning loss calculation,
    #     and optionally implements gradient ascent for specific 'answer' tokens.
    #     """

    #     output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    #     output_hidden_states = (
    #         output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    #     )
    #     return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    #     # 如果没有启用 COCONUT 或没有 labels (推理模式), 则直接调用父类的 forward 方法
    #     if not self.enable_coconut or labels is None:
    #         # 把 kwargs 也传递给父类
    #         return super().forward(
    #             input_ids=input_ids,
    #             pixel_values=pixel_values,
    #             image_grid_thw=image_grid_thw,
    #             video_grid_thw=video_grid_thw,
    #             second_per_grid_ts=second_per_grid_ts,
    #             attention_mask=attention_mask,
    #             position_ids=position_ids,
    #             past_key_values=past_key_values,
    #             inputs_embeds=inputs_embeds,
    #             labels=labels,
    #             use_cache=use_cache,
    #             output_attentions=output_attentions,
    #             output_hidden_states=output_hidden_states,
    #             return_dict=return_dict,
    #             cache_position=cache_position,
    #             **kwargs  # <<<<<<< 添加这一行
    #         )

    #     # ==================== COCONUT 训练模式下的特殊处理 ====================

    #     # 1. 调用父类的 forward 方法，并强制返回 logits 和 hidden_states，但不让父类计算损失。
    #     #    因为我们要自己精确控制损失的计算和符号。
    #     base_outputs = super().forward(
    #         input_ids=input_ids,
    #         pixel_values=pixel_values,
    #         image_grid_thw=image_grid_thw,
    #         video_grid_thw=video_grid_thw,
    #         second_per_grid_ts=second_per_grid_ts,
    #         attention_mask=attention_mask,
    #         position_ids=position_ids,
    #         past_key_values=past_key_values,
    #         inputs_embeds=inputs_embeds,
    #         labels=None,
    #         use_cache=use_cache,
    #         output_attentions=output_attentions,
    #         output_hidden_states=True,
    #         return_dict=True,
    #         cache_position=cache_position,
    #         **kwargs  # <<<<<<< 添加这一行
    #     )

    #     logits = base_outputs.logits
    #     hidden_states = base_outputs.hidden_states[-1] # 取最后一层的 hidden_states

    #     # 2. 手动计算 LM 损失 (CrossEntropyLoss)，并根据 latent_token_id 和 unlearn_mask 进行调整
    #     # Shift so that tokens < n predict n
    #     shift_logits = logits[..., :-1, :].contiguous()
    #     shift_labels = labels[..., 1:].contiguous()
        
    #     # Flatten the tokens
    #     # batch_size, seq_len_shifted = shift_labels.shape # unused variable
    #     flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    #     flat_shift_labels = shift_labels.view(-1)

    #     # 计算每个 token 的交叉熵损失 (reduction='none' 获取每个 token 的损失)
    #     # CrossEntropyLoss 会自动忽略 label_ids == -100 的位置
    #     per_token_lm_loss = F.cross_entropy(
    #         flat_shift_logits,
    #         flat_shift_labels,
    #         reduction='none',
    #         ignore_index=-100
    #     )
        
    #     # 构建一个 mask，用于额外忽略 COCONUT 的特殊 token (latent_start, latent, latent_end)
    #     # 即使它们的 label_id 不是 -100，我们也根据论文原则将其损失贡献设为0。
    #     # 只有当这些 id 被正确初始化时才进行匹配
    #     valid_lm_loss_mask_specific_tokens = torch.ones_like(flat_shift_labels, dtype=torch.bool)
    #     if self.latent_token_id != -1:
    #         valid_lm_loss_mask_specific_tokens &= (flat_shift_labels != self.latent_token_id)
    #     if self.start_latent_id != -1:
    #         valid_lm_loss_mask_specific_tokens &= (flat_shift_labels != self.start_latent_id)
    #     if self.end_latent_id != -1:
    #         valid_lm_loss_mask_specific_tokens &= (flat_shift_labels != self.end_latent_id)
            
    #     # 同样，只考虑那些不是 -100 的有效标签
    #     is_not_ignored_by_ce = (flat_shift_labels != -100)
        
    #     # 最终的有效 LM 损失 mask：既不是 -100，也不是 COCONUT 特殊 token
    #     final_valid_lm_loss_mask = is_not_ignored_by_ce & valid_lm_loss_mask_specific_tokens
        
    #     # 将被屏蔽的 token 的损失设置为0
    #     per_token_lm_loss = per_token_lm_loss * final_valid_lm_loss_mask.float()
        
    #     # 初始化语言模型总损失
    #     lm_loss_final = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        
    #     # 确保至少有一个有效 token 参与 LM Loss 计算
    #     if torch.any(final_valid_lm_loss_mask): 
    #         # 如果提供了 unlearn_mask，则对它也进行 shift 和 flatten
    #         if unlearn_mask is not None:
    #             shift_unlearn_mask = unlearn_mask[..., 1:].contiguous().view(-1)
                
    #             # unlearn_mask 也要和 final_valid_lm_loss_mask 结合，只处理有效的语言 token
    #             unlearn_part_mask = shift_unlearn_mask & final_valid_lm_loss_mask
    #             learn_part_mask = (~shift_unlearn_mask) & final_valid_lm_loss_mask
                
    #             # 计算正常学习部分的损失 (梯度下降)
    #             if torch.any(learn_part_mask):
    #                 # 避免除以零
    #                 learn_lm_loss = (per_token_lm_loss * learn_part_mask.float()).sum() / learn_part_mask.sum().clamp(min=1e-9)
    #                 lm_loss_final += learn_lm_loss
                
    #             # 计算遗忘部分的损失 (梯度上升，取负号)
    #             if torch.any(unlearn_part_mask):
    #                 # 避免除以零
    #                 unlearn_lm_loss = (per_token_lm_loss * unlearn_part_mask.float()).sum() / unlearn_part_mask.sum().clamp(min=1e-9)
    #                 lm_loss_final += ( -unlearn_lm_loss * self.unlearn_lm_loss_weight) # 应用权重
    #         else:
    #             # 如果没有 unlearn_mask，所有有效语言 token 都进行正常梯度下降
    #             lm_loss_final = per_token_lm_loss.sum() / final_valid_lm_loss_mask.sum().clamp(min=1e-9)
    #     else:
    #         # 如果没有任何有效 token 参与 LM Loss 计算，则 lm_loss_final 保持 0.0
    #         pass


    #     # 3. 识别 latent token 的位置并收集其隐藏状态
    #     latent_hidden_states = []
    #     # 注意: 这里的 input_ids 仍然是原始的，未 shifted 的
    #     for b in range(input_ids.shape[0]):
    #         for i in range(input_ids.shape[1]):
    #             token_id = input_ids[b, i].item()
    #             if token_id == self.latent_token_id and self.latent_token_id != -1: # 确保 latent_token_id 有效
    #                 latent_hidden_states.append(hidden_states[b, i, :])

    #     # 4. 计算 COCONUT 的对比损失
    #     unlearning_contrastive_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    #     if self.forget_concept_vector is not None and len(latent_hidden_states) > 0:
    #         latent_hidden_states_tensor = torch.stack(latent_hidden_states) # (N_latent_tokens, hidden_dim)

    #         # 对比损失：最小化 latent 隐藏状态与遗忘概念向量的相似度
    #         forget_vector_normalized = F.normalize(self.forget_concept_vector, dim=-1) # (hidden_dim,)
    #         latent_normalized = F.normalize(latent_hidden_states_tensor, dim=-1) # (N_latent_tokens, hidden_dim)

    #         # 计算余弦相似度
    #         similarities = torch.mm(latent_normalized, forget_vector_normalized.unsqueeze(-1)).squeeze(-1) # (N_latent_tokens,)

    #         # MSE loss：让相似度尽可能接近 0 (即正交)
    #         unlearning_contrastive_loss = F.mse_loss(similarities, torch.zeros_like(similarities))

    #     # 5. 组合总损失
    #     total_loss = lm_loss_final + self.unlearning_loss_weight * unlearning_contrastive_loss

    #     # 返回与父类 `Qwen2_5_VLCausalLMOutputWithPast` 兼容的输出格式
    #     return Qwen2_5_VLCausalLMOutputWithPast(
    #         loss=total_loss,
    #         logits=logits,
    #         past_key_values=base_outputs.past_key_values,
    #         hidden_states=base_outputs.hidden_states,
    #         attentions=base_outputs.attentions,
    #         rope_deltas=self.rope_deltas,
    #     )
    
    
    # def forward(
    #     self,
    #     input_ids: Optional[torch.LongTensor] = None,
    #     pixel_values: Optional[torch.FloatTensor] = None,
    #     image_grid_thw: Optional[Tuple[int, int, int]] = None,
    #     video_grid_thw: Optional[Tuple[int, int, int]] = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     position_ids: Optional[torch.LongTensor] = None,
    #     past_key_values: Optional[list[torch.FloatTensor]] = None,
    #     inputs_embeds: Optional[torch.FloatTensor] = None,
    #     labels: Optional[torch.LongTensor] = None,
    #     use_cache: Optional[bool] = None,
    #     output_attentions: Optional[bool] = None,
    #     output_hidden_states: Optional[bool] = None,
    #     return_dict: Optional[bool] = None,
    #     cache_position: Optional[torch.LongTensor] = None,
    #     unlearn_mask: Optional[torch.BoolTensor] = None,
    #     **kwargs
    # ) -> Qwen2_5_VLCausalLMOutputWithPast:
    #     """
    #     完整的 COCONUT Unlearning Forward 实现
    #     1. 自动识别 Latent Tokens 并提取隐藏状态
    #     2. 计算隐式推理层与遗忘概念的正交化损失 (Contrastive Loss)
    #     3. 针对 Answer 层进行精准的梯度上升 (Gradient Ascent)
    #     4. 正常学习非遗忘部分的知识
    #     """
    #     # --- 调试打印开始 ---
    #     # print(f"\n[DEBUG] input_ids shape: {input_ids.shape}")
    #     # # 打印第一条数据的 token IDs (前 100 个和后 100 个，防止太长)
    #     # sample_input = input_ids[0].tolist()
    #     # print(f"[DEBUG] First sample input_ids: {sample_input}")

    #     # if self.end_latent_id in sample_input:
    #     #     end_idx = sample_input.index(self.end_latent_id)
    #     #     print(f"[DEBUG] Found <latent end> (ID: {self.end_latent_id}) at index: {end_idx}")
    #     #     # 打印 latent end 之后的一些 token 看看是不是 answer
    #     #     next_tokens = sample_input[end_idx+1 : end_idx+10]
    #     #     print(f"[DEBUG] Tokens after <latent end>: {next_tokens}")
    #     #     if self.tokenizer:
    #     #         print(f"[DEBUG] Decoded answer start: {self.tokenizer.decode(next_tokens)}")
    #     # else:
    #     #     print(f"[DEBUG] <latent end> (ID: {self.end_latent_id}) NOT FOUND in input_ids!")

    #     # # 顺便看看 labels
    #     # if labels is not None:
    #     #     sample_labels = labels[0].tolist()
    #     #     # 找到第一个不是 -100 的位置
    #     #     first_label_idx = next((i for i, x in enumerate(sample_labels) if x != -100), None)
    #     #     print(f"[DEBUG] First valid label index: {first_label_idx}")
    #     # # --- 调试打印结束 -
    #     # pdb.set_trace()
        
    #     output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    #     output_hidden_states = True  # 强制开启，因为我们需要分析隐藏状态
    #     return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    #     # ==========================================================
    #     # 1. 执行基础的前向传播 (一次性获取所有状态，保证效率)
    #     # ==========================================================
    #     base_outputs = super().forward(
    #         input_ids=input_ids,
    #         pixel_values=pixel_values,
    #         image_grid_thw=image_grid_thw,
    #         video_grid_thw=video_grid_thw,
    #         attention_mask=attention_mask,
    #         position_ids=position_ids,
    #         past_key_values=past_key_values,
    #         inputs_embeds=inputs_embeds,
    #         labels=None, # 我们手动计算损失，不让基类计算
    #         use_cache=use_cache,
    #         output_attentions=output_attentions,
    #         output_hidden_states=output_hidden_states,
    #         return_dict=True,
    #         cache_position=cache_position,
    #         **kwargs
    #     )
    #     # pdb.set_trace()
    #     logits = base_outputs.logits
    #     # 取最后一层的隐藏状态: (Batch, Seq_Len, Hidden_Dim)
    #     last_hidden_state = base_outputs.hidden_states[-1] 

    #     if labels is None:
    #         return base_outputs

    #     # ==========================================================
    #     # 2. 语言模型损失计算 (LM Loss) - 包含梯度上升逻辑
    #     # ==========================================================
    #     # Shift 操作以对齐输出与目标
    #     shift_logits = logits[..., :-1, :].contiguous()
    #     shift_labels = labels[..., 1:].contiguous()
        
    #     flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    #     flat_labels = shift_labels.view(-1)

    #     # 基础交叉熵损失 (不进行平均)
    #     per_token_loss = F.cross_entropy(flat_logits, flat_labels, reduction='none', ignore_index=-100)
        
    #     # 准备掩码
    #     valid_loss_mask = (flat_labels != -100)
        
    #     # 排除 COCONUT 特殊 Token (这些 Token 不参与常规文本 Loss 计算)
    #     if self.latent_token_id != -1:
    #         valid_loss_mask &= (flat_labels != self.latent_token_id)
    #     if self.start_latent_id != -1:
    #         valid_loss_mask &= (flat_labels != self.start_latent_id)
    #     if self.end_latent_id != -1:
    #         valid_loss_mask &= (flat_labels != self.end_latent_id)

    #     lm_loss_final = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)



    #     if unlearn_mask is None:
    #         raise ValueError("严重错误：unlearn_mask 为空！梯度上升逻辑未激活！")
    #     else:
    #         print(f"DEBUG: unlearn_mask shape: {unlearn_mask.shape}, sum: {unlearn_mask.sum()}")        
    #     if torch.any(valid_loss_mask):
    #         if unlearn_mask is not None:
    #             # 处理 unlearn_mask 的偏移和展平
    #             flat_unlearn_mask = unlearn_mask[..., 1:].contiguous().view(-1)
                
    #             # 分离需要遗忘的和需要保留的
    #             unlearn_part = flat_unlearn_mask & valid_loss_mask
    #             learn_part = (~flat_unlearn_mask) & valid_loss_mask
                
    #             # --- 正常学习 (梯度下降) ---
    #             if torch.any(learn_part):
    #                 lm_loss_final += per_token_loss[learn_part].mean()
                
    #             # --- 精准遗忘 (梯度上升) ---
    #             if torch.any(unlearn_part):
    #                 unlearn_loss = per_token_loss[unlearn_part].mean()
    #                 # 关键：添加截断防止梯度爆炸。当 Loss 很大时（意味着已经忘掉了），不再强制增加 Loss
    #                 # unlearn_loss_clamped = torch.clamp(unlearn_loss, max=8.0) 
    #                 lm_loss_final -= (unlearn_loss * self.unlearn_lm_loss_weight)
    #         else:
    #             # 普通训练模式
    #             lm_loss_final = per_token_loss[valid_loss_mask].mean()

    #     # ==========================================================
    #     # 3. 隐式推理对比损失 (Contrastive Unlearning Loss)
    #     # ==========================================================
    #     unlearning_contrastive_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    #     if self.forget_concept_vector is not None and self.latent_token_id != -1:
    #         # 找到所有 Latent Token 在 input_ids 中的位置
    #         # 注意：我们要看的是 input_ids 产生 hidden_state 的位置
    #         latent_pos_mask = (input_ids == self.latent_token_id)
            
    #         # 提取对应的隐藏状态 (Num_Latents, Hidden_Dim)
    #         latent_states = last_hidden_state[latent_pos_mask]
            
    #         if latent_states.shape[0] > 0:
    #             # 确保概念向量与状态在同一设备和精度
    #             concept_vec = self.forget_concept_vector.to(device=latent_states.device, dtype=latent_states.dtype)
                
    #             # 归一化以计算余弦相似度
    #             latent_norm = F.normalize(latent_states, p=2, dim=-1)
    #             concept_norm = F.normalize(concept_vec, p=2, dim=-1)
                
    #             # 计算每个 latent token 状态与遗忘概念的相似度
    #             # (Num_Latents, Hidden_Dim) * (Hidden_Dim, 1) -> (Num_Latents,)
    #             similarities = torch.mm(latent_norm, concept_norm.unsqueeze(-1)).squeeze(-1)
                
    #             # 目标是相似度为 0 (正交)，使用 MSE 引导
    #             unlearning_contrastive_loss = F.mse_loss(similarities, torch.zeros_like(similarities))

    #     # ==========================================================
    #     # 4. 最终损失汇总
    #     # ==========================================================
    #     print('ga loss', lm_loss_final)
    #     print('contrastive loss', unlearning_contrastive_loss)
    #     total_loss = lm_loss_final + (self.unlearning_loss_weight * unlearning_contrastive_loss)

    #     # 兼容性包装返回
    #     return Qwen2_5_VLCausalLMOutputWithPast(
    #         loss=total_loss,
    #         logits=logits,
    #         past_key_values=base_outputs.past_key_values,
    #         hidden_states=base_outputs.hidden_states,
    #         attentions=base_outputs.attentions,
    #         rope_deltas=kwargs.get("rope_deltas", None),
    #     )


    # def forward(
    #     self,
    #     input_ids: Optional[torch.LongTensor] = None,
    #     pixel_values: Optional[torch.FloatTensor] = None,
    #     image_grid_thw: Optional[Tuple[int, int, int]] = None,
    #     video_grid_thw: Optional[Tuple[int, int, int]] = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     position_ids: Optional[torch.LongTensor] = None,
    #     past_key_values: Optional[list[torch.FloatTensor]] = None,
    #     inputs_embeds: Optional[torch.FloatTensor] = None,
    #     labels: Optional[torch.LongTensor] = None,
    #     use_cache: Optional[bool] = None,
    #     output_attentions: Optional[bool] = None,
    #     output_hidden_states: Optional[bool] = None,
    #     return_dict: Optional[bool] = None,
    #     cache_position: Optional[torch.LongTensor] = None,
    #     **kwargs
    # ) -> Qwen2_5_VLCausalLMOutputWithPast:
    #     """
    #     COCONUT Unlearning Forward 实现
    #     1. 动态定位 <latent end> 之后的 Answer 部分
    #     2. 对 Answer 执行梯度上升 (GA)，对之前部分执行正常学习
    #     3. 对 Latent Tokens 提取隐藏状态并执行正交化 Contrastive Loss
    #     """

    #     output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    #     # 强制开启 hidden_states，因为对比损失需要它
    #     output_hidden_states = True 
    #     return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    #     # ==========================================================
    #     # 1. 执行基础前向传播
    #     # ==========================================================
    #     base_outputs = super().forward(
    #         input_ids=input_ids,
    #         pixel_values=pixel_values,
    #         image_grid_thw=image_grid_thw,
    #         video_grid_thw=video_grid_thw,
    #         attention_mask=attention_mask,
    #         position_ids=position_ids,
    #         past_key_values=past_key_values,
    #         inputs_embeds=inputs_embeds,
    #         labels=None, # 手动计算 Loss
    #         use_cache=use_cache,
    #         output_attentions=output_attentions,
    #         output_hidden_states=output_hidden_states,
    #         return_dict=True,
    #         cache_position=cache_position,
    #         **kwargs
    #     )

    #     logits = base_outputs.logits
    #     last_hidden_state = base_outputs.hidden_states[-1] 

    #     if labels is None:
    #         return base_outputs

    #     # ==========================================================
    #     # 2. 动态生成 Unlearn Mask (识别 Answer 区域)
    #     # ==========================================================
    #     # 逻辑：在每一行中找到第一个 <latent end> 出现的位置，之后的所有 Token 视为 Answer
    #     if self.end_latent_id != -1:
    #         # is_end_token: (Batch, Seq) -> True 在 <latent end> 的位置
    #         is_end_token = (input_ids == self.end_latent_id)
    #         # 使用 cumsum，第一个 <latent end> 之后的所有位置累加值都会 > 0
    #         # 我们从 <latent end> 的下一个位置开始算 GA，所以减去当前位置
    #         dynamic_unlearn_mask = (is_end_token.cumsum(dim=1) - is_end_token.long()) > 0
    #     else:
    #         # 如果没定义 end_id，则默认全不遗忘（仅作为 fallback）
    #         dynamic_unlearn_mask = torch.zeros_like(input_ids, dtype=torch.bool)

    #     # ==========================================================
    #     # 3. 语言模型损失计算 (LM Loss)
    #     # ==========================================================
    #     # Shift 操作对齐 (预测下一个 token)
    #     shift_logits = logits[..., :-1, :].contiguous()
    #     shift_labels = labels[..., 1:].contiguous()
    #     shift_unlearn_mask = dynamic_unlearn_mask[..., 1:].contiguous()
        
    #     flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    #     flat_labels = shift_labels.view(-1)
    #     flat_unlearn_mask = shift_unlearn_mask.view(-1)

    #     # 基础交叉熵损失
    #     per_token_loss = F.cross_entropy(flat_logits, flat_labels, reduction='none', ignore_index=-100)
    #     valid_loss_mask = (flat_labels != -100)
        
    #     # 排除 COCONUT 特殊 Token 不参与常规文本 Loss
    #     if self.latent_token_id != -1:
    #         valid_loss_mask &= (flat_labels != self.latent_token_id)
    #     if self.start_latent_id != -1:
    #         valid_loss_mask &= (flat_labels != self.start_latent_id)
    #     if self.end_latent_id != -1:
    #         valid_loss_mask &= (flat_labels != self.end_latent_id)

    #     lm_loss_final = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        
    #     # 分离需要遗忘的和需要保留的
    #     unlearn_part_mask = flat_unlearn_mask & valid_loss_mask
    #     learn_part_mask = (~flat_unlearn_mask) & valid_loss_mask
        
    #     # --- 正常学习 (梯度下降: Loss 为正) ---
    #     if torch.any(learn_part_mask):
    #         learn_loss = per_token_loss[learn_part_mask].mean()
    #         lm_loss_final += learn_loss
        
    #     # --- 精准遗忘 (梯度上升: Loss 取负) ---
    #     ga_loss_value = 0.0
    #     if torch.any(unlearn_part_mask):
    #         unlearn_loss = per_token_loss[unlearn_part_mask].mean()
    #         # 这里的减号是关键，实现梯度上升
    #         ga_loss_value = -(unlearn_loss * self.unlearn_lm_loss_weight)
    #         lm_loss_final += ga_loss_value

    #     # ==========================================================
    #     # 4. 隐式推理对比损失 (Contrastive Unlearning Loss)
    #     # ==========================================================
    #     unlearning_contrastive_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    #     if self.forget_concept_vector is not None and self.latent_token_id != -1:
    #         # 找到 latent tokens 的位置
    #         latent_pos_mask = (input_ids == self.latent_token_id)
    #         latent_states = last_hidden_state[latent_pos_mask]
            
    #         if latent_states.shape[0] > 0:
    #             concept_vec = self.forget_concept_vector.to(device=latent_states.device, dtype=latent_states.dtype)
    #             latent_norm = F.normalize(latent_states, p=2, dim=-1)
    #             concept_norm = F.normalize(concept_vec, p=2, dim=-1)
    #             # 计算余弦相似度并推向 0 (正交)
    #             similarities = torch.mm(latent_norm, concept_norm.unsqueeze(-1)).squeeze(-1)
    #             unlearning_contrastive_loss = F.mse_loss(similarities, torch.zeros_like(similarities))

    #     # ==========================================================
    #     # 5. 最终汇总与打印
    #     # ==========================================================
    #     # total_loss = lm_loss_final + (self.unlearning_loss_weight * unlearning_contrastive_loss)
    #     total_loss = lm_loss_final
    #     if self.training:
    #         # 打印监控，确认 ga_loss 是否为负数
    #         print(f">>> [LOSS] Total: {total_loss.item():.6f} | "
    #               f"GA Part: {ga_loss_value if isinstance(ga_loss_value, float) else ga_loss_value.item():.6f} | "
    #               f"Contrastive: {unlearning_contrastive_loss.item():.6f}")

    #     return Qwen2_5_VLCausalLMOutputWithPast(
    #         loss=total_loss,
    #         logits=logits,
    #         past_key_values=base_outputs.past_key_values,
    #         hidden_states=base_outputs.hidden_states,
    #         attentions=base_outputs.attentions,
    #         rope_deltas=kwargs.get("rope_deltas", None),
    #     )






    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[Tuple[int, int, int]] = None,
        video_grid_thw: Optional[Tuple[int, int, int]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Qwen2_5_VLCausalLMOutputWithPast:

        if labels is None or not self.training:
            return super().forward(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
                **kwargs
            )

        # 1. 预计算全序列 RoPE
        full_position_ids, full_rope_deltas = self.get_rope_index(
            input_ids,
            image_grid_thw,
            video_grid_thw,
            attention_mask,
        )

        kwargs.pop("rope_deltas", None)
        kwargs.pop("position_ids", None)
        # 强制在子调用中允许使用 cache (如果可能)，或者显式处理 None
        # 注意：在 Gradient Checkpointing 开启时，此处传递 use_cache 会被 transformers 忽略
        
        batch_size = input_ids.shape[0]
        idx_start = (input_ids[0] == self.start_latent_id).nonzero(as_tuple=True)[0].item()
        idx_end = (input_ids[0] == self.end_latent_id).nonzero(as_tuple=True)[0].item()
        num_latent_steps = idx_end - idx_start - 1

        # ==========================================================
        # Phase A: Prefix
        # ==========================================================
        prefix_ids = input_ids[:, :idx_start + 1]
        
        outputs = super().forward(
            input_ids=prefix_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache, # 遵循外部设置
            output_hidden_states=True,
            return_dict=True,
            **kwargs
        )
        
        all_logits = [outputs.logits]
        current_hidden_state = outputs.hidden_states[-1][:, -1:, :] 
        all_step_hidden_states = [current_hidden_state]
        
        # --- 核心修复：不再依赖 past_key_values 获取长度 ---
        # 图像展开后的真实长度就在 logits 的第二维
        current_offset = outputs.logits.shape[1] 
        current_pkvs = outputs.past_key_values # 可能是 None

        # ==========================================================
        # Phase B: Latent Steps
        # ==========================================================
        latent_logits = []
        
        # 如果开启了梯度检查点，current_pkvs 为 None
        # 我们需要累积所有的 inputs_embeds 以便在没有 cache 的情况下前向传播
        # 但为了简化且考虑到推理逻辑，我们假设在 COCONUT 训练中应尽量关闭 GC 
        # 或者我们在这里手动处理。
        
        # 如果 current_pkvs 是 None，说明无法增量推理，
        # 我们需要拼接之前的 hidden states。但为了训练效率，
        # 建议在 yaml 配置中设置 gradient_checkpointing: false
        
        for i in range(num_latent_steps):
            step_pos_ids = full_position_ids[:, :, current_offset : current_offset + 1]
            
            step_outputs = super().forward(
                input_ids=None,
                inputs_embeds=current_hidden_state,
                position_ids=step_pos_ids,
                rope_deltas=full_rope_deltas,
                past_key_values=current_pkvs,
                use_cache=use_cache,
                output_hidden_states=True,
                return_dict=True,
                **kwargs
            )
            
            current_hidden_state = step_outputs.hidden_states[-1]
            current_pkvs = step_outputs.past_key_values
            
            latent_logits.append(step_outputs.logits)
            all_step_hidden_states.append(current_hidden_state)
            current_offset += 1

        # ==========================================================
        # Phase C: Suffix
        # ==========================================================
        suffix_ids = input_ids[:, idx_end:]
        suffix_pos_ids = full_position_ids[:, :, current_offset:]
        suffix_embeds = self.get_input_embeddings()(suffix_ids)
        suffix_inputs = torch.cat([current_hidden_state, suffix_embeds[:, 1:, :]], dim=1)
        
        suffix_outputs = super().forward(
            input_ids=None,
            inputs_embeds=suffix_inputs,
            position_ids=suffix_pos_ids,
            rope_deltas=full_rope_deltas,
            past_key_values=current_pkvs,
            use_cache=use_cache,
            output_hidden_states=True,
            return_dict=True,
            **kwargs
        )
        
        all_logits.append(suffix_outputs.logits)

        # ==========================================================
        # 结果拼接与对齐
        # ==========================================================
        combined_logits = torch.cat([all_logits[0]] + latent_logits + [all_logits[-1]], dim=1)
        
        prefix_len = all_logits[0].shape[1]
        new_labels_prefix = torch.full((batch_size, prefix_len), -100, device=labels.device)
        new_labels_latent = torch.full((batch_size, num_latent_steps), -100, device=labels.device)
        new_labels_suffix = labels[:, idx_end:] 
        
        aligned_labels = torch.cat([new_labels_prefix, new_labels_latent, new_labels_suffix], dim=1)

        # Loss 计算
        shift_logits = combined_logits[..., :-1, :].contiguous()
        shift_labels = aligned_labels[..., 1:].contiguous()
        
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        
        per_token_loss = F.cross_entropy(flat_logits, flat_labels, reduction='none', ignore_index=-100)
        total_loss = torch.tensor(0.0, device=combined_logits.device)

        # GA Loss
        ga_mask = (flat_labels != -100)
        ga_loss = torch.tensor(0.0, device=combined_logits.device)
        if torch.any(ga_mask):
            ga_loss = per_token_loss[ga_mask].mean()
            total_loss = -(ga_loss * self.unlearn_lm_loss_weight)

        # Contrastive Loss
        latent_states = torch.cat(all_step_hidden_states, dim=1)
        unlearning_contrastive_loss = torch.tensor(0.0, device=combined_logits.device)
        if self.forget_concept_vector is not None:
            concept_vec = self.forget_concept_vector.to(latent_states.device)
            latent_norm = F.normalize(latent_states, p=2, dim=-1)
            concept_norm = F.normalize(concept_vec, p=2, dim=-1)
            cos_sim = (latent_norm * concept_norm).sum(dim=-1).mean()
            # 使用绝对值，让latent states远离concept vector（无论正相关还是负相关）
            unlearning_contrastive_loss = torch.abs(cos_sim)
            total_loss += (self.unlearning_loss_weight * unlearning_contrastive_loss)

        if self.training:
            print(f">>> GA_Loss: {-ga_loss.item():.4f} | Cos_Sim: {unlearning_contrastive_loss.item():.4f}")

        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=total_loss,
            logits=combined_logits,
            past_key_values=suffix_outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )

        
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        image_grid_thw: torch.LongTensor = None,
        max_new_tokens: int = 512,
        num_latent_tokens: int = 4,
        **kwargs
    ):
        device = self.device
        
        # 0. 基础准备
        if getattr(self, "start_latent_id", -1) == -1:
            self.latent_token_id = 151665
            self.start_latent_id = 151666
            self.end_latent_id = 151667
        
        # 1. 预填充 (Prefill)
        attention_mask = kwargs.pop("attention_mask", torch.ones_like(input_ids, device=device))
        outputs = super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
            output_hidden_states=True
        )
        
        past_key_values = outputs.past_key_values
        current_hidden_state = outputs.hidden_states[-1][:, -1:, :]
        
        # 初始化位置信息 (mROPE)
        p_ids, _ = self.get_rope_index(input_ids, image_grid_thw, attention_mask=attention_mask)
        current_p_id = p_ids[:, :, -1:].to(device)
        current_len = input_ids.shape[1]

        # 2. 隐空间迭代 (Coconut 循环) - 方案A: 记录latent token IDs
        start_embed = self.get_input_embeddings()(torch.tensor([[self.start_latent_id]], device=device))
        end_embed = self.get_input_embeddings()(torch.tensor([[self.end_latent_id]], device=device))

        # 记录latent tokens，用于最终输出（方案A关键修改）
        latent_token_ids = [self.start_latent_id]

        # 序列: <start> -> num_latent 次隐变量 -> <end>
        latent_inputs = [start_embed] + [None] * num_latent_tokens + [end_embed]

        for step_idx, step_embed in enumerate(latent_inputs):
            current_p_id = current_p_id.clone()
            current_p_id[:, 0, :] += 1 # 递增 T 维
            cache_pos = torch.tensor([current_len], device=device)
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=device, dtype=attention_mask.dtype)], dim=-1)

            # 使用上一步的 hidden_state 或当前的特殊 embedding
            actual_input = step_embed if step_embed is not None else current_hidden_state

            model_out = self.model(
                inputs_embeds=actual_input,
                past_key_values=past_key_values,
                position_ids=current_p_id,
                cache_position=cache_pos,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True
            )
            past_key_values = model_out.past_key_values
            current_hidden_state = model_out.last_hidden_state
            current_len += 1

            # 记录latent token IDs（中间步骤使用latent_token_id）
            if step_embed is None and step_idx < len(latent_inputs) - 1:
                latent_token_ids.append(self.latent_token_id)
            elif step_idx == len(latent_inputs) - 1:
                latent_token_ids.append(self.end_latent_id)

        # 3. 手动文本解码循环 (取代 super().generate)
        generated_tokens = []
        # 获取停止符号 ID (Qwen 通常是 151643 或 151645)
        eos_token_id = self.config.eos_token_id if isinstance(self.config.eos_token_id, int) else self.config.eos_token_id[0]

        for i in range(max_new_tokens):
            # 预测下一个词
            logits = self.lm_head(current_hidden_state)


            probs = torch.softmax(logits[:, -1, :], dim=-1)
            top_val, top_idx = torch.topk(probs, 5) # 看看概率最高的前5个词

            if i == 0:
                print(f"[DEBUG] First Token Candidates: {top_idx[0].tolist()}")
                print(f"[DEBUG] First Token Probs: {top_val[0].tolist()}")

            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

            token_val = next_token.item()
            if token_val == eos_token_id:
                break
            generated_tokens.append(token_val)

            # 为下一轮准备输入
            current_p_id = current_p_id.clone()
            current_p_id[:, 0, :] += 1
            cache_pos = torch.tensor([current_len], device=device)
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=device, dtype=attention_mask.dtype)], dim=-1)

            model_out = self.model(
                input_ids=next_token, # 文本阶段直接传 ID
                past_key_values=past_key_values,
                position_ids=current_p_id,
                cache_position=cache_pos,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True
            )
            past_key_values = model_out.past_key_values
            current_hidden_state = model_out.last_hidden_state
            current_len += 1

        # 4. 构造返回格式（方案A：返回 input_ids + latent_tokens + generated_tokens）
        if not generated_tokens:
            print("[WARNING] No tokens were generated! Returning input with latent tokens.")
            generated_tokens = []

        # 拼接完整序列：input_ids + latent_token_ids + generated_tokens
        latent_tokens_tensor = torch.tensor(latent_token_ids, device=device, dtype=input_ids.dtype).unsqueeze(0)
        generated_tokens_tensor = torch.tensor(generated_tokens, device=device, dtype=input_ids.dtype).unsqueeze(0)

        full_sequence = torch.cat([input_ids, latent_tokens_tensor, generated_tokens_tensor], dim=1)

        print(f"[DEBUG] Solution A: input {input_ids.shape[1]} + latent {len(latent_token_ids)} + generated {len(generated_tokens)} = {full_sequence.shape[1]}")
        return full_sequence

        # 拼接输入和生成的tokens，返回完整序列
        generated_tokens_tensor = torch.tensor(generated_tokens, device=device, dtype=input_ids.dtype).unsqueeze(0)
        full_sequence = torch.cat([input_ids, generated_tokens_tensor], dim=1)

        print(f"[DEBUG] Returning full sequence: input {input_ids.shape[1]} + generated {len(generated_tokens)} = {full_sequence.shape[1]}")
        return full_sequence





    def _extend_pids(self, p_ids):
        """ 将 3D 位置编码向后扩展一个时间步 """
        # p_ids 形状为 (1, 3, seq_len)
        # 我们取最后一个时间步的位置，并让 T 维度 (索引 0) + 1
        new_p_id = p_ids[:, :, -1:].clone()
        new_p_id[:, 0, :] += 1 
        return torch.cat([p_ids, new_p_id], dim=2)


    def get_rope_index(self, *args, **kwargs):
        """ 
        使用 *args 和 **kwargs 转发所有参数，
        确保兼容官方 forward 调用的 5 个参数。
        """
        return super().get_rope_index(*args, **kwargs)




    def visual_context_embed(self, input_ids, pixel_values, image_grid_thw):
        """ 
        正确地将文本和图像合并成 initial inputs_embeds
        针对 Qwen2_5_VL 的结构进行了优化
        """
        # 1. 使用外层包装类（self）而不是 self.model
        # 2. 我们通过一次不计算 Loss 的 forward 获取初始的 hidden_states
        # Qwen2.5-VL 的 inputs_embeds 是在 forward 的最开始阶段生成的
        
        # 这里的 kwargs 需要包含所有 Qwen2.5-VL 需要的输入
        outputs = super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            return_dict=True
        )
        
        # hidden_states[0] 通常是输入层合并了 vision embedding 后的结果
        return outputs.hidden_states[0]
