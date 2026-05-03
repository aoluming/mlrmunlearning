"""
COCOUN Trajectory Model: Trajectory-Level Unlearning with Logit-Lens Critic and Redirector

核心创新：
1. Logit-Lens Critic: 在每个latent step诊断风险
2. Latent Redirector: 动态重写高风险latent states
3. Sequence-Level Reward: 可选的RL优化

使用方法：
在yaml中设置: stage: cocoun_trajectory
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, List
from transformers import Qwen2_5_VLForConditionalGeneration, PreTrainedTokenizer
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast

# 导入原有的cocoun_model作为基础
from .cocoun_model import Cocoun_model


class LogitLensCritic(nn.Module):
    """
    Logit-Lens Critic: 诊断每个latent step的风险

    在每个latent step将hidden state解码到vocab空间，
    检查是否产生被禁止概念的token
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        forbidden_token_ids: List[int],
        risk_threshold: float = 0.1,
    ):
        super().__init__()
        self.forbidden_token_ids = set(forbidden_token_ids)
        self.risk_threshold = risk_threshold

        # 轻量级投影层：hidden_state -> logits
        self.logit_projection = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        latent_states: torch.Tensor,  # (B, L, D) 或 (B, D)
        return_top_risks: bool = True,
        top_k: int = 5
    ) -> Dict[str, torch.Tensor]:
        """
        评估latent states的风险

        Returns:
            risk_scores: (B, L) 每个latent state的风险分数
            forbidden_probs: (B, L, num_forbidden) 禁止token的概率
        """
        is_single_step = latent_states.dim() == 2
        if is_single_step:
            latent_states = latent_states.unsqueeze(1)

        batch_size, num_steps, hidden_dim = latent_states.shape

        # 投影到logits空间
        logits = self.logit_projection(latent_states)  # (B, L, vocab_size)

        # 计算softmax概率
        probs = F.softmax(logits, dim=-1)

        # 提取禁止token的概率
        forbidden_indices = list(self.forbidden_token_ids)

        # 如果没有禁止token，返回零风险
        if len(forbidden_indices) == 0:
            device = latent_states.device
            results = {
                'risk_scores': torch.zeros(batch_size, num_steps, device=device),
                'forbidden_probs': torch.zeros(batch_size, num_steps, 0, device=device),
                'high_risk_mask': torch.zeros(batch_size, num_steps, dtype=torch.bool, device=device),
                'num_high_risk': torch.zeros(batch_size, dtype=torch.long, device=device),
            }
            if is_single_step:
                for key in results:
                    if isinstance(results[key], torch.Tensor):
                        results[key] = results[key].squeeze(1)
            return results

        forbidden_probs = probs[..., forbidden_indices]  # (B, L, num_forbidden)

        # 风险分数 = 禁止token的最大概率
        risk_scores = forbidden_probs.max(dim=-1)[0]  # (B, L)

        # 统计超过阈值的latent steps
        high_risk_mask = risk_scores > self.risk_threshold

        results = {
            'risk_scores': risk_scores,
            'forbidden_probs': forbidden_probs,
            'high_risk_mask': high_risk_mask,
            'num_high_risk': high_risk_mask.sum(dim=-1),
        }

        if return_top_risks:
            # 返回每个step top-k高风险token
            top_risk_probs, top_risk_indices = forbidden_probs.topk(
                min(top_k, len(forbidden_indices)), dim=-1
            )
            results['top_risk_probs'] = top_risk_probs
            results['top_risk_indices'] = top_risk_indices

        if is_single_step:
            for key in results:
                if results[key] is not None and isinstance(results[key], torch.Tensor):
                    results[key] = results[key].squeeze(1)

        return results


class LatentRedirector(nn.Module):
    """
    轻量级Latent Redirector: 将高风险latent states重定向到中性轨迹

    策略：
    1. 检测风险：使用Critic的输出
    2. 学习重定向变换
    3. 动态重写高风险states
    """

    def __init__(
        self,
        hidden_size: int,
        num_latent_steps: int,
        redirect_strength: float = 0.5,
        use_mlp_redirector: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_latent_steps = num_latent_steps
        self.redirect_strength = redirect_strength
        self.use_mlp_redirector = use_mlp_redirector

        if use_mlp_redirector:
            # 学习一个重定向MLP
            self.redirect_mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.GELU(),
                nn.Linear(hidden_size * 2, hidden_size),
            )
        else:
            # 每个step特定的redirector
            self.step_specific_redirectors = nn.ModuleList([
                nn.Linear(hidden_size, hidden_size)
                for _ in range(num_latent_steps)
            ])

    def forward(
        self,
        latent_states: torch.Tensor,  # (B, L, D)
        high_risk_mask: torch.Tensor,  # (B, L)
    ) -> Tuple[torch.Tensor, Dict]:
        """
        重定向高风险latent states

        Returns:
            redirected_states: (B, L, D)
            redirect_info: 重定向的统计信息
        """
        batch_size, num_steps, hidden_dim = latent_states.shape

        # 初始化输出
        redirected_states = latent_states.clone()

        # 如果没有高风险，直接返回
        if not high_risk_mask.any():
            return redirected_states, {'num_redirected': 0}

        # 提取高风险的states
        high_risk_indices = high_risk_mask.nonzero(as_tuple=True)

        if self.use_mlp_redirector:
            # 使用MLP重定向
            high_risk_states = latent_states[high_risk_indices]  # (N, D)

            # 计算重定向方向
            with torch.cuda.amp.autocast(enabled=False):
                redirect_delta = self.redirect_mlp(high_risk_states.float())
                redirected_states[high_risk_indices] = (
                    high_risk_states +
                    redirect_delta * self.redirect_strength
                )
        else:
            # Step-specific重定向
            for step_idx in range(num_steps):
                step_mask = high_risk_mask[:, step_idx]
                if step_mask.any():
                    step_states = latent_states[:, step_idx][step_mask]
                    redirect_delta = self.step_specific_redirectors[step_idx](step_states)
                    redirected_states[:, step_idx][step_mask] = (
                        step_states + redirect_delta * self.redirect_strength
                    )

        # 统计信息
        redirect_info = {
            'num_redirected': high_risk_mask.sum().item(),
            'redirect_ratio': high_risk_mask.sum().item() / high_risk_mask.numel(),
        }

        return redirected_states, redirect_info


class Cocoun_trajectory_model(Cocoun_model):
    """
    COCONUT Trajectory Model: 扩展cocoun_model，增加轨迹级遗忘

    新增功能：
    1. Logit-Lens Critic: 监控每个latent step的风险
    2. Latent Redirector: 动态重写高风险states
    3. Trajectory Loss: 确保整条轨迹的安全性
    """

    def __init__(
        self,
        config,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        latent_token_id: Optional[int] = None,
        start_latent_id: Optional[int] = None,
        end_latent_id: Optional[int] = None,
        forget_concept_vector: Optional[torch.Tensor] = None,
        unlearning_loss_weight: float = 1.0,
        unlearn_lm_loss_weight: float = 20,
        # 新增参数
        forbidden_token_ids: Optional[List[int]] = None,
        risk_threshold: float = 0.1,
        redirect_strength: float = 0.5,
        use_trajectory_unlearning: bool = True,
    ):
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            latent_token_id=latent_token_id,
            start_latent_id=start_latent_id,
            end_latent_id=end_latent_id,
            forget_concept_vector=forget_concept_vector,
            unlearning_loss_weight=unlearning_loss_weight,
            unlearn_lm_loss_weight=unlearn_lm_loss_weight,
        )

        self.use_trajectory_unlearning = use_trajectory_unlearning
        self.risk_threshold = risk_threshold
        self.redirect_strength = redirect_strength
        self.forbidden_token_ids = forbidden_token_ids if forbidden_token_ids is not None else []

        # 延迟初始化：不在__init__中创建critic和redirector，而是在第一次forward时创建
        self.critic = None
        self.redirector = None

        if use_trajectory_unlearning:
            print(f"[TRAJECTORY] Trajectory unlearning enabled with "
                  f"risk_threshold={risk_threshold}, redirect_strength={redirect_strength}")

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
        """
        扩展的forward方法，集成trajectory unlearning
        """

        if labels is None or not self.training:
            # 推理模式：直接调用父类
            return super().forward(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
                **kwargs
            )

        if not self.use_trajectory_unlearning:
            # 如果不使用trajectory unlearning，使用原始的cocoun forward
            return super().forward(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                **kwargs
            )

        # ==========================================================
        # 延迟初始化critic和redirector（避免from_pretrained时的段错误）
        # ==========================================================
        if self.critic is None:
            device = next(self.parameters()).device
            self.critic = LogitLensCritic(
                hidden_size=self.config.hidden_size,
                vocab_size=self.config.vocab_size,
                forbidden_token_ids=self.forbidden_token_ids,
                risk_threshold=self.risk_threshold,
            ).to(device)
            print(f"[TRAJECTORY] Critic initialized on device: {device}")

        if self.redirector is None:
            device = next(self.parameters()).device
            self.redirector = LatentRedirector(
                hidden_size=self.config.hidden_size,
                num_latent_steps=32,  # 默认值，会在forward中动态调整
                redirect_strength=self.redirect_strength,
                use_mlp_redirector=True,
            ).to(device)
            print(f"[TRAJECTORY] Redirector initialized on device: {device}")

        # ==========================================================
        # Trajectory Unlearning模式
        # ==========================================================

        # 1. 预计算RoPE（复用原有逻辑）
        full_position_ids, full_rope_deltas = self.get_rope_index(
            input_ids,
            image_grid_thw,
            video_grid_thw,
            attention_mask,
        )

        kwargs.pop("rope_deltas", None)
        kwargs.pop("position_ids", None)

        batch_size = input_ids.shape[0]
        idx_start = (input_ids[0] == self.start_latent_id).nonzero(as_tuple=True)[0].item()
        idx_end = (input_ids[0] == self.end_latent_id).nonzero(as_tuple=True)[0].item()
        num_latent_steps = idx_end - idx_start - 1

        # ==========================================================
        # Phase A: Prefix
        # ==========================================================
        prefix_ids = input_ids[:, :idx_start + 1]

        # 调用祖父类的forward，绕过Cocoun_model的forward逻辑
        outputs = Qwen2_5_VLForConditionalGeneration.forward(
            self,
            input_ids=prefix_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache,
            output_hidden_states=True,
            return_dict=True,
            **kwargs
        )

        all_logits = [outputs.logits]
        current_hidden_state = outputs.hidden_states[-1][:, -1:, :]
        all_step_hidden_states = [current_hidden_state]
        current_offset = outputs.logits.shape[1]
        current_pkvs = outputs.past_key_values

        # ==========================================================
        # Phase B: Latent Steps (with Trajectory Monitoring)
        # ==========================================================
        latent_logits = []

        for i in range(num_latent_steps):
            step_pos_ids = full_position_ids[:, :, current_offset : current_offset + 1]

            step_outputs = Qwen2_5_VLForConditionalGeneration.forward(
                self,
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

            current_hidden_state = step_outputs.hidden_states[-1][:, -1:, :]
            current_pkvs = step_outputs.past_key_values

            latent_logits.append(step_outputs.logits)
            all_step_hidden_states.append(current_hidden_state)
            current_offset += 1

        # ==========================================================
        # NEW: Trajectory-Level Intervention
        # ==========================================================
        # 收集所有latent states (只包含真正的latent steps，不包含prefix)
        latent_steps_only = all_step_hidden_states[1:]  # 跳过第一个prefix step

        if len(latent_steps_only) == 0:
            # 如果没有latent steps，跳过trajectory intervention
            print("[TRAJECTORY] Warning: No latent steps found, skipping trajectory intervention")
            all_latent_states = torch.zeros(batch_size, 1, self.config.hidden_size,
                                           device=combined_logits.device, dtype=combined_logits.dtype)
        else:
            all_latent_states = torch.cat(latent_steps_only, dim=1)  # (B, L, D)

        # Step 1: Critic评估风险
        try:
            critic_output = self.critic(all_latent_states)
        except Exception as e:
            print(f"[TRAJECTORY] Error in critic: {e}, all_latent_states.shape={all_latent_states.shape}")
            # 创建空的critic输出
            batch_size = all_latent_states.size(0)
            num_steps = all_latent_states.size(1)
            critic_output = {
                'risk_scores': torch.zeros(batch_size, num_steps, device=all_latent_states.device),
                'forbidden_probs': torch.zeros(batch_size, num_steps, len(self.forbidden_token_ids) if self.forbidden_token_ids else 1,
                                               device=all_latent_states.device),
                'high_risk_mask': torch.zeros(batch_size, num_steps, dtype=torch.bool, device=all_latent_states.device),
                'num_high_risk': torch.zeros(batch_size, dtype=torch.long, device=all_latent_states.device),
            }

        # Step 2: Redirector重定向高风险states
        try:
            redirected_states, redirect_info = self.redirector(
                all_latent_states,
                critic_output['high_risk_mask'],
            )
        except Exception as e:
            print(f"[TRAJECTORY] Error in redirector: {e}, using original states")
            redirected_states = all_latent_states
            redirect_info = {'num_redirected': 0, 'redirect_ratio': 0.0}

        # Step 3: 如果发生了重定向，使用重定向后的states
        if redirect_info['num_redirected'] > 0:
            # 将重定向后的states重新split回list
            step_length = all_latent_states.size(1)
            redirected_list = [
                redirected_states[:, i:i+1, :]
                for i in range(step_length)
            ]
            all_step_hidden_states = redirected_list

            # 更新current_hidden_state为最后一个重定向的state
            current_hidden_state = redirected_states[:, -1:, :]

            print(f"[TRAJECTORY] Redirected {redirect_info['num_redirected']} "
                  f"({redirect_info['redirect_ratio']:.2%}) high-risk latent steps")

        # ==========================================================
        # Phase C: Suffix
        # ==========================================================
        suffix_ids = input_ids[:, idx_end:]
        suffix_pos_ids = full_position_ids[:, :, current_offset:]
        suffix_embeds = self.get_input_embeddings()(suffix_ids)
        suffix_inputs = torch.cat([current_hidden_state, suffix_embeds[:, 1:, :]], dim=1)

        suffix_outputs = Qwen2_5_VLForConditionalGeneration.forward(
            self,
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
        # 结果拼接与Loss计算
        # ==========================================================
        combined_logits = torch.cat([all_logits[0]] + latent_logits + [all_logits[-1]], dim=1)

        prefix_len = all_logits[0].shape[1]
        new_labels_prefix = torch.full((batch_size, prefix_len), -100, device=labels.device)
        new_labels_latent = torch.full((batch_size, num_latent_steps), -100, device=labels.device)
        new_labels_suffix = labels[:, idx_end:]

        aligned_labels = torch.cat([new_labels_prefix, new_labels_latent, new_labels_suffix], dim=1)

        # Loss计算
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

        # ==========================================================
        # NEW: Trajectory-Level Loss
        # ==========================================================
        # 1. Trajectory Risk Loss: 惩罚高风险的latent steps
        trajectory_risk_loss = critic_output['risk_scores'].mean()
        total_loss += (self.unlearning_loss_weight * trajectory_risk_loss)

        # 2. Contrastive Loss (保留原有的)
        if self.forget_concept_vector is not None:
            concept_vec = self.forget_concept_vector.to(all_latent_states.device)
            latent_norm = F.normalize(all_latent_states, p=2, dim=-1)
            concept_norm = F.normalize(concept_vec, p=2, dim=-1)
            cos_sim = (latent_norm * concept_norm).sum(dim=-1).mean()
            unlearning_contrastive_loss = torch.abs(cos_sim)
            total_loss += (self.unlearning_loss_weight * unlearning_contrastive_loss)

        if self.training:
            print(f">>> GA_Loss: {-ga_loss.item():.4f} | "
                  f"Trajectory_Risk: {trajectory_risk_loss.item():.4f} | "
                  f"Redirected: {redirect_info['num_redirected']}")

        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=total_loss,
            logits=combined_logits,
            past_key_values=suffix_outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )
