# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from .processor_utils import DatasetProcessor, greedy_knapsack, infer_seqlen


if TYPE_CHECKING:
    from ..mm_plugin import AudioInput, ImageInput, VideoInput


logger = logging.get_logger(__name__)


@dataclass
class SupervisedDatasetProcessor(DatasetProcessor):
    def _encode_data_example(
        self,
        prompt: Sequence[Dict[str, str]],
        response: Sequence[Dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        audios: Sequence["AudioInput"],
    ) -> Tuple[List[int], List[int]]:
        messages = self.template.mm_plugin.process_messages(prompt + response, images, videos, audios, self.processor)
        input_ids, labels = self.template.mm_plugin.process_token_ids(
            [], [], images, videos, audios, self.tokenizer, self.processor
        )
        encoded_pairs = self.template.encode_multiturn(self.tokenizer, messages, system, tools)
        total_length = len(input_ids) + (1 if self.template.efficient_eos else 0)
        if self.data_args.mask_history:
            encoded_pairs = encoded_pairs[::-1]  # high priority for last turns

        for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
            if total_length >= self.data_args.cutoff_len:
                break

            source_len, target_len = infer_seqlen(
                len(source_ids), len(target_ids), self.data_args.cutoff_len - total_length
            )
            source_ids = source_ids[:source_len]
            target_ids = target_ids[:target_len]
            total_length += source_len + target_len

            if self.data_args.train_on_prompt:
                source_label = source_ids
            elif self.template.efficient_eos:
                source_label = [self.tokenizer.eos_token_id] + [IGNORE_INDEX] * (source_len - 1)
            else:
                source_label = [IGNORE_INDEX] * source_len

            if self.data_args.mask_history and turn_idx != 0:  # train on the last turn only
                target_label = [IGNORE_INDEX] * target_len
            else:
                target_label = target_ids

            if self.data_args.mask_history:  # reversed sequences
                input_ids = source_ids + target_ids + input_ids
                labels = source_label + target_label + labels
            else:
                input_ids += source_ids + target_ids
                labels += source_label + target_label

        if self.template.efficient_eos:
            input_ids += [self.tokenizer.eos_token_id]
            labels += [self.tokenizer.eos_token_id]
        # import pdb
        # pdb.set_trace()
        
        return input_ids, labels

    def preprocess_dataset(self, examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
        # for multiturn examples, we only mask the prompt part in each prompt-response pair.
        model_inputs = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            input_ids, labels = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
            )
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)
            model_inputs["images"].append(examples["_images"][i])
            model_inputs["videos"].append(examples["_videos"][i])
            model_inputs["audios"].append(examples["_audios"][i])

        return model_inputs

    def print_data_example(self, example: Dict[str, List[int]]) -> None:
        valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:\n{}".format(example["labels"]))
        print(f"labels:\n{self.tokenizer.decode(valid_labels, skip_special_tokens=False)}")


@dataclass
class CocounDatasetProcessor(SupervisedDatasetProcessor):
    """
    处理 COCONUT（Chain of Continuous Thought）式数据的处理器。
    将 <think>...</think> 部分替换为 latent tokens，
    用于 Qwen2.5-VL 的遗忘任务训练。
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 定义 COCONUT 特殊 token
        self.latent_token = "<latent>"
        self.start_latent_token = "<latent_start>"
        self.end_latent_token = "<latent_end>"
        
        # 添加到 tokenizer
        num_added_toks = self.tokenizer.add_tokens(
            [self.latent_token, self.start_latent_token, self.end_latent_token],
            special_tokens=True
        )
        
        # 获取 token IDs
        self.latent_token_id = self.tokenizer.convert_tokens_to_ids(self.latent_token)
        self.start_latent_id = self.tokenizer.convert_tokens_to_ids(self.start_latent_token)
        self.end_latent_id = self.tokenizer.convert_tokens_to_ids(self.end_latent_token)
        
        print(f"Added {num_added_toks} new tokens for COCONUT")
        print(f"latent_token_id: {self.latent_token_id}")
        print(f"start_latent_id: {self.start_latent_id}")
        # import pdb
        # pdb.set_trace()
        print(f"end_latent_id: {self.end_latent_id}")
    
    def _extract_think_and_answer(self, content: str) -> Tuple[str, str]:
        """
        从 assistant 的内容中提取 <think> 和答案部分。
        
        Args:
            content: 原始内容，格式如 "<think>...</think>答案部分"
        
        Returns:
            (think_content, answer_content)
        """
        think_start_idx = content.find("<think>")
        think_end_idx = content.find("</think>")
        
        if think_start_idx != -1 and think_end_idx != -1:
            think_content = content[think_start_idx + len("<think>") : think_end_idx]
            answer_content = content[think_end_idx + len("</think>") :].strip()
        else:
            # 没有 think 部分，整个作为答案
            think_content = ""
            answer_content = content
        
        return think_content, answer_content
    
    def _create_latent_thought_segment(self, n_latent_tokens: int = 10) -> List[int]:
        """
        创建 COCONUT 风格的 latent thought 段。
        
        Args:
            n_latent_tokens: latent token 的数量
        
        Returns:
            [start_latent_id, latent_token_id, latent_token_id, ..., end_latent_id]
        """
        return (
            [self.start_latent_id]
            + [self.latent_token_id] * n_latent_tokens
            + [self.end_latent_id]
        )
    
    def _encode_data_example(
        self,
        prompt: Sequence[Dict[str, str]],
        response: Sequence[Dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        audios: Sequence["AudioInput"],
        n_latent_tokens: int = 10,  # COCONUT 参数
        use_latent_thought: bool = True,  # 是否启用 latent thought
    ) -> Tuple[List[int], List[int]]:
        """
        编码数据示例，集成 COCONUT 的 latent thought 机制。
        """
        # 获取多模态处理后的 token IDs（通常是视觉 token）
        messages = self.template.mm_plugin.process_messages(prompt + response, images, videos, audios, self.processor)
        input_ids, labels = self.template.mm_plugin.process_token_ids(
            [], [], images, videos, audios, self.tokenizer, self.processor
        )
        
        # 这里记录多模态 token 的长度（用于后续处理）
        mm_token_len = len(input_ids)
        
        encoded_pairs = self.template.encode_multiturn(self.tokenizer, messages, system, tools)
        total_length = len(input_ids) + (1 if self.template.efficient_eos else 0)
        
        if self.data_args.mask_history:
            encoded_pairs = encoded_pairs[::-1]
        
        for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
            if total_length >= self.data_args.cutoff_len:
                break
            
            source_len, target_len = infer_seqlen(
                len(source_ids), len(target_ids), self.data_args.cutoff_len - total_length
            )
            source_ids = source_ids[:source_len]
            target_ids = target_ids[:target_len]
            total_length += source_len + target_len
            
            if self.data_args.train_on_prompt:
                source_label = source_ids
            elif self.template.efficient_eos:
                source_label = [self.tokenizer.eos_token_id] + [IGNORE_INDEX] * (source_len - 1)
            else:
                source_label = [IGNORE_INDEX] * source_len
            
            # ==================== COCONUT 处理 ====================
            # 这里是关键修改：处理 target_ids（assistant 的回复）
            if use_latent_thought and turn_idx == len(encoded_pairs) - 1:  # 只在最后一轮处理
                # 需要找到原始内容并提取 think 和 answer
                # 这需要从 messages 中获取最后一条 assistant 消息
                
                # 获取最后一条消息（应该是 assistant 的回复）
                assistant_messages = [m for m in messages if m.get("role") == "assistant"]
                if assistant_messages:
                    last_assistant_msg = assistant_messages[-1]["content"]
                    think_content, answer_content = self._extract_think_and_answer(last_assistant_msg)
                    
                    if think_content:
                        # 创建 latent thought segment
                        latent_segment = self._create_latent_thought_segment(n_latent_tokens)
                        latent_segment_len = len(latent_segment)
                        
                        # 重新 tokenize answer 部分
                        answer_ids = self.tokenizer.encode(answer_content, add_special_tokens=False)
                        if self.template.efficient_eos:
                            answer_ids += [self.tokenizer.eos_token_id]
                        
                        # 重新构造 target_ids：latent tokens + answer
                        # 注意：latent tokens 的 labels 应该是 IGNORE_INDEX
                        target_ids = latent_segment + answer_ids
                        target_len = len(target_ids)
                        
                        # 对应的 target_label
                        target_label = [IGNORE_INDEX] * latent_segment_len + answer_ids
            # =====================================================
            
            if self.data_args.mask_history and turn_idx != 0:
                target_label = [IGNORE_INDEX] * target_len
            else:
                target_label = target_ids
            
            if self.data_args.mask_history:
                input_ids = source_ids + target_ids + input_ids
                labels = source_label + target_label + labels
            else:
                input_ids += source_ids + target_ids
                labels += source_label + target_label
        
        if self.template.efficient_eos:
            input_ids += [self.tokenizer.eos_token_id]
            labels += [self.tokenizer.eos_token_id]
        
        print(f"[COCONUT] input_ids length: {len(input_ids)}, labels length: {len(labels)}")
        print(f"[COCONUT] labels: {labels}")
        
        return input_ids, labels
    
    def preprocess_dataset(self, examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """
        预处理数据集，集成 COCONUT 处理。
        """
        model_inputs = defaultdict(list)
        
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue
            
            # 调用 COCONUT 版本的编码方法
            input_ids, labels = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
                n_latent_tokens=getattr(self.data_args, 'n_latent_tokens', 10),
                use_latent_thought=getattr(self.data_args, 'use_latent_thought', True),
            )
            
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)
            model_inputs["images"].append(examples["_images"][i])
            model_inputs["videos"].append(examples["_videos"][i])
            model_inputs["audios"].append(examples["_audios"][i])
        
        return model_inputs
    
    def print_data_example(self, example: Dict[str, List[int]]) -> None:
        valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:\n{}".format(example["labels"]))
        print(f"labels:\n{self.tokenizer.decode(valid_labels, skip_special_tokens=False)}")

@dataclass
class PackedSupervisedDatasetProcessor(SupervisedDatasetProcessor):
    def preprocess_dataset(self, examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        # TODO: use `position_ids` to achieve packing
        # build inputs with format `<bos> X1 Y1 <eos> <bos> X2 Y2 <eos>`
        # and labels with format `<ignore> ... <ignore> Y1 <eos> <ignore> ... <ignore> Y2 <eos>`
        valid_num = 0
        batch_input_ids, batch_labels, batch_images, batch_videos, batch_audios = [], [], [], [], []
        lengths = []
        length2indexes = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            input_ids, labels = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
            )
            length = len(input_ids)
            if length > self.data_args.cutoff_len:
                logger.warning_rank0(f"Dropped lengthy example with length {length} > {self.data_args.cutoff_len}.")
            else:
                lengths.append(length)
                length2indexes[length].append(valid_num)
                batch_input_ids.append(input_ids)
                batch_labels.append(labels)
                batch_images.append(examples["_images"][i] or [])
                batch_videos.append(examples["_videos"][i] or [])
                batch_audios.append(examples["_audios"][i] or [])
                valid_num += 1

        model_inputs = defaultdict(list)
        knapsacks = greedy_knapsack(lengths, self.data_args.cutoff_len)
        for knapsack in knapsacks:
            packed_input_ids, packed_attention_masks, packed_labels = [], [], []
            packed_images, packed_videos, packed_audios = [], [], []
            for i, length in enumerate(knapsack):
                index = length2indexes[length].pop()
                packed_input_ids += batch_input_ids[index]
                packed_labels += batch_labels[index]
                packed_images += batch_images[index]
                packed_videos += batch_videos[index]
                packed_audios += batch_audios[index]
                if self.data_args.neat_packing:
                    packed_attention_masks += [i + 1] * len(batch_input_ids[index])  # start from 1
                else:
                    packed_attention_masks += [1] * len(batch_input_ids[index])

            if len(packed_input_ids) < self.data_args.cutoff_len + 1:  # avoid flash_attn drops attn mask
                pad_length = self.data_args.cutoff_len - len(packed_input_ids) + 1
                packed_input_ids += [self.tokenizer.pad_token_id] * pad_length
                packed_labels += [IGNORE_INDEX] * pad_length
                if self.data_args.neat_packing:
                    packed_attention_masks += [0] * pad_length
                else:
                    packed_attention_masks += [1] * pad_length  # more efficient flash_attn

            if len(packed_input_ids) != self.data_args.cutoff_len + 1:
                raise ValueError("The length of packed example should be identical to the cutoff length.")

            model_inputs["input_ids"].append(packed_input_ids)
            model_inputs["attention_mask"].append(packed_attention_masks)
            model_inputs["labels"].append(packed_labels)
            model_inputs["images"].append(packed_images or None)
            model_inputs["videos"].append(packed_videos or None)
            model_inputs["audios"].append(packed_audios or None)

        return model_inputs
