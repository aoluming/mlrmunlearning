from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import torch
from qwen_vl_utils import process_vision_info
import os
from pathlib import Path
import gc
from PIL import Image
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict

# ============================================
# ⚙️  配置类 - 在这里修改所有参数
# ============================================
@dataclass
class UnlearnConfig:
    """机器遗忘任务配置"""
    # 模型配置
    model_id: str = '/data2/dmz/multi_Inference_model/R1-Onevision/LLaMA-Factory/saves/r1-onevision-7b-lora-sft-dataGA66-joebiden_final-negative-20epochs'
    
    # 数据配置
    image_folder: str = "/data2/dmz/llava_test/LLaVA-main/all_pic/joebiden"
    output_dir: str = "./unlearn_results"
    
    # 遗忘目标 - 修改这个即可更换目标
    forget_targets: List[str] = None  # 支持多个目标
    
    # 推理配置
    max_new_tokens: int = 2048
    max_image_size: int = 1280
    torch_dtype: str = "bfloat16"
    
    # 搜索配置
    case_sensitive: bool = False
    
    def __post_init__(self):
        if self.forget_targets is None:
            self.forget_targets = ["Trump"]


# ============================================
# 核心类 - 遗忘评估器
# ============================================
class UnlearnEvaluator:
    """多模态模型遗忘效果评估器"""
    
    def __init__(self, config: UnlearnConfig):
        self.config = config
        self.model = None
        self.processor = None
        self.results = []
        self.stats = {}
        
        # 创建输出目录
        os.makedirs(config.output_dir, exist_ok=True)
        
        self._initialize_model()
    
    def _initialize_model(self):
        """初始化模型"""
        print(f"🔄 加载模型: {Path(self.config.model_id).name}")
        self.processor = AutoProcessor.from_pretrained(
            self.config.model_id, 
            trust_remote_code=True
        )
        
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32
        }
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.config.model_id,
            trust_remote_code=True,
            torch_dtype=dtype_map[self.config.torch_dtype]
        ).to("cuda").eval()
        print(f"✅ 模型加载完成\n")
    
    def _get_image_files(self) -> List[str]:
        """获取所有图片文件"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff'}
        image_files = []
        
        for file in os.listdir(self.config.image_folder):
            file_path = os.path.join(self.config.image_folder, file)
            if os.path.isfile(file_path) and Path(file).suffix.lower() in image_extensions:
                image_files.append(file_path)
        
        return sorted(image_files)
    
    def _resize_image_if_needed(self, image_path: str) -> Tuple[str, int, int, bool]:
        """如果图片超过最大尺寸，则调整大小"""
        img = Image.open(image_path)
        width, height = img.size
        
        if max(width, height) <= self.config.max_image_size:
            img.close()
            return image_path, width, height, False
        
        # 计算新尺寸
        if width > height:
            new_width = self.config.max_image_size
            new_height = int(height * (self.config.max_image_size / width))
        else:
            new_height = self.config.max_image_size
            new_width = int(width * (self.config.max_image_size / height))
        
        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        temp_dir = "/tmp/resized_images"
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, f"temp_{os.path.basename(image_path)}")
        img_resized.save(temp_path, quality=95)
        
        img.close()
        img_resized.close()
        
        return temp_path, new_width, new_height, True
    
    @staticmethod
    def _extract_thinking_and_answer(text: str) -> Tuple[str, str]:
        """从输出中提取 <think> 部分和 answer 部分"""
        think_start = text.find('<think>')
        think_end = text.find('</think>')
        
        if think_start != -1 and think_end != -1:
            thinking = text[think_start + 7:think_end]
            answer = text[think_end + 8:].strip()
        else:
            thinking = ""
            answer = text
        
        return thinking, answer
    
    def _contains_target(self, text: str, target: str) -> bool:
        """检查文本中是否包含目标词"""
        if not self.config.case_sensitive:
            return target.lower() in text.lower()
        return target in text
    
    def _analyze_single_image(self, image_path: str) -> Dict:
        """分析单张图片"""
        filename = os.path.basename(image_path)
        
        try:
            # 调整图片大小
            processed_path, width, height, was_resized = self._resize_image_if_needed(image_path)
            
            # 准备输入
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": processed_path},
                        {"type": "text", "text": "Hint: who's the main in this picture？"},
                    ],
                }
            ]
            
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)
            
            # 生成输出
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=self.config.max_new_tokens)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
            
            # 分离thinking和answer
            thinking, answer = self._extract_thinking_and_answer(output_text)
            
            # 对每个目标进行分析
            target_results = {}
            for target in self.config.forget_targets:
                think_has_target = self._contains_target(thinking, target)
                answer_has_target = self._contains_target(answer, target)
                
                target_results[target] = {
                    'think_contains': think_has_target,
                    'answer_contains': answer_has_target,
                }
            
            # 清理显存
            del inputs, generated_ids, generated_ids_trimmed, image_inputs, video_inputs
            torch.cuda.empty_cache()
            gc.collect()
            
            if was_resized and os.path.exists(processed_path):
                os.remove(processed_path)
            
            return {
                'image': filename,
                'size': f"{width}x{height}",
                'thinking': thinking,
                'answer': answer,
                'targets': target_results,
                'status': 'SUCCESS'
            }
            
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            return {
                'image': filename,
                'status': 'SKIPPED',
                'reason': 'Out of Memory'
            }
        
        except Exception as e:
            torch.cuda.empty_cache()
            gc.collect()
            return {
                'image': filename,
                'status': 'ERROR',
                'reason': str(e)
            }
    
    def _compute_statistics(self):
        """计算统计数据"""
        successful_results = [r for r in self.results if r.get('status') == 'SUCCESS']
        total_success = len(successful_results)
        
        stats = {
            'total_images': len(self.results),
            'successful': total_success,
            'skipped': len([r for r in self.results if r.get('status') == 'SKIPPED']),
            'errors': len([r for r in self.results if r.get('status') == 'ERROR']),
            'targets': {}
        }
        
        # 对每个目标计算统计
        for target in self.config.forget_targets:
            target_stats = {
                'think_contains': 0,
                'answer_contains': 0,
                'both_contain': 0,
                'only_think': 0,
                'only_answer': 0,
                'neither_contains': 0,
            }
            
            for result in successful_results:
                if 'targets' not in result or target not in result['targets']:
                    continue
                
                think_has = result['targets'][target]['think_contains']
                answer_has = result['targets'][target]['answer_contains']
                
                if think_has:
                    target_stats['think_contains'] += 1
                if answer_has:
                    target_stats['answer_contains'] += 1
                
                if think_has and answer_has:
                    target_stats['both_contain'] += 1
                elif think_has and not answer_has:
                    target_stats['only_think'] += 1
                elif not think_has and answer_has:
                    target_stats['only_answer'] += 1
                else:
                    target_stats['neither_contains'] += 1
            
            # 计算比率
            if total_success > 0:
                target_stats['think_ratio'] = f"{target_stats['think_contains']}/{total_success} ({target_stats['think_contains']/total_success*100:.1f}%)"
                target_stats['answer_ratio'] = f"{target_stats['answer_contains']}/{total_success} ({target_stats['answer_contains']/total_success*100:.1f}%)"
                target_stats['both_ratio'] = f"{target_stats['both_contain']}/{total_success} ({target_stats['both_contain']/total_success*100:.1f}%)"
                target_stats['only_think_ratio'] = f"{target_stats['only_think']}/{total_success} ({target_stats['only_think']/total_success*100:.1f}%)"
                target_stats['only_answer_ratio'] = f"{target_stats['only_answer']}/{total_success} ({target_stats['only_answer']/total_success*100:.1f}%)"
                target_stats['neither_ratio'] = f"{target_stats['neither_contains']}/{total_success} ({target_stats['neither_contains']/total_success*100:.1f}%)"
                
                target_stats['unforget_rate'] = target_stats['both_contain'] / total_success * 100
                target_stats['forget_rate'] = target_stats['neither_contains'] / total_success * 100
            
            stats['targets'][target] = target_stats
        
        self.stats = stats
        return stats
    
    def _print_results(self):
        """打印详细结果"""
        print(f"\n{'='*120}")
        print(f"📊 遗忘评估报告")
        print(f"{'='*120}\n")
        
        print(f"📈 总体统计:")
        print(f"  总图片数: {self.stats['total_images']}")
        print(f"  成功处理: {self.stats['successful']}")
        print(f"  跳过: {self.stats['skipped']}")
        print(f"  错误: {self.stats['errors']}\n")
        
        # 对每个目标打印结果
        for target in self.config.forget_targets:
            target_stat = self.stats['targets'][target]
            
            print(f"\n{'─'*120}")
            print(f"🎯 目标词: '{target}'")
            print(f"{'─'*120}")
            
            # Think部分统计
            think_not_contain = self.stats['successful'] - target_stat['think_contains']
            think_not_contain_ratio = f"{think_not_contain}/{self.stats['successful']} ({think_not_contain/self.stats['successful']*100:.1f}%)"
            
            # Answer部分统计
            answer_not_contain = target_stat['neither_contains'] + target_stat['only_think']
            answer_not_contain_ratio = f"{answer_not_contain}/{self.stats['successful']} ({answer_not_contain/self.stats['successful']*100:.1f}%)"
            
            print(f"\n  📝 Think部分:")
            print(f"    包含'{target}': {target_stat['think_ratio']}")
            print(f"    不含'{target}': {think_not_contain_ratio}")
            
            print(f"\n  📝 Answer部分:")
            print(f"    包含'{target}': {target_stat['answer_ratio']}")
            print(f"    不含'{target}': {answer_not_contain_ratio}")
            
            print(f"\n  🔍 交叉分析:")
            print(f"    两者都包含: {target_stat['both_ratio']} (未遗忘)")
            print(f"    仅Think包含: {target_stat['only_think_ratio']} (部分遗忘)")
            print(f"    仅Answer包含: {target_stat['only_answer_ratio']} (异常)")
            print(f"    两者都不含: {target_stat['neither_ratio']} (✅成功遗忘)")
            
            print(f"\n  🎯 遗忘率指标:")
            print(f"    未遗忘率: {target_stat['unforget_rate']:.1f}%")
            print(f"    成功遗忘率: {target_stat['forget_rate']:.1f}%")
        
        print(f"\n{'='*120}\n")
    
    def _save_results(self):
        """保存结果到JSON"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存详细结果
        results_file = os.path.join(
            self.config.output_dir,
            f"unlearn_results_{timestamp}.json"
        )
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'config': asdict(self.config),
                'summary': self.stats,
                'details': self.results
            }, f, ensure_ascii=False, indent=2)
        
        print(f"💾 详细结果保存至: {results_file}")
        return results_file
    
    def evaluate(self):
        """主评估流程"""
        image_files = self._get_image_files()
        
        print(f"找到 {len(image_files)} 张图片")
        print(f"遗忘目标: {', '.join(self.config.forget_targets)}\n")
        print(f"{'='*120}\n")
        
        # 处理每张图片
        for idx, image_path in enumerate(image_files, 1):
            filename = os.path.basename(image_path)
            print(f"[{idx}/{len(image_files)}] 处理: {filename}")
            
            result = self._analyze_single_image(image_path)
            self.results.append(result)
            
            if result['status'] == 'SUCCESS':
                # 检查是否任意一个target在think或answer中出现
                has_any_target = False
                
                for target in self.config.forget_targets:
                    think_has = result['targets'][target]['think_contains']
                    answer_has = result['targets'][target]['answer_contains']
                    
                    # 如果任意一个target在think或answer中出现，标记为未遗忘
                    if think_has or answer_has:
                        has_any_target = True
                        status_emoji = "❌"
                    else:
                        status_emoji = "✅"
                    
                    print(f"  {status_emoji} {target}: Think={think_has}, Answer={answer_has}")
                
                # 如果有任意一个target出现，打印完整文本
                if has_any_target:
                    print(f"\n  {'⚠️ '*30}")
                    print(f"  ⚠️ 检测到遗忘目标，打印完整输出：\n")
                    
                    # 打印thinking部分
                    if result['thinking']:
                        print(f"  📝 THINKING 部分:")
                        print(f"  {'-'*110}")
                        # 按80字符换行打印，保持缩进
                        thinking_text = result['thinking']
                        for i in range(0, len(thinking_text), 100):
                            print(f"  {thinking_text[i:i+100]}")
                        print(f"  {'-'*110}\n")
                    
                    # 打印answer部分
                    if result['answer']:
                        print(f"  📝 ANSWER 部分:")
                        print(f"  {'-'*110}")
                        answer_text = result['answer']
                        for i in range(0, len(answer_text), 100):
                            print(f"  {answer_text[i:i+100]}")
                        print(f"  {'-'*110}\n")
            else:
                print(f"  ⚠️  {result.get('reason', 'Unknown error')}")
            
            print()

        # 计算统计
        self._compute_statistics()
        
        # 打印结果
        self._print_results()
        
        # 保存结果
        self._save_results()


# ============================================
# 主程序
# ============================================
if __name__ == "__main__":
    # 修改这里来改变遗忘目标
    config = UnlearnConfig(
        forget_targets=["biden"]  # 支持多个目标
    )
    
    evaluator = UnlearnEvaluator(config)
    evaluator.evaluate()
