from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from src.llamafactory.model.cocoun_model import Cocoun_model
import torch
from qwen_vl_utils import process_vision_info
import os
from pathlib import Path
import gc
from PIL import Image

import sys
import traceback

MODEL_ID = '/data2/dmz/multi_Inference_model/R1-Onevision/LLaMA-Factory/saves/r1-onevision-7b-lora-sft-dataGA66-joebiden_final-cocoun_trajectory-70epochs'

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     MODEL_ID,
#     trust_remote_code=True,
#     torch_dtype=torch.bfloat16
# ).to("cuda").eval()
print("hellllo")
model = Cocoun_model.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
# 图片目录
IMAGE_DIR = "/datanfs2/dmz/12_27/LLaVA-main/imgsdon"  #/datanfs2/dmz/12_27/LLaVA-main/imgsdon    /datanfs2/dmz/12_27/LLaVA-main/all_pic/joebiden/recognized

# 图片最大尺寸（长边），控制显存占用
MAX_IMAGE_SIZE = 1024  # 可以根据显存大小调整：768, 1024, 1280

# 支持的图片格式
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}

# 图片预处理函数：控制分辨率
def resize_image(image_path, max_size=MAX_IMAGE_SIZE):
    """调整图片大小以控制显存占用，保持宽高比"""
    img = Image.open(image_path)

    # 转换为RGB模式（处理RGBA等格式）
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # 获取原始尺寸
    width, height = img.size
    print(f"  Original size: {width}x{height}")

    # 如果图片已经小于max_size，不需要resize
    if max(width, height) <= max_size:
        return img

    # 计算新的尺寸（保持宽高比）
    if width > height:
        new_width = max_size
        new_height = int(height * max_size / width)
    else:
        new_height = max_size
        new_width = int(width * max_size / height)

    # Resize
    img_resized = img.resize((new_width, new_height), Image.LANCZOS)
    print(f"  Resized to: {new_width}x{new_height}")

    return img_resized

# 获取所有图片文件
def get_image_files(directory):
    image_files = []
    for file in Path(directory).iterdir():
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
            image_files.append(file)
    return sorted(image_files)

image_files = get_image_files(IMAGE_DIR)
print(f"Found {len(image_files)} images in {IMAGE_DIR}")

# 保存结果
results = []

# 遍历所有图片
for idx, image_path in enumerate(image_files):
    print(f"\n{'='*60}")
    print(f"Processing image {idx+1}/{len(image_files)}: {image_path.name}")
    print(f"{'='*60}")

    try:
        # 预处理图片：resize控制显存
        img = resize_image(image_path, max_size=MAX_IMAGE_SIZE)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},  # 直接传入PIL Image对象
                    {"type": "text", "text": "Hint: who's the main in this picture？"},
                ],
            }
        ]

        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=4096)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        print(f"Result: {output_text}")
        results.append({
            "image": str(image_path),
            "result": output_text
        })

        # 清理显存
        del inputs, generated_ids, generated_ids_trimmed
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"[ERROR] Generation failed for {image_path.name}: {e}")
        traceback.print_exc(file=sys.stdout)
        results.append({
            "image": str(image_path),
            "result": f"ERROR: {str(e)}"
        })

        # 出错时也要清理显存
        gc.collect()
        torch.cuda.empty_cache()

# 输出汇总
print(f"\n\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
for result in results:
    print(f"\nImage: {result['image']}")
    print(f"Result: {result['result']}")