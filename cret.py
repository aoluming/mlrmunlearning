# from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
# import torch
# from qwen_vl_utils import process_vision_info

# # MODEL_ID = "/data2/dmz/multi_Inference_model/R1-Onevision/R1-Onevision-7B"
# MODEL_ID = '/data2/dmz/multi_Inference_model/R1-Onevision/LLaMA-Factory/saves/r1-onevision-7b-lora-sft-dataGA66-trump_final-negative-50epochs'
# # MODEL_ID = "Fancy-MLLM/R1-Onevision-7B"
# processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     MODEL_ID,
#     trust_remote_code=True,
#     torch_dtype=torch.bfloat16
# ).to("cuda").eval()

# messages = [
#     {
#         "role": "user",
#         "content": [
#             {"type": "image", "image": "/data2/dmz/llava_test/LLaVA-main/imgsdon/Trump_Image1.jpg"},
#             {"type": "text", "text": "Hint: who's the main in this picture？"},
#         ],
#     }
# ]

# # Preparation for inference
# text = processor.apply_chat_template(
#     messages, tokenize=False, add_generation_prompt=True
# )
# image_inputs, video_inputs = process_vision_info(messages)
# inputs = processor(
#     text=[text],
#     images=image_inputs,
#     videos=video_inputs,
#     padding=True,
#     return_tensors="pt",
# )
# inputs = inputs.to(model.device)

# generated_ids = model.generate(**inputs, max_new_tokens=4096)
# generated_ids_trimmed = [
#     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
# ]
# output_text = processor.batch_decode(
#     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
# )
# print(output_text)



from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import torch
from qwen_vl_utils import process_vision_info
import os
from pathlib import Path
import gc
from PIL import Image

MODEL_ID = '/data2/dmz/multi_Inference_model/R1-Onevision/LLaMA-Factory/saves/r1-onevision-7b-lora-sft-dataGA66-joe_final-cocoun-33epochs'

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
).to("cuda").eval()

# 图片文件夹路径
image_folder = "/data2/dmz/llava_test/LLaVA-main/imgsdon"
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff'}

# 获取所有图片
image_files = []
for file in os.listdir(image_folder):
    file_path = os.path.join(image_folder, file)
    if os.path.isfile(file_path) and Path(file).suffix.lower() in image_extensions:
        image_files.append(file_path)

image_files.sort()

print(f"找到 {len(image_files)} 张图片，开始推理...\n")
print("=" * 80)

# ===== 设置最大尺寸限制 =====
MAX_SIZE = 1280  # 根据你的显存情况，1280应该是安全的
# ========================

def resize_image_if_needed(image_path, max_size=1280):
    """如果图片超过最大尺寸，则调整大小"""
    img = Image.open(image_path)
    width, height = img.size
    
    # 如果图片不大，直接返回原路径
    if max(width, height) <= max_size:
        img.close()
        return image_path, width, height, False
    
    # 需要调整大小
    if width > height:
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        new_height = max_size
        new_width = int(width * (max_size / height))
    
    # 使用高质量的 LANCZOS 重采样
    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # 保存到临时文件
    temp_dir = "/tmp/resized_images"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"temp_{os.path.basename(image_path)}")
    img_resized.save(temp_path, quality=95)
    
    img.close()
    img_resized.close()
    
    return temp_path, new_width, new_height, True

# 统计
success_count = 0
skip_count = 0
error_count = 0

# 遍历每张图片进行推理
for idx, image_path in enumerate(image_files, 1):
    print(f"\n[{idx}/{len(image_files)}] 正在处理: {os.path.basename(image_path)}")
    print("-" * 80)
    
    try:
        # 调整图片大小（如果需要）
        processed_path, width, height, was_resized = resize_image_if_needed(image_path, MAX_SIZE)
        
        if was_resized:
            print(f"原始尺寸: {Image.open(image_path).size}")
            print(f"已调整为: {width}x{height}")
        else:
            print(f"原始尺寸: {width}x{height} (无需调整)")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": processed_path},
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
        
        # 生成输出
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=2048)  # 减少到2048
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        
        print(f"答案: {output_text[0]}")
        success_count += 1
        
        # 清理显存
        del inputs, generated_ids, generated_ids_trimmed, image_inputs, video_inputs
        torch.cuda.empty_cache()
        gc.collect()
        
        # 删除临时文件
        if was_resized and os.path.exists(processed_path):
            os.remove(processed_path)
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"⚠️ 显存不足，跳过此图片")
        skip_count += 1
        torch.cuda.empty_cache()
        gc.collect()
        continue
        
    except Exception as e:
        print(f"❌ 处理图片时出错: {str(e)}")
        error_count += 1
        torch.cuda.empty_cache()
        gc.collect()
    
    print("-" * 80)
    
    # 每10张图片打印一次显存状态
    if idx % 10 == 0:
        print(f"\n📊 显存状态:")
        print(f"  已分配: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        print(f"  已缓存: {torch.cuda.memory_reserved()/1024**3:.2f} GB\n")

print(f"\n{'=' * 80}")
print(f"✅ 推理完成！")
print(f"  总共: {len(image_files)} 张")
print(f"  成功: {success_count} 张")
print(f"  跳过: {skip_count} 张")
print(f"  错误: {error_count} 张")



# from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
# import torch
# from qwen_vl_utils import process_vision_info

# # MODEL_ID = "/data2/dmz/multi_Inference_model/R1-Onevision/R1-Onevision-7B"
# MODEL_ID = '/data2/dmz/multi_Inference_model/R1-Onevision/LLaMA-Factory/saves/r1-onevision-7b-lora-sft-dataGA66-trump_final-negative-50epochs'
# # MODEL_ID = "Fancy-MLLM/R1-Onevision-7B"
# processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     MODEL_ID,
#     trust_remote_code=True,
#     torch_dtype=torch.bfloat16
# ).to("cuda").eval()

# messages = [
#     {
#         "role": "user",
#         "content": [
#             {"type": "image", "image": "/data2/dmz/llava_test/LLaVA-main/imgsdon/Trump_Image1.jpg"},
#             {"type": "text", "text": "Hint: who's the main in this picture？"},
#         ],
#     }
# ]

# # Preparation for inference
# text = processor.apply_chat_template(
#     messages, tokenize=False, add_generation_prompt=True
# )
# image_inputs, video_inputs = process_vision_info(messages)
# inputs = processor(
#     text=[text],
#     images=image_inputs,
#     videos=video_inputs,
#     padding=True,
#     return_tensors="pt",
# )
# inputs = inputs.to(model.device)

# generated_ids = model.generate(**inputs, max_new_tokens=4096)
# generated_ids_trimmed = [
#     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
# ]
# output_text = processor.batch_decode(
#     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
# )
# print(output_text)



# from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# from src.models.cocoun_model import Cocoun_model
# import torch
# from qwen_vl_utils import process_vision_info
# import os
# from pathlib import Path
# import gc
# from PIL import Image

# MODEL_ID = '/data2/dmz/multi_Inference_model/R1-Onevision/LLaMA-Factory/saves/r1-onevision-7b-lora-sft-dataGA66-trump_final-cocoun-100epochs'

# processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
# # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
# #     MODEL_ID,
# #     trust_remote_code=True,
# #     torch_dtype=torch.bfloat16
# # ).to("cuda").eval()

# model = Cocoun_model.from_pretrained(
#     MODEL_ID,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
#     trust_remote_code=True
# )

# # 图片文件夹路径
# image_folder = "/data2/dmz/llava_test/LLaVA-main/imgsdon"
# image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff'}

# # 获取所有图片
# image_files = []
# for file in os.listdir(image_folder):
#     file_path = os.path.join(image_folder, file)
#     if os.path.isfile(file_path) and Path(file).suffix.lower() in image_extensions:
#         image_files.append(file_path)

# image_files.sort()

# print(f"找到 {len(image_files)} 张图片，开始推理...\n")
# print("=" * 80)

# # ===== 设置最大尺寸限制 =====
# MAX_SIZE = 1280  # 根据你的显存情况，1280应该是安全的
# # ========================

# def resize_image_if_needed(image_path, max_size=1280):
#     """如果图片超过最大尺寸，则调整大小"""
#     img = Image.open(image_path)
#     width, height = img.size
    
#     # 如果图片不大，直接返回原路径
#     if max(width, height) <= max_size:
#         img.close()
#         return image_path, width, height, False
    
#     # 需要调整大小
#     if width > height:
#         new_width = max_size
#         new_height = int(height * (max_size / width))
#     else:
#         new_height = max_size
#         new_width = int(width * (max_size / height))
    
#     # 使用高质量的 LANCZOS 重采样
#     img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
#     # 保存到临时文件
#     temp_dir = "/tmp/resized_images"
#     os.makedirs(temp_dir, exist_ok=True)
#     temp_path = os.path.join(temp_dir, f"temp_{os.path.basename(image_path)}")
#     img_resized.save(temp_path, quality=95)
    
#     img.close()
#     img_resized.close()
    
#     return temp_path, new_width, new_height, True

# # 统计
# success_count = 0
# skip_count = 0
# error_count = 0

# # 遍历每张图片进行推理
# for idx, image_path in enumerate(image_files, 1):
#     print(f"\n[{idx}/{len(image_files)}] 正在处理: {os.path.basename(image_path)}")
#     print("-" * 80)
    
#     try:
#         # 调整图片大小（如果需要）
#         processed_path, width, height, was_resized = resize_image_if_needed(image_path, MAX_SIZE)
        
#         if was_resized:
#             print(f"原始尺寸: {Image.open(image_path).size}")
#             print(f"已调整为: {width}x{height}")
#         else:
#             print(f"原始尺寸: {width}x{height} (无需调整)")
        
#         messages = [
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "image", "image": processed_path},
#                     {"type": "text", "text": "Hint: who's the main in this picture？"},
#                 ],
#             }
#         ]
        
#         # Preparation for inference
#         text = processor.apply_chat_template(
#             messages, tokenize=False, add_generation_prompt=True
#         )
#         image_inputs, video_inputs = process_vision_info(messages)
#         inputs = processor(
#             text=[text],
#             images=image_inputs,
#             videos=video_inputs,
#             padding=True,
#             return_tensors="pt",
#         )
#         inputs = inputs.to(model.device)
        
#         # 生成输出
#         with torch.no_grad():
#             generated_ids = model.generate(**inputs, max_new_tokens=2048)  # 减少到2048
#             generated_ids_trimmed = [
#                 out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
#             ]
#             output_text = processor.batch_decode(
#                 generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
#             )
        
#         print(f"答案: {output_text[0]}")
#         success_count += 1
        
#         # 清理显存
#         del inputs, generated_ids, generated_ids_trimmed, image_inputs, video_inputs
#         torch.cuda.empty_cache()
#         gc.collect()
        
#         # 删除临时文件
#         if was_resized and os.path.exists(processed_path):
#             os.remove(processed_path)
        
#     except torch.cuda.OutOfMemoryError as e:
#         print(f"⚠️ 显存不足，跳过此图片")
#         skip_count += 1
#         torch.cuda.empty_cache()
#         gc.collect()
#         continue
        
#     except Exception as e:
#         print(f"❌ 处理图片时出错: {str(e)}")
#         error_count += 1
#         torch.cuda.empty_cache()
#         gc.collect()
    
#     print("-" * 80)
    
#     # 每10张图片打印一次显存状态
#     if idx % 10 == 0:
#         print(f"\n📊 显存状态:")
#         print(f"  已分配: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
#         print(f"  已缓存: {torch.cuda.memory_reserved()/1024**3:.2f} GB\n")

# print(f"\n{'=' * 80}")
# print(f"✅ 推理完成！")
# print(f"  总共: {len(image_files)} 张")
# print(f"  成功: {success_count} 张")
# print(f"  跳过: {skip_count} 张")
# print(f"  错误: {error_count} 张")
