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

MODEL_ID = '/data2/dmz/multi_Inference_model/R1-Onevision/LLaMA-Factory/saves/r1-onevision-7b-lora-sft-dataGA66-joebiden_final-cocoun-33epochs'

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
# /data2/dmz/llava_test/LLaVA-main/elon/1732086634981.png   /data2/dmz/llava_test/LLaVA-main/all_pic/joebiden/recognized/image_12.jpg    /data2/dmz/llava_test/LLaVA-main/imgsdon/Trump_Image104.jpg
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "/data2/dmz/llava_test/LLaVA-main/imgsdon/Trump_Image104.jpg"},
            {"type": "text", "text": "Hint: who's the main in this picture？"},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
# import pdb
# pdb.set_trace()
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(model.device)

print(f"[DEBUG] inputs shape: {inputs.input_ids.shape}")
print(f"[DEBUG] inputs keys: {inputs.keys()}")

try:
    generated_ids = model.generate(**inputs, max_new_tokens=4096)
    print(f"[DEBUG] generated_ids shape: {generated_ids.shape}")
    print(f"[DEBUG] generated_ids: {generated_ids}")

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    print(f"[DEBUG] generated_ids_trimmed: {generated_ids_trimmed}")

    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(f"[DEBUG] output_text: {output_text}")
    print(output_text)
except Exception as e:
    print(f"[ERROR] Generation failed: {e}")
    traceback.print_exc(file=sys.stdout)