import os
import argparse
from PIL import Image
from llamafactory.chat import ChatModel
from llamafactory.extras.misc import torch_gc

def get_image_files_from_directory(directory, extensions=None):
    """Gets all image files with specified extensions from a directory."""
    if extensions is None:
        extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
    image_files = [os.path.join(directory, f) for f in os.listdir(directory)
                   if f.lower().endswith(extensions)]
    return image_files

def main(cli_args):
    # --- 模型和适配器配置 ---
    # 您可以在这里修改为您需要的模型配置
    args = {
        "model_name_or_path": "/data2/dmz/multi_Inference_model/R1-Onevision/R1-Onevision-7B",
        "adapter_name_or_path": "/data2/dmz/multi_Inference_model/R1-Onevision/LLaMA-Factory/saves/r1-onevision-7b-lora-sft-dataGA66-trump_final-negative-34epochs-3new",
        "finetuning_type": "lora",
        "template": "qwen2_vl", # 修正：使用适用于 Qwen-VL 模型的模板
        "infer_backend": "huggingface",
        "trust_remote_code": True,
        "max_new_tokens":1000
    }
    
    # 创建 ChatModel 实例
    # 这会在后台加载模型和适配器，可能需要一些时间
    print("正在加载模型，请稍候...")
    chat_model = ChatModel(args)
    print("模型加载完成。")

    # --- 推理输入 ---
    image_dir = cli_args.image_dir
    output_file = cli_args.output_file
    prompt = "what's his name?"
    # prompt = "图片中的人是谁？请用中文回答"
    # ------------------------------------

    if not os.path.isdir(image_dir):
        print(f"错误: 文件夹未找到 at {image_dir}")
        return

    image_files = get_image_files_from_directory(image_dir)

    if not image_files:
        print(f"文件夹中没有找到图片: {image_dir}")
        return

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for image_path in image_files:
            print(f"\n正在处理图片: {image_path}")
            # 创建符合格式的消息
            messages = [{"role": "user", "content": prompt}]

            # 使用Pillow库打开图片
            try:
                pil_image = Image.open(image_path)
            except Exception as e:
                print(f"加载图片失败 {image_path}: {e}")
                continue

            # 执行推理
            print("正在进行推理...")
            responses = chat_model.chat(
                messages=messages,
                images=[pil_image] # 将图片以列表形式传入
            )

            # 打印模型的回答
            result_header = f"\n模型对 {os.path.basename(image_path)} 的回答:"
            print(result_header)
            f.write(result_header + "\n")
            for response in responses:
                print(response.response_text)
                f.write(response.response_text + "\n")

    # 清理模型以释放显存
    del chat_model
    torch_gc()
    print("\n所有推理完成并已清理资源。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="对一个文件夹中的图片进行批量推理。")
    parser.add_argument("--image_dir", type=str, default="/data2/dmz/llava_test/LLaVA-main/imgsdon",required=True, help="包含要处理的图片的文件夹路径。")
    parser.add_argument("--output_file", type=str, default="../output/results-trump-16-2.txt", help="输出结果的文件路径。")
    cli_args = parser.parse_args()
    main(cli_args) 







# import os
# from PIL import Image
# from llamafactory.chat import ChatModel
# from llamafactory.extras.misc import torch_gc

# def main():
#     # --- 模型和适配器配置 ---
#     # 您可以在这里修改为您需要的模型配置
#     args = {
#         "model_name_or_path": "/data2/dmz/multi_Inference_model/R1-Onevision/R1-Onevision-7B",
#         "adapter_name_or_path": "/data2/dmz/multi_Inference_model/R1-Onevision/LLaMA-Factory/saves/r1-onevision-7b-lora-sft-dataGA66-trump_final-negative-18epochs-3",
#         "finetuning_type": "lora",
#         "template": "qwen2_vl", # 修正：使用适用于 Qwen-VL 模型的模板
#         "infer_backend": "huggingface",
#         "temperature":0.01,
#         "trust_remote_code": True,
#     }
    
#     # 创建 ChatModel 实例
#     # 这会在后台加载模型和适配器，可能需要一些时间
#     print("正在加载模型，请稍候...")
#     chat_model = ChatModel(args)
#     print("模型加载完成。")

#     # --- 推理输入 ---
#     # 请在这里修改为您想要测试的图片路径和问题
#     image_path = "/data2/dmz/llava_test/LLaVA-main/imgsdon/Trump_Image313.jpg"
#     prompt = "图片中是谁?用中文回答"
#     # ------------------------------------

#     if not os.path.exists(image_path):
#         print(f"错误: 图片文件未找到 at {image_path}")
#         return

#     # 创建符合格式的消息
#     messages = [{"role": "user", "content": prompt}]
    
#     # 使用Pillow库打开图片
#     pil_image = Image.open(image_path)

#     # 执行推理
#     print("\n正在进行推理...")
#     responses = chat_model.chat(
#         messages=messages,
#         images=[pil_image] # 将图片以列表形式传入
#     )

#     # 打印模型的回答
#     print("\n模型回答:")
#     for response in responses:
#         print(response.response_text)

#     # 清理模型以释放显存
#     del chat_model
#     torch_gc()
#     print("\n推理完成并已清理资源。")

# if __name__ == "__main__":
#     main() 