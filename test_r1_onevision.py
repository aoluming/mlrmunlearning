import os
import argparse
import torch
from PIL import Image
import requests
from io import BytesIO
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

# It's recommended to set trust_remote_code=True for Qwen-VL,
# as it may contain custom code for model processing.
# You will also need to have upgraded transformers, accelerate, and installed qwen-vl-utils.
# pip install -U "transformers>=4.42" accelerate "qwen-vl-utils[decord]==0.0.8"
# pip install bitsandbytes # If you plan to use quantization

def load_image(image_file):
    """Loads a single image from a file path or URL."""
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def get_image_files_from_directory(directory, extensions=None):
    """Gets all image files with specified extensions from a directory."""
    if extensions is None:
        extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
    image_files = [os.path.join(directory, f) for f in os.listdir(directory)
                   if f.lower().endswith(extensions)]
    return image_files

def eval_model(args):
    # Load the processor and model from Hugging Face
    # trust_remote_code=True is necessary for models like Qwen-VL.
    print("Loading processor...")
    # Constrain image resolution to prevent OOM errors during inference
    processor = AutoProcessor.from_pretrained(
        args.model_base, 
        trust_remote_code=True,
        max_pixels=448*448
    )

    print("Loading base model in float16 (half-precision)...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_base,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # If a LoRA model path is provided, load and merge the weights
    if args.model_path:
        print(f"Loading LoRA weights from {args.model_path}...")
        model = PeftModel.from_pretrained(model, args.model_path)
        print("Merging LoRA weights...")
        model = model.merge_and_unload()

    # The prompt to be used for all images.
    prompt = "图片中是谁?用中文回答"

    # Get all image files from the specified directory
    image_dir = args.image_file
    image_files = get_image_files_from_directory(image_dir)

    if not image_files:
        print(f"No images found in directory: {image_dir}")
        return

    # Process each image file
    for image_path in image_files:
        print(f"Processing image: {image_path}")
        try:
            image = load_image(image_path)
        except Exception as e:
            print(f"Failed to load image {image_path}: {e}")
            continue

        # Create the message structure for Qwen-VL processor
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Apply the chat template and prepare inputs for the model
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text], images=[image], return_tensors="pt"
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate output from the model
        with torch.no_grad():
            # The generation parameters can be customized here
            output = model.generate(
                **inputs,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                return_dict_in_generate=True
            )

        # Decode the generated text, removing the prompt part
        prompt_len = len(inputs["input_ids"][0])
        generated_ids = output.sequences[0][prompt_len:]
        generated_text = processor.decode(generated_ids, skip_special_tokens=True)
        
        print(f"Generated_text for {os.path.basename(image_path)}: {generated_text}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Path to the fine-tuned LoRA adapter weights. Can be empty if using the base model directly.
    parser.add_argument("--model-path", type=str, default="./LLaMA-Factory/saves/r1-onevision-7b-lora-sft-dataGA66-trump_final-negative-18epochs")
    # Path to the base model (e.g., Qwen/Qwen2.5-VL-7B-Instruct)
    parser.add_argument("--model-base", type=str, default="./R1-Onevision-7B")
    parser.add_argument("--image-file", type=str, required=True,
                        help="Path to the folder containing all images to be processed.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature for generation.")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p sampling parameter.")
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for beam search.")
    parser.add_argument("--max_new_tokens", type=int, default=600, help="Maximum number of new tokens to generate.")
    args = parser.parse_args()

    # Check if the image_file path is a directory
    if not os.path.isdir(args.image_file):
        raise ValueError(f"The --image-file argument must be a directory. Provided: {args.image_file}")

    eval_model(args) 