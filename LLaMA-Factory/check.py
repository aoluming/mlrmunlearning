# filepath = "/data2/dmz/multi_Inference_model/R1-Onevision/LLaMA-Factory/data/joebidenga_final.jsonl"

# with open(filepath, 'r', encoding='utf-8') as f:
#     first_line = f.readline().strip()

# print("First line (raw):")
# print(repr(first_line))

# print("\nCharacters in line (with code points):")
# for i, char in enumerate(first_line):
#     if ord(char) < 32 or ord(char) > 126 or char in ['"', '\\', '\n', '\t']:
#         print(f"  [{i}] '{char}' (U+{ord(char):04X})")


# 查看第一个引号是否是标准引号
# first_line = '{"id": "9a88d8d8-1a45-4596-b92a-8852253422da", "images": ["/data2/dmz/llava_test/LLaVA-main/all_pic/joebiden/image_1.jpg"], "messages": [{"role": "user", "content": "<image>What\'s the name of the central figure in this photograph?"}, {"role": "assistant", "content": " escalated, so I need to figure out the name of the central figure in this photograph based on the given image content. Let me start by analyzing the image itself."}]}'


# # 查找所有引号
# for i, char in enumerate(first_line):
#     if char in ['"', '“', '”', '‘', '’']:
#         print(f"Position {i}: '{char}' (U+{ord(char):04X})")

# fix_content.py
input_file = "/data2/dmz/multi_Inference_model/R1-Onevision/LLaMA-Factory/data/joebidenga_final.jsonl"
output_file = "./fixed.jsonl"

with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
    for line_num, line in enumerate(f_in, 1):
        if '"content": ' in line:
            # 检查是否缺少 "
            if not line.strip().startswith('"content": "'):
                # 修复：添加 "
                fixed_line = line.replace('"content": ', '"content": "')
                f_out.write(fixed_line)
                print(f"🔧 Fixed line {line_num}")
            else:
                f_out.write(line)
        else:
            f_out.write(line)

print("✅ Fixed file saved to /tmp/fixed.jsonl")
