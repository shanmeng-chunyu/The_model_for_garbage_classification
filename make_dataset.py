import os
import json
import random

# ================= 配置区域 =================

# 1. 图片数据的根目录
# 如果脚本就在图片文件夹里，用 "." 即可
# 如果在别处，请填入绝对路径，例如 r"C:\Users\21048\Desktop\MyGarbageImages"
ROOT_DIR = r"C:\Users\21048\Desktop\The_model_for_garbage_classification\DATASET\test" 

# 2. 输出的 json 文件名
OUTPUT_FILE = "garbage_dataset——test.json"

# 3. 你想让以什么样的问题来提问？(可以多写几个，随机选，增加泛化性)
PROMPTS = [
    "这张图片里是什么垃圾？",
    "<image>\n请对这个垃圾进行分类。",
    "图片中的物品属于什么垃圾类别？",
    "<image>\n这是什么？",
    "帮我识别一下图中的垃圾。"
]

# ===========================================

def create_dataset():
    dataset_data = []
    
    # 支持的图片格式
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    print(f"正在遍历目录: {ROOT_DIR} ...")

    # os.walk 会遍历所有子文件夹
    for root, dirs, files in os.walk(ROOT_DIR):
        for file in files:
            # 检查后缀名
            ext = os.path.splitext(file)[1].lower()
            if ext in valid_extensions:
                
                # 1. 获取图片的绝对路径
                abs_image_path = os.path.join(root, file)
                
                # [关键] 为了让 LLaMA-Factory 更好读取，建议把路径里的反斜杠 \ 换成斜杠 /
                abs_image_path = abs_image_path.replace("\\", "/")

                # 2. 获取分类名称 (假设分类名就是父文件夹的名字)
                # 例如: .../厨余垃圾_苹果/001.jpg -> class_name = "厨余垃圾_苹果"
                class_name = os.path.basename(root)
                
                # 3. 构建对话数据 (ShareGPT 格式)
                # 必须包含 <image> 占位符，通常放在用户提问里
                question = random.choice(PROMPTS)
                if "<image>" not in question:
                    question = "<image>\n" + question
                
                entry = {
                    "image": abs_image_path, # 图片路径
                    "conversations": [
                        {
                            "from": "human",
                            "value": question
                        },
                        {
                            "from": "gpt",
                            "value": f"这是{class_name}。" # 模型的回答
                        }
                    ]
                }
                
                dataset_data.append(entry)

    # 保存 JSON 文件
    if dataset_data:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(dataset_data, f, ensure_ascii=False, indent=2)
        print(f"\n成功生成数据集！")
        print(f"共包含 {len(dataset_data)} 张图片。")
        print(f"文件已保存为: {os.path.abspath(OUTPUT_FILE)}")
    else:
        print("\n警告: 没有找到任何图片，请检查 ROOT_DIR 路径是否正确。")

if __name__ == "__main__":
    create_dataset()