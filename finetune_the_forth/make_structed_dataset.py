import os
import json
import random

# ================= 配置区域 =================

# 1. 图片数据的根目录 (保持你之前的路径)
TRAIN_DIR = r"C:\Users\21048\Desktop\The_model_for_garbage_classification\DATASET\train"

# 2. 输出的数据集文件名
OUTPUT_FILE = "garbage_structured_dataset.json"

# ===========================================

def create_structured_dataset():
    if not os.path.exists(TRAIN_DIR):
        print(f"❌ 错误: 找不到文件夹 {TRAIN_DIR}")
        return

    print(f"正在扫描目录: {TRAIN_DIR} ...")
    
    dataset_data = []
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    # --- 1. 定义固定的 Prompt 模板 ---
    # 这里我们明确告诉模型：你要按照这个格式填空
    # 注意：为了让模型理解 {class} 是占位符，我们在 Prompt 里保留花括号或者用文字说明
    system_instruction = (
        "<image>\n"
        "请识别图中的垃圾物品。\n"
        "输出回答时，请严格按照下面我给出的格式进行回答，不要包含任何多余的解释：\n"
        "图中展示的物品是{garbage_object},根据垃圾分类的标准，应该属于{garbage_type}"
    )

    # 遍历文件夹
    for root, dirs, files in os.walk(TRAIN_DIR):
        # 获取文件夹名称 (例如 "厨余垃圾_巴旦木")
        folder_name = os.path.basename(root)
        
        if root == TRAIN_DIR:
            continue

        # --- 2. 解析语义 (拆分 type 和 class) ---
        # 格式示例： "厨余垃圾_巴旦木" -> type="厨余垃圾", class="巴旦木"
        if "_" in folder_name:
            garbage_type, garbage_class = folder_name.split("_", 1)
        else:
            # 防御性处理：如果没有下划线，暂且认为它既是类也是名
            garbage_type = folder_name
            garbage_class = folder_name

        # --- 3. 构建标准答案 ---
        # 这里我们将真实的 type 和 class 填入模板，作为训练的目标
        target_answer = f"图中展示的物品是{garbage_class},根据垃圾分类的标准，应该属于{garbage_type}"

        # --- 4. 生成数据条目 ---
        for file in files:
            if os.path.splitext(file)[1].lower() in valid_extensions:
                # 获取绝对路径并修复反斜杠
                abs_image_path = os.path.join(root, file)
                abs_image_path = abs_image_path.replace("\\", "/")
                
                entry = {
                    "image": abs_image_path,
                    "conversations": [
                        {
                            "from": "human",
                            "value": system_instruction
                        },
                        {
                            "from": "gpt",
                            "value": target_answer
                        }
                    ]
                }
                
                dataset_data.append(entry)

    # --- 5. 打乱数据 (非常重要！) ---
    print(f"📊 共生成 {len(dataset_data)} 条数据，正在打乱顺序...")
    random.seed(42) 
    random.shuffle(dataset_data)

    # --- 6. 保存文件 ---
    if dataset_data:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(dataset_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n🎉 成功生成结构化数据集！")
        print(f"📁 文件已保存为: {os.path.abspath(OUTPUT_FILE)}")
        print("-" * 30)
        print("👀 数据样例预览 (模型将学习输出这种格式):")
        print(f"问: {dataset_data[0]['conversations'][0]['value']}")
        print(f"答: {dataset_data[0]['conversations'][1]['value']}")
        print("-" * 30)
        print("👉 请将此文件复制到 LLaMA-Factory/data 文件夹中覆盖原文件。")
    else:
        print("⚠️ 警告: 没有找到任何图片。")

if __name__ == "__main__":
    create_structured_dataset()