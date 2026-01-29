import json
import os
import random

# ================= 配置区域 =================


TRAIN_DIR = r"The_model_for_garbage_classification\DATASET\test"

OUTPUT_FILE = "garbage_structured_dataset_test.json"


# ===========================================

def create_structured_dataset():
    if not os.path.exists(TRAIN_DIR):
        print(f"❌ 错误: 找不到文件夹 {TRAIN_DIR}")
        return

    print(f"正在扫描目录: {TRAIN_DIR} ...")

    dataset_data = []
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    system_instruction = (
        "<image>\n"
        "请识别图中的垃圾物品。\n"
        "输出回答时，请严格按照下面我给出的格式进行回答，不要包含任何多余的解释：\n"
        "图中展示的物品是{garbage_object},根据垃圾分类的标准，应该属于{garbage_type}"
    )

    for root, dirs, files in os.walk(TRAIN_DIR):

        folder_name = os.path.basename(root)

        if root == TRAIN_DIR:
            continue

        if "_" in folder_name:
            garbage_type, garbage_class = folder_name.split("_", 1)
        else:

            garbage_type = folder_name
            garbage_class = folder_name

        system_instruction = (
            "<image>\n"
            "请识别图中的垃圾物品。\n"
            "输出回答时，请严格按照下面我给出的格式进行回答，不要包含任何多余的解释：\n"
            "图中展示的物品是{garbage_object},根据垃圾分类的标准，应该属于{garbage_type}"
        )
        target_answer = f"图中展示的物品是{garbage_class},根据垃圾分类的标准，应该属于{garbage_type}"

        for file in files:
            if os.path.splitext(file)[1].lower() in valid_extensions:
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

    print(f"📊 共生成 {len(dataset_data)} 条数据，正在打乱顺序...")
    random.seed(42)
    random.shuffle(dataset_data)

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
