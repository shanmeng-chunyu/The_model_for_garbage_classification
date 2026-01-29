import json
import os
import random

ROOT_DIR = r"The_model_for_garbage_classification\DATASET\test"

OUTPUT_FILE = "garbage_dataset——test.json"

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
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    print(f"正在遍历目录: {ROOT_DIR} ...")

    for root, dirs, files in os.walk(ROOT_DIR):
        for file in files:

            ext = os.path.splitext(file)[1].lower()
            if ext in valid_extensions:

                abs_image_path = os.path.join(root, file)

                abs_image_path = abs_image_path.replace("\\", "/")

                class_name = os.path.basename(root)

                question = random.choice(PROMPTS)
                if "<image>" not in question:
                    question = "<image>\n" + question

                entry = {
                    "image": abs_image_path,
                    "conversations": [
                        {
                            "from": "human",
                            "value": question
                        },
                        {
                            "from": "gpt",
                            "value": f"这是{class_name}。"
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
