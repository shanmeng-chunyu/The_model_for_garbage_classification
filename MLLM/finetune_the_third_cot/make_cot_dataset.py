import json
import os
import random

# ================= 配置区域 =================


TRAIN_DIR = r"C:\Users\21048\Desktop\The_model_for_garbage_classification\DATASET\test"

OUTPUT_FILE = "garbage_dataset_test.json"

PROMPT = "<image>\n请仔细观察这张图片，分析图中物品的特征，并判断它属于哪种垃圾分类。"

random.seed(42)


# ===========================================

def create_cot_dataset():
    if not os.path.exists(TRAIN_DIR):
        print(f"❌ 错误: 找不到文件夹 {TRAIN_DIR}")
        return

    print(f"正在扫描目录: {TRAIN_DIR} ...")

    dataset_data = []
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    for root, dirs, files in os.walk(TRAIN_DIR):
        class_name = os.path.basename(root)

        if root == TRAIN_DIR:
            continue

        if "_" in class_name:
            main_category, item_name = class_name.split("_", 1)
        else:
            main_category = class_name
            item_name = class_name

        # 构建思维链
        PROMPT = "<image>\n请仔细观察这张图片，分析图中物品的特征，并判断它属于哪种垃圾分类。"
        cot_answer = (
            f"通过观察图片，图中展示的物品是{item_name}。\n"
            f"根据垃圾分类标准，{item_name}属于{main_category}。\n"
            f"因此，该图片的垃圾分类结果是：{class_name}。"
        )

        for file in files:
            if os.path.splitext(file)[1].lower() in valid_extensions:
                abs_image_path = os.path.join(root, file)
                abs_image_path = abs_image_path.replace("\\", "/")

                entry = {
                    "image": abs_image_path,
                    "conversations": [
                        {
                            "from": "human",
                            "value": PROMPT
                        },
                        {
                            "from": "gpt",
                            "value": cot_answer
                        }
                    ]
                }

                dataset_data.append(entry)

    print(f"📊 原始数据生成完毕，共 {len(dataset_data)} 条。")
    print("🔀 正在彻底打乱数据集顺序...")

    random.shuffle(dataset_data)

    print("✅ 打乱完成。")

    if dataset_data:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(dataset_data, f, ensure_ascii=False, indent=2)

        print(f"\n🎉 成功生成并打乱 CoT 数据集！")
        print(f"📁 文件已保存为: {os.path.abspath(OUTPUT_FILE)}")

        print("-" * 30)
        print("👀 乱序预览 (前3条):")
        for i in range(min(3, len(dataset_data))):
            print(f"[{i + 1}] {dataset_data[i]['conversations'][1]['value']}")
        print("-" * 30)
        print("👉 下一步：请将此文件复制到 LLaMA-Factory/data 文件夹中覆盖原文件。")
    else:
        print("⚠️ 警告: 没有找到任何图片。")


if __name__ == "__main__":
    create_cot_dataset()
