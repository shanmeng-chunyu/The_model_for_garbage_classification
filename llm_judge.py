import json
import os
import csv
from openai import OpenAI
from tqdm import tqdm

# ================= 配置区域 =================

input_file = r"C:\Users\21048\Desktop\The_model_for_garbage_classification\LLaMA-Factory\saves\Qwen2-VL-2B-Instruct\lora\eval_results\generated_predictions.jsonl"
output_file = "judged_results.csv"

# API 配置 (DeepSeek)
API_KEY = "sk-ff73ef6e14ff49938293f25e7d203228" # <--- 填你的 Key
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"

# 设置为 None 跑全量，设置为数字跑测试
TEST_LIMIT = None 

# ===========================================

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def check_answer_with_llm(label, predict):
    prompt = f"""
你是一个公正的垃圾分类评判员。
请判断【预测值】的意思是否与【标准标签】一致（或属于标签描述的类别）。

标准标签：{label}
预测值：{predict}

判断规则：
1. 如果预测值包含了标签中的核心物品名称，视为正确。
2. 如果预测值是标签的同义词（如"土豆"和"马铃薯"），视为正确。
3. 如果预测值完全错误或不相关，视为错误。

请只回答“是”或“否”，不要包含其他文字。
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一个严格的判卷老师，只输出'是'或'否'。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        return "是" in response.choices[0].message.content.strip()
    except Exception as e:
        print(f" API Error: {e}")
        return False

def get_processed_count(filename):
    """获取已经处理了多少行"""
    if not os.path.exists(filename):
        return 0
    with open(filename, 'r', encoding='utf-8-sig') as f:
        # 减去 1 因为有一行是标题 (Header)
        return sum(1 for row in f) - 1

def main():
    if not os.path.exists(input_file):
        print("错误：找不到输入文件！")
        return

    # 1. 读取所有输入数据
    with open(input_file, 'r', encoding='utf-8-sig') as f: # 注意 utf-8-sig
        all_lines = f.readlines()

    if TEST_LIMIT:
        all_lines = all_lines[:TEST_LIMIT]
    
    total_lines = len(all_lines)
    
    # 2. 检查已经跑了多少
    processed_count = get_processed_count(output_file)
    
    if processed_count >= total_lines:
        print(f"🎉 所有 {total_lines} 条数据都已经评估过了！无需重新运行。")
        return

    print(f"总任务: {total_lines} 条 | 已完成: {processed_count} 条 | 剩余: {total_lines - processed_count} 条")
    print(f"🚀 开始断点续传...\n")

    # 3. 准备写入 (追加模式 'a')
    file_exists = os.path.exists(output_file)
    
    with open(output_file, 'a', newline='', encoding='utf-8-sig') as csvfile:
        fieldnames = ["行号", "标签", "预测", "判定"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # 如果是新文件，写入表头
        if not file_exists or os.path.getsize(output_file) == 0:
            writer.writeheader()

        # 4. 跳过已处理的，处理剩下的
        # 使用 tqdm 显示进度
        for i in tqdm(range(processed_count, total_lines), desc="评估进度"):
            line = all_lines[i].strip()
            if not line: continue
            
            data = json.loads(line)
            label = data.get('label', '')
            predict = data.get('predict', '')
            
            # 调用 API
            is_correct = check_answer_with_llm(label, predict)
            
            # 立即写入文件 (防止崩溃丢失)
            clean_label = label.replace("\n", "").replace("\r", "").strip()
            clean_predict = predict.replace("\n", " ").replace("\r", " ").strip()
            # ------------------------------------

            # 立即写入文件 (防止崩溃丢失)
            writer.writerow({
                "行号": i + 1,
                "标签": clean_label,   # 使用清洗后的标签
                "预测": clean_predict, # 使用清洗后的预测
                "判定": "1" if is_correct else "0"
            })
            csvfile.flush() # 强制刷新缓存到硬盘

    print("\n✅ 所有评估完成！")
    print(f"结果已保存到: {output_file}")

if __name__ == "__main__":
    main()