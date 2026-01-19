import json
import os


json_file_path = r"C:\Users\21048\Desktop\The_model_for_garbage_classification\LLaMA-Factory\saves\Qwen2-VL-2B-Instruct\lora\eval_results_finetune_structed_test\generated_predictions.jsonl"


output_csv_path = "finetune_extracted_results.csv" 

def extract_and_show():
    if not os.path.exists(json_file_path):
        print(f"错误：找不到文件 {json_file_path}")
        return

    print(f"正在读取文件: {json_file_path} ...\n")
    
    results = []

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
    
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line: continue 
                
                try:
                    
                    data = json.loads(line)
                    
            
                    predict = data.get('predict', '无预测')
                    label = data.get('label', '无标签')
                    
                    
                    results.append({"行号": line_num, "预测": predict, "标签": label})
                    
            
                    if line_num <= 5:
                        print(f"--- 第 {line_num} 条 ---")
                        print(f"🤖 预测: {predict}")
                        print(f"✅ 标签: {label}")
                        
                except json.JSONDecodeError:
                    print(f"警告: 第 {line_num} 行不是有效的 JSON 格式，已跳过。")

    except Exception as e:
        print(f"发生错误: {e}")
        print("尝试作为整个列表读取...")
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
            for i, item in enumerate(data_list):
                predict = item.get('predict', '无预测')
                label = item.get('label', '无标签')
                results.append({"行号": i+1, "预测": predict, "标签": label})


    if results and output_csv_path:
        import csv
        with open(output_csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
            fieldnames = ['行号', '预测', '标签']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            writer.writerows(results)
            
        print(f"\n🎉 成功提取了 {len(results)} 条结果！")
        print(f"文件已保存为: {os.path.abspath(output_csv_path)}")
        print("你可以去文件夹里用 Excel 打开查看。")

if __name__ == "__main__":
    extract_and_show()