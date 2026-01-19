import csv
import os

# 结果文件的路径
csv_file = r"C:\Users\21048\Desktop\The_model_for_garbage_classification\judged_results.csv"

def calculate():
    if not os.path.exists(csv_file):
        print("❌ 找不到结果文件，请先运行评估脚本。")
        return

    total = 0
    correct = 0

    print(f"正在统计 {csv_file} ...")

    with open(csv_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            result = row.get("判定", "").strip()
            
        
            total += 1
            
            
            if result == "1" or "正确" in result:
                correct += 1

    if total == 0:
        print("⚠️ 文件是空的，没有数据。")
        return

    accuracy = (correct / total) * 100

    print("=" * 30)
    print(f"📊 最终评估报告")
    print("=" * 30)
    print(f"📥 总样本数:  {total}")
    print(f"✅ 正确数量:  {correct}")
    print(f"❌ 错误数量:  {total - correct}")
    print("-" * 30)
    print(f"🏆 最终准确率: {accuracy:.2f}%")
    print("=" * 30)

if __name__ == "__main__":
    calculate()