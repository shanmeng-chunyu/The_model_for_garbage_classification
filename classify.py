import argparse
import json
import os

import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image

from data_process import ImagePreprocessor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_unique_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return path
    else:
        i = 1
        while True:
            new_path = f"{path}_{i}"
            if not os.path.exists(new_path):
                os.makedirs(new_path)
                return new_path
            i += 1

def classify(args):
    #设置保存路径
    output_dir = create_unique_dir(args.output_dir)
    output_file_path = os.path.join(output_dir, "classification.txt")

    #加载类别
    with open(args.class_file, 'r', encoding='utf-8') as f:
        # 读取所有行，并用strip()移除每行末尾的换行符，同时过滤掉空行
        class_names = [line.strip() for line in f if line.strip()]
        class_names.sort()
    num_classes = len(class_names)
    transform = ImagePreprocessor([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #加载模型
    net = models.vit_b_16()
    num_head = net.heads.head.in_features
    net.heads.head = nn.Linear(num_head, num_classes)
    net.to(device)
    net.load_state_dict(torch.load(args.model_path, map_location=device))
    net.eval()

    #进行分类
    with open(output_file_path, "w", encoding="utf-8") as f:
        #图片预处理
        for filename in os.listdir(args.source):
            img_path = os.path.join(args.source, filename)
            image = Image.open(img_path).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)

            #模型推理
            with torch.no_grad():
                outputs = net(image)
                probability = F.softmax(outputs, dim=1)
                max_prob, predicted = torch.max(probability, 1)
                predicted_class = class_names[predicted.item()]

            #打印，保存结果
            result = f"{filename}的分类结果为：{predicted_class},置信度为：{max_prob.item()*100:.4f}%"
            f.write(result+'\n')
            return predicted_class


SILICONFLOW_API_KEY = os.environ.get("SILICONFLOW_API_KEY")
SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/chat/completions"


def get_description(predicted_class):
    if not SILICONFLOW_API_KEY:
        return "错误"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}"
    }

    prompt = f"""
    你是一位环保科普专家。请注意，接下来你需要针对 {predicted_class} 进行科普。
    请用平实易懂的语言，为我讲解一下这种东西应该如何妥善处理，以及为什么需要这样做。
    如果我已经告诉你了这是什么类别的垃圾，请在介绍时直接把这个垃圾当成这个类别，如果我没告诉你这个垃圾属于什么类别，请你告诉我这个垃圾属于什么类别。
    你的讲解应包含以下几个方面，请分段作答，结构清晰明确，总字数控制在200字以内：
    1.{{垃圾名称}}是什么：首先，用一两句话简单介绍一下这个物品的材质或用途。
    2.丢弃前的准备：在丢弃它之前，我需要做什么准备工作？（例如：是否需要清空、冲洗、保持干燥或完整等）
    3.它的“后续旅程”：它被收走以后，会经历怎样的处理流程？它有什么潜在的回收利用价值吗？
    4.不当处理的后果：如果我没有正确处理它，可能会带来什么不好的后果？
    请直接开始讲解，不要说任何无关的客套话。
"""
    data = {
        "model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 200,
        "temperature": 0.7,
    }
    try:
        response = requests.post(url=SILICONFLOW_API_URL, headers=headers, data=json.dumps(data), timeout=20)
        response.raise_for_status()
        result_json = response.json()
        description = result_json['choices'][0]['message']['content'].strip()
        return description
    except requests.exceptions.RequestException as e:
        print(f"调用LLM API时发生网络错误: {e}")
        return "抱歉，获取详细介绍时遇到了网络问题，请稍后再试。"
    except (KeyError, IndexError) as e:
        print(f"解析LLM API响应时出错: {e}")
        return "抱歉，无法解析来自服务器的详细介绍。"


def main(args):
    predicted_class = classify(args)
    print(f"预测结果为: {predicted_class},是否正确？\n[y/n]")
    truth = input().lower()
    if truth == "y":
        description = get_description(predicted_class)
    elif truth == "n":
        print(f"请输入垃圾名：我们会为您重新分类")
        ground_truth = input()
        description = get_description(ground_truth)
    print(description)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="./image", type=str,help='需要分类的图片所在目录')
    parser.add_argument('--output_dir', default='./run/classify/exp', type=str,help='分类结果保存目录')
    parser.add_argument('--model_path', default='./run/train/exp_ViT3.0_best_highlr/best_model.pth', type=str,
                        help='模型所在目录')
    parser.add_argument('--class_file',default='./class_name.txt',type=str,help='分类标签所在文件')
    args = parser.parse_args()
    main(args)
