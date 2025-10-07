import argparse
import os

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
    net = models.resnet50().to(device)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, num_classes)
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
            print(result)
            f.write(result+'\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="./image", type=str,help='需要分类的图片所在目录')
    parser.add_argument('--output_dir', default='./run/classify/exp', type=str,help='分类结果保存目录')
    parser.add_argument('--model_path', default='./run/train/exp_best/best_model.pth', type=str, help='模型所在目录')
    parser.add_argument('--class_file',default='./class_name.txt',type=str,help='分类标签所在文件')
    args = parser.parse_args()
    classify(args)