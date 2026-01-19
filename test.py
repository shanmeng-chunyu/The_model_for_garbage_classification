import argparse
import os

import torch
import torch.nn as nn
import torchvision.models as models
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

# 假设 data_process.py 和 dataset.py 文件与此脚本在同一目录下
from data_process import ImagePreprocessor
from dataset import Dataset

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def create_unique_dir(path):
    """创建一个唯一的目录，如果已存在则在末尾添加数字后缀。"""
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


def main(args):
    # 创建保存结果的目录
    save_dir = create_unique_dir(args.save_dir)
    result_file = os.path.join(save_dir, 'test_result_ViT.txt')
    print(f"测试结果将保存在: {result_file}")

    # --- 数据预处理和加载 ---
    # 测试集使用与验证集相同的预处理流程，不进行数据增强
    test_transform = ImagePreprocessor([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # 创建dataset
    test_dataset = Dataset(root=os.path.join(args.data_dir, 'test'), transform=test_transform)

    # --- 新增功能: 打印测试图片总数 ---
    print(f"总共找到 {len(test_dataset)} 张图片用于测试。")

    # 创建dataloader
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             pin_memory=True)

    # 从数据集中获取分类数量
    num_classes = len(test_dataset.classes)
    print(f"在测试集中检测到 {num_classes} 个类别。")

    # --- 模型加载 ---
    # 初始化与训练时相同的模型结构
    Net = models.vit_b_16()

    # 修改最后一个全连接层以匹配分类数量
    num_heads = Net.heads.head.in_features
    Net.heads.head = nn.Linear(num_heads, num_classes)

    # 加载训练好的模型权重
    if not os.path.exists(args.model_path):
        print(f"错误: 模型权重文件不存在于 '{args.model_path}'。请检查路径。")
        return

    print(f"正在从 '{args.model_path}' 加载模型权重...")
    Net.load_state_dict(torch.load(args.model_path, map_location=device))
    Net.to(device)
    print("模型加载成功。")

    # 定义损失函数
    criterion = nn.CrossEntropyLoss().to(device)

    # --- 开始测试 ---
    Net.eval()  # 切换到评估模式
    running_loss_test = 0.0
    accuracy_test = 0

    print("\n" + "-" * 10 + " 开始在测试集上评估 " + "-" * 10)
    with torch.no_grad():  # 在评估阶段不计算梯度
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 使用 autocast 进行混合精度推理，与验证过程保持一致
            with autocast():
                outputs = Net(inputs)
                loss = criterion(outputs, labels)

            running_loss_test += loss.item() * inputs.size(0)
            accuracy_test += (outputs.argmax(1) == labels).sum().item()

    # 计算并打印最终的损失和准确率
    final_loss = running_loss_test / len(test_dataset)
    final_accuracy = accuracy_test / len(test_dataset) * 100

    print(f"Test Loss: {final_loss:.4f}")
    print(f"Test Accuracy: {final_accuracy:.4f}%")
    print("-" * 35)

    # 将结果保存到文件
    with open(result_file, 'w') as f:
        f.write("Test Loss\tTest Accuracy\n")
        f.write(f"{final_loss:.4f}\t{final_accuracy:.4f}\n")

    print("评估完成。")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="在测试集上评估已训练的ViT模型")
    parser.add_argument('--data_dir', type=str, default='../垃圾图片库',
                        help='包含train/val/test文件夹的数据集根目录')
    parser.add_argument('--model_path', type=str, default='./run/train/exp_ViT3.0_best_highlr/best_model.pth',
                        help='已训练好的模型权重文件路径')
    parser.add_argument('--save_dir', type=str, default='./run/test/exp',
                        help='保存测试结果文件的目录')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='数据加载使用的工作线程数')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='每个批次的图片数量')

    args = parser.parse_args()
    main(args)
