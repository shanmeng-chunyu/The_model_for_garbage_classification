import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.metrics import confusion_matrix, recall_score
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

# 引入项目依赖
from data_process import ImagePreprocessor
from dataset import Dataset

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- 绘图字体设置 (解决中文乱码) ---
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


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


def load_and_parse_classes(class_file_path):
    """
    读取并排序类别名，解析大类和小类结构。
    格式要求: [大类_小类]
    """
    if not os.path.exists(class_file_path):
        raise FileNotFoundError(f"找不到类别文件: {class_file_path}")

    with open(class_file_path, 'r', encoding='utf-8') as f:
        # 读取并按字符排序，确保与Dataset加载逻辑一致（假设Dataset也是按文件名排序）
        class_names = sorted([line.strip() for line in f.readlines() if line.strip()])

    grand_classes = []
    sub_classes = []
    grand_map = {}  # 小类索引 -> 大类索引

    # 解析结构
    # 假设格式严格为 [Grand_Sub]
    for idx, name in enumerate(class_names):
        clean_name = name.replace('[', '').replace(']', '')
        parts = clean_name.split('_', 1)  # 只分割第一个下划线

        g_name = parts[0]
        s_name = parts[1] if len(parts) > 1 else parts[0]

        sub_classes.append(s_name)

        if g_name not in grand_classes:
            grand_classes.append(g_name)

        grand_map[idx] = grand_classes.index(g_name)

    return class_names, sub_classes, grand_classes, grand_map


def plot_confusion_matrix(cm, classes, title, save_path, figsize=(10, 8), annot=True):
    """绘制混淆矩阵"""
    plt.figure(figsize=figsize)
    # 对于类别很多的情况，annot=True会导致文字重叠，这里做个自动判断
    do_annot = annot if len(classes) < 20 else False

    sns.heatmap(cm, annot=do_annot, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"已保存: {save_path}")


def plot_top_low_recall(recalls, class_names, save_path, top_k=20):
    """绘制召回率最低的Top-K类别条形图 (修正版)"""
    # 1. 处理可能的 NaN 值 (防止测试集中没有该类别导致报错或画不出图)
    recalls = np.nan_to_num(recalls, nan=0.0)

    # 2. 创建DataFrame并排序
    df = pd.DataFrame({'Class': class_names, 'Recall': recalls})
    # 按召回率升序排序（低的在前），如果一样低，按名字排
    df = df.sort_values(['Recall', 'Class']).head(top_k)

    plt.figure(figsize=(12, 10))

    # 3. 绘制条形图
    # 这里的 data=df 会自动按 df 的顺序从下往上画
    barplot = sns.barplot(x='Recall', y='Class', data=df, palette='Reds_r')

    # 4. 修正后的文字标注逻辑
    # 不使用 row.index，而是使用 enumerate 生成的 0, 1, 2... 绝对坐标
    for i, (index, row) in enumerate(df.iterrows()):
        # i 是当前画图的行号 (0, 1, 2...)
        # row.Recall 是数值

        # 如果召回率太小(比如0)，文字稍微往右挪一点，防止和Y轴重叠
        text_x_pos = row.Recall + 0.01

        plt.text(text_x_pos, i, f"{row.Recall:.2%}",
                 color='black', va="center", fontsize=10)

    plt.title(f'Top-{top_k} Lowest Recall Categories (需要重点优化的类别)')
    plt.xlabel('Recall Rate')
    plt.xlim(0, 1.15)  # 稍微加宽X轴，给右边的文字留出空间
    plt.grid(axis='x', linestyle='--', alpha=0.5)  # 加个虚线网格，方便看对应的刻度
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"已保存: {save_path}")


def main(args):
    # 1. 准备目录
    save_dir = create_unique_dir(args.save_dir)
    print(f"结果将保存至: {save_dir}")

    # 2. 加载类别信息
    class_file = os.path.join(args.data_dir, 'class_name.txt')  # 假设文件名
    # 如果找不到默认文件，尝试从args传入（这里简化处理，假设一定有这个文件）
    # 在实际使用中，你可能需要手动创建一个包含所有类名的txt，或者修改此处代码适配你的目录结构
    if not os.path.exists(class_file):
        # 如果没有class_name文件，尝试直接读取train目录下的文件夹名
        print(f"警告: 未找到 {class_file}，将尝试从目录结构推断类别...")
        train_dir = os.path.join(args.data_dir, 'train')
        temp_classes = sorted(os.listdir(train_dir))
        # 临时写入一个文件以便复用逻辑（或者直接修改逻辑）
        class_file = os.path.join(save_dir, 'temp_classes.txt')
        with open(class_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(temp_classes))

    full_class_names, sub_class_names, grand_class_names, grand_map = load_and_parse_classes(class_file)
    num_classes = len(full_class_names)
    print(f"共检测到 {num_classes} 个细分类别，归属于 {len(grand_class_names)} 个大类。")

    # 3. 数据加载
    test_transform = ImagePreprocessor([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    test_dataset = Dataset(root=os.path.join(args.data_dir, 'test'), transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # 4. 模型加载
    print(f"加载模型: {args.model_path}")
    Net = models.vit_b_16()
    num_heads = Net.heads.head.in_features
    Net.heads.head = nn.Linear(num_heads, num_classes)
    Net.load_state_dict(torch.load(args.model_path, map_location=device))
    Net.to(device)
    Net.eval()

    # 5. 推理循环
    y_true = []
    y_pred = []

    print("开始推理...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)  # labels 是索引

            with autocast():
                outputs = Net(inputs)
                _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 6. 计算小类别指标
    print("正在计算小类别指标...")
    # 计算每个类别的召回率
    # recall_score returns array of shape (n_classes,)
    sub_recalls = recall_score(y_true, y_pred, average=None, labels=range(num_classes))

    # 绘制 Top-20 低召回率图
    plot_top_low_recall(sub_recalls, sub_class_names,
                        os.path.join(save_dir, 'top20_low_recall.png'))

    # 绘制小类别混淆矩阵 (尺寸设置大一些)
    sub_cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(sub_cm, sub_class_names, "Sub-Category Confusion Matrix",
                          os.path.join(save_dir, 'sub_confusion_matrix.png'),
                          figsize=(24, 20), annot=False)  # 关闭annot防止数字重叠

    # 7. 计算大类别指标
    print("正在计算大类别指标...")
    # 将预测结果和真实标签映射到大类索引
    y_true_grand = np.array([grand_map[idx] for idx in y_true])
    y_pred_grand = np.array([grand_map[idx] for idx in y_pred])

    grand_cm = confusion_matrix(y_true_grand, y_pred_grand)

    # 绘制大类别混淆矩阵
    plot_confusion_matrix(grand_cm, grand_class_names, "Grand-Category Confusion Matrix",
                          os.path.join(save_dir, 'grand_confusion_matrix.png'),
                          figsize=(10, 8), annot=True)

    # 保存详细文本报告
    with open(os.path.join(save_dir, 'recall_report.txt'), 'w', encoding='utf-8') as f:
        f.write("Class Name\tRecall\n")
        for name, rec in zip(full_class_names, sub_recalls):
            f.write(f"{name}\t{rec:.4f}\n")

    print("所有图表已生成完毕。")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="垃圾分类模型召回率深度分析")
    parser.add_argument('--data_dir', type=str, default='../垃圾图片库',
                        help='包含 class_name.txt 和 test 文件夹的目录')
    parser.add_argument('--model_path', type=str,
                        default='./run/train/exp_ViT3.0_best_highlr/best_model.pth',
                        help='模型权重路径')
    parser.add_argument('--save_dir', type=str, default='./run/recall',
                        help='结果保存目录')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=256)

    args = parser.parse_args()
    main(args)
