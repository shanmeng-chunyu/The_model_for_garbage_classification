import argparse

import torch
import torch.optim as optim
import os
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

import torchvision.models as models
from data_process import ImagePreprocessor
from data_process import ImageAugmentor
from dataset import Dataset

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

def main(args):
    #创建目录，防止重名
    save_dir = create_unique_dir(args.save_dir)
    # 初始化SummaryWriter，数据可视化
    writer = SummaryWriter(log_dir=os.path.join(save_dir, 'logs'))
    #创建结果保存的txt文件
    result_file = os.path.join(save_dir, 'result.txt')
    with open(result_file, 'w') as f:
        f.write("Epoch\tTrain Loss\tTrain Accuracy\tValid Loss\tValid Accuracy\n")
    # 图片预处理
    val_transform = ImagePreprocessor([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_transform = ImageAugmentor([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # 创建dataset
    train_dataset = Dataset(root=os.path.join(args.data_dir, 'train'),transform=train_transform)
    val_dataset = Dataset(root=os.path.join(args.data_dir, 'val'), transform=val_transform)
    # 创建dataloader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)
    # 记录分类类别
    num_classes = len(train_dataset.classes)
    # 加载模型
    Net = models.resnet50(pretrained=True)
    # 修改最后一个全连接层以匹配分类数量
    num_ftrs = Net.fc.in_features
    Net.fc = nn.Linear(num_ftrs, num_classes)
    Net.to(device)
    # 显示模型结构
    inputs_example, _ = next(iter(train_loader))
    writer.add_graph(Net, inputs_example.to(device))
    # 定义损失函数
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)
    # 定义优化器
    # 将参数分为两组,不同层用不同学习率
    base_params = []
    fc_params = []
    for name, param in Net.named_parameters():
        if 'fc' in name:
            fc_params.append(param)
        else:
            base_params.append(param)
    optimizer = optim.Adam([
        {'params': base_params, 'lr': args.lr},
        {'params': fc_params, 'lr': args.lr * 10}
    ], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=1e-6)
    # 初始化 GradScaler
    scaler = GradScaler()
    # 开始训练
    best_val_acc = 0
    for epoch in range(args.epochs):
        Net.train()
        running_loss = 0.0
        accuracy = 0
        print('-' * 10 + f"Epoch {epoch + 1}" + '-' * 10)
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = Net(inputs)
            loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * inputs.size(0)
            accuracy += (outputs.argmax(1) == labels).sum().item()
        epoch_loss_train = running_loss / len(train_dataset)
        epoch_accuracy_train = accuracy / len(train_dataset) * 100
        print(f"Train Loss:{epoch_loss_train:.4f}\nTrain Accuracy:{epoch_accuracy_train:.4f}%")
        writer.add_scalar('Loss/train', epoch_loss_train, epoch)
        writer.add_scalar('Accuracy/train', epoch_accuracy_train, epoch)

        Net.eval()
        running_loss_val = 0.0
        accuracy_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = Net(inputs)
                loss = criterion(outputs, labels)
                running_loss_val += loss.item() * inputs.size(0)
                accuracy_val += (outputs.argmax(1) == labels).sum().item()
            epoch_loss_val = running_loss_val / len(val_dataset)
            epoch_accuracy_val = accuracy_val / len(val_dataset) * 100
            print(f"Val Loss:{epoch_loss_val:.4f}\nVal Accuracy:{epoch_accuracy_val:.4f}%")
            writer.add_scalar('Loss/val', epoch_loss_val, epoch)
            writer.add_scalar('Accuracy/val', epoch_accuracy_val, epoch)
        with open(result_file, "a") as f:
            f.write(
                f"{epoch + 1}\t{epoch_loss_train:.4f}\t{epoch_accuracy_train:.4f}\t{epoch_loss_val:.4f}\t{epoch_accuracy_val:.4f}\n")
        scheduler.step()
        if epoch_accuracy_val > best_val_acc:
            best_val_acc = epoch_accuracy_val
            model_save_path = os.path.join(save_dir, "best_model.pth")
            torch.save(Net.state_dict(), model_save_path)
            print(f"Best val Accuracy: {best_val_acc:.4f}%")
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../垃圾图片库')
    parser.add_argument('--save_dir', type=str, default='./run/train/exp')
    parser.add_argument('--num_workers', type=int, default=24)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()
    main(args)
