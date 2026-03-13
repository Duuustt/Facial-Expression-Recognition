import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from emotion_model import CNNEmotionModel

# 数据集路径配置
train_dir = "train"  # 训练集目录
test_dir = "test"  # 测试集目录

# 训练配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 70  # 训练轮次
batch_size = 64  # 批次大小
learning_rate = 0.001  # 学习率
classes = os.listdir(train_dir)  # 从目录结构获取类别列表

# 数据增强策略(训练集/测试集不同)
data_transforms = {
    'train': transforms.Compose([
        transforms.Grayscale(),  # 灰度转换
        transforms.Resize((48, 48)),  # 统一尺寸
        transforms.RandomHorizontalFlip(),  # 随机水平翻转(增强多样性)
        transforms.RandomRotation(15),  # 随机旋转(±15度)
        transforms.RandomAffine(  # 随机平移(10%范围内)
            degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize((0.5,), (0.5,))  # 标准化
    ]),
    'test': transforms.Compose([  # 测试集只需基础转换
        transforms.Grayscale(),  # 灰度转换
        transforms.Resize((48, 48)),  # 统一尺寸
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize((0.5,), (0.5,))  # 标准化
    ])
}

# 创建数据集和数据加载器
datasets = {
    'train': ImageFolder(train_dir, transform=data_transforms['train']),
    'test': ImageFolder(test_dir, transform=data_transforms['test'])
}
dataloaders = {
    'train': DataLoader(
        datasets['train'],
        batch_size=batch_size,
        shuffle=True),  # 训练集需要打乱顺序
    'test': DataLoader(
        datasets['test'],
        batch_size=batch_size,
        shuffle=False)  # 测试集保持原始顺序
}

# 初始化模型
model = CNNEmotionModel(num_classes=len(classes)).to(device)
criterion = nn.CrossEntropyLoss()  # 分类任务常用损失函数
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam优化器
# 余弦退火学习率调度器(每10个epoch重置学习率)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# 训练指标记录
train_accs, test_accs, train_losses, test_losses = [], [], [], []

# 训练循环
for epoch in range(epochs):
    # === 训练阶段 ===
    model.train()  # 设置训练模式
    total_loss, correct, total = 0, 0, 0
    for inputs, labels in dataloaders['train']:
        inputs, labels = inputs.to(device), labels.to(device)

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 计算梯度
        optimizer.step()  # 更新权重

        # 统计指标
        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)  # 获取预测类别
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    # 计算本轮训练指标
    train_acc = correct / total
    train_loss = total_loss / len(dataloaders['train'])
    train_accs.append(train_acc)
    train_losses.append(train_loss)

    # === 验证阶段 ===
    model.eval()  # 设置评估模式
    correct, total, test_loss = 0, 0, 0
    all_preds, all_labels = [], []  # 记录所有预测/真实标签(用于混淆矩阵)
    with torch.no_grad():  # 禁用梯度计算
        for inputs, labels in dataloaders['test']:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            # 收集预测结果
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算本轮验证指标
    test_acc = correct / total
    test_losses.append(test_loss / len(dataloaders['test']))
    test_accs.append(test_acc)

    # 更新学习率
    scheduler.step()

    # 打印进度
    print(f"Epoch [{epoch + 1}/{epochs}] Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

# === 训练后处理 ===
# 保存模型权重
torch.save(model.state_dict(), "emotion_cnn_model.pth")

# 绘制训练曲线
plt.figure(figsize=(12, 5))
# 准确率曲线
plt.subplot(1, 2, 1)
plt.plot(train_accs, label='Train Acc')
plt.plot(test_accs, label='Test Acc')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# 损失曲线
plt.subplot(1, 2, 2)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("training_curves.png")  # 保存图像
plt.show()

# 生成混淆矩阵
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(xticks_rotation=45)  # 旋转x轴标签
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
