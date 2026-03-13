import torch.nn as nn


class CNNEmotionModel(nn.Module):
    def __init__(self, num_classes=7):
        """
        卷积神经网络情感识别模型
        输入: 48x48灰度图像
        输出: 7类情感概率分布

        参数:
            num_classes: 情感分类数量(默认7类)
        """
        super(CNNEmotionModel, self).__init__()

        # 特征提取器 (卷积层)
        self.features = nn.Sequential(
            # 第1卷积块
            nn.Conv2d(1, 64, kernel_size=3, padding=1),  # 输入通道1(灰度), 输出64通道
            nn.BatchNorm2d(64),  # 批量归一化
            nn.LeakyReLU(0.1),  # 带泄漏的ReLU(负斜率0.1)
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 保持通道数不变
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),  # 空间尺寸减半(48->24)

            # 第2卷积块
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 通道数翻倍
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),  # 空间尺寸减半(24->12)

            # 第3卷积块
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 通道数再次翻倍
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2)  # 空间尺寸减半(12->6)
        )

        # 分类器 (全连接层)
        self.classifier = nn.Sequential(
            nn.Flatten(),  # 将3D特征图展平为1D向量
            nn.Linear(256 * 6 * 6, 512),  # 输入特征维度: 256*6*6=9216
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),  # 50%概率丢弃神经元防止过拟合
            nn.Linear(512, 256),  # 压缩特征维度
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)  # 输出层(7个情感类别)
        )

    def forward(self, x):
        """前向传播流程"""
        x = self.features(x)  # 通过卷积层提取特征
        x = self.classifier(x)  # 通过全连接层分类
        return x