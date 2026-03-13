import torch
from torchvision import transforms
from PIL import Image
from emotion_model import CNNEmotionModel

# 设置计算设备(GPU优先)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 情感类别标签(与训练数据顺序一致)
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# 加载预训练模型
model = CNNEmotionModel(num_classes=len(class_names)).to(device)
# 加载模型权重(自动适配CPU/GPU)
model.load_state_dict(torch.load("emotion_cnn_model.pth", map_location=device))
model.eval()  # 设置为评估模式(关闭dropout等训练专用层)


def preprocess_image(image_path):
    """
    图像预处理流程:
    1. 转换为灰度图
    2. 缩放到模型输入尺寸(48x48)
    3. 转换为张量
    4. 标准化(将像素值从[0,1]映射到[-1,1])
    """
    transform = transforms.Compose([
        transforms.Grayscale(),  # 单通道灰度图
        transforms.Resize((48, 48)),  # 统一尺寸
        transforms.ToTensor(),  # 转换为PyTorch张量
        transforms.Normalize((0.5,), (0.5,))  # 标准化: (像素值 - 0.5)/0.5
    ])
    image = Image.open(image_path)  # 加载图像
    # 增加批次维度并发送到设备
    return transform(image).unsqueeze(0).to(device)


def predict_emotion(image_path):
    """预测单张图像的情感类别"""
    input_tensor = preprocess_image(image_path)
    with torch.no_grad():  # 禁用梯度计算(节省内存)
        output = model(input_tensor)  # 模型推理
        pred = torch.argmax(output, 1).item()  # 获取最高概率类别索引
    return class_names[pred]  # 返回类别名称


if __name__ == "__main__":
    """交互式预测循环"""
    while True:
        img_path = input("请输入图像路径：(输入'q'退出)")
        if img_path == 'q':
            break
        emotion = predict_emotion(img_path)
        print(f"识别的情感为：{emotion}")
