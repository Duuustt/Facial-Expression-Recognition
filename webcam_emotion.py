import cv2
import torch
import torchvision.transforms as transforms
from emotion_model import CNNEmotionModel
from PIL import Image

# 设备配置和类别标签
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# 加载预训练模型
model = CNNEmotionModel(num_classes=len(class_names)).to(device)
model.load_state_dict(torch.load("emotion_cnn_model.pth", map_location=device))
model.eval()  # 设置为评估模式


def preprocess(frame):
    """实时视频帧预处理"""
    transform = transforms.Compose([
        transforms.Grayscale(),  # 转为单通道
        transforms.Resize((48, 48)),  # 模型输入尺寸
        transforms.ToTensor(),  # 转为张量
        transforms.Normalize((0.5,), (0.5,))  # 标准化
    ])
    # 将OpenCV图像转换为PIL格式并预处理
    image = Image.fromarray(frame)
    return transform(image).unsqueeze(0).to(device)  # 增加批次维度


# 初始化摄像头和Haar级联分类器
cap = cv2.VideoCapture(0)  # 0表示默认摄像头
# 加载OpenCV自带的人脸检测器
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# 实时检测循环
while True:
    ret, frame = cap.read()  # 读取一帧
    if not ret:
        break  # 读取失败时退出

    # 转换为灰度图(人脸检测需要)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 人脸检测(调整参数可优化检测效果)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # 遍历检测到的所有人脸
    for (x, y, w, h) in faces:
        # 提取人脸区域(ROI)
        face = frame[y:y + h, x:x + w]
        # 预处理人脸图像
        face_input = preprocess(face)

        # 情感预测
        with torch.no_grad():
            output = model(face_input)
            pred = torch.argmax(output, 1).item()
            emotion = class_names[pred]

        # 在原始帧上绘制结果
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 人脸框
        # 在框上方显示情感标签
        cv2.putText(frame, emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # 显示实时画面
    cv2.imshow('Webcam Emotion Recognition', frame)

    # 按'q'键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()