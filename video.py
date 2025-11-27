import cv2
from transformers import AutoModelForCausalLM,AutoTokenizer
import torchvision.models as models
import torch
import torch.nn as nn
import transformers
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont 
import numpy as np
import os
type_path=r"class_name.txt"
model_weight_path=r"best_model.pth"
model=models.vit_b_16(weights=None)
NUM_CLASS=214
my_classes=[]
with open(type_path,'r',encoding='utf-8') as f:
    for line in f:
        class_name=line.strip()
    
        if class_name:
            my_classes.append(class_name)
my_classes.sort()

print(f"正在替换分类头以匹配 {NUM_CLASS} 个类别...")
try:
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, NUM_CLASS)
except Exception as e:
    print(f"替换分类头失败: {e}")
    print("请确保你的 torchvision 版本与训练时一致")
    exit()
model.load_state_dict(torch.load(model_weight_path, map_location='cpu'))

print("------已成功加载模型------")

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print(f"--- 模型已准备就绪，运行在 {device} ---")

data_preprocessor=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])

])


font_path = r"C:\Windows\Fonts\simhei.ttf"

if not os.path.exists(font_path):
    print(f"警告: 找不到字体文件 {font_path}。正在尝试 'msyh.ttf' (微软雅黑)...")
    font_path = r"C:\Windows\Fonts\msyh.ttf"
    if not os.path.exists(font_path):
        print("错误: 找不到 'simhei.ttf' 或 'msyh.ttf'。")
        print("请从 C:\\Windows\\Fonts 文件夹中找到一个 .ttf 中文字体并更新 font_path 变量。")
        exit()

font_size = 28
try:
    font = ImageFont.truetype(font_path, font_size)
    print(f"成功加载中文字体: {font_path}")
except Exception as e:
    print(f"加载字体失败: {e}")
    exit()

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    rgb_frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image=Image.fromarray(rgb_frame)
    input_tensor=data_preprocessor(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs=model(input_tensor)
    number=outputs.argmax(1).item()
    display_text = f"Predicted: {my_classes[number]}"
    
    # 2. 将 OpenCV 图像 (BGR) 转换为 PIL 图像 (RGB)
    #    (注意：你之前在推理时已经转过一次，但那是在另一个变量里
    #     我们这里需要用原始的 'frame' 来转换)
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # 3. 创建一个绘图对象
    draw = ImageDraw.Draw(pil_img)
    
    # 4. 在图像上绘制文本 (使用我们加载的字体)
    #    (10, 10) 是坐标, fill=(0, 255, 0) 是 RGB 颜色 (绿色)
    draw.text((10, 10), display_text, font=font, fill=(0, 255, 0))
    
    # 5. 将 PIL 图像 (RGB) 转换回 OpenCV 图像 (BGR) 以便显示
    frame_with_text = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # 6. 显示最终的图像
    cv2.imshow('COLOR Webcam Feed', frame_with_text)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()