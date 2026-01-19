# 首先在导入部分添加必要的库
import json
import os
import threading
import warnings
from tkinter import filedialog, messagebox

import customtkinter as ctk
# 新增: 导入视频处理所需的库
import cv2
import numpy as np
import requests
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageTk, ImageDraw, ImageFont
# 添加LLM相关导入
from transformers import AutoTokenizer, AutoModelForCausalLM

from data_process import ImagePreprocessor
from model.model_lora import *
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from out.trainer_utils import setup_seed

warnings.filterwarnings('ignore')

# 设置CustomTkinter主题
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

# LLM API 配置
# SILICONFLOW_API_KEY = os.environ.get("SILICONFLOW_API_KEY")
SILICONFLOW_API_KEY = "sk-yjntmvfcbgfrdojoyrmukreyszxrshxprlctsbqfobalzcye"
SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/chat/completions"


class ImageClassifierGUI:
    # 在初始化方法中添加变量（大约在第27-70行左右的__init__方法中）
    def __init__(self):
        # 初始化主窗口
        self.root = ctk.CTk()
        self.root.title("AI 智能垃圾分类系统")

        # 设置窗口最小尺寸
        self.root.minsize(1100, 700)

        # 居中显示窗口
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        width = 1200
        height = 800
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        self.root.geometry(f"{width}x{height}+{x}+{y}")

        # 全局字体族
        self.font_family = "Microsoft YaHei UI" if os.name == "nt" else "PingFang SC"
        self.font_main = (self.font_family, 13)
        self.font_bold = (self.font_family, 13, "bold")
        self.font_title = (self.font_family, 22, "bold")

        # 模型相关变量
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_type = None
        self.class_names = []
        self.transform = None
        self.is_model_loaded = False
        self.model_path = "./run/train/exp_ViT3.0_best_highlr/best_model.pth"

        # LLM相关变量
        self.llm_model = None
        self.llm_tokenizer = None
        self.llm_initialized = False
        self.llm_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 图像相关变量
        self.current_image_path = None
        self.current_image_display = None

        # 视频识别相关变量
        self.is_video_running = False
        self.video_thread = None
        self.cap = None
        self.last_video_prediction = None
        self.last_video_confidence = 0

        # 垃圾介绍生成方式选项
        self.description_mode = "api"

        # 创建UI组件
        self.create_widgets()

        # 绑定窗口大小变化事件
        self.root.bind("<Configure>", self.on_window_resize)

        # 启动时加载模型
        self.load_model_on_startup()

    def create_widgets(self):
        """创建现代化UI组件"""
        # 配置网格布局
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        # --- 左侧侧边栏 ---
        self.sidebar = ctk.CTkFrame(self.root, width=260, corner_radius=0, fg_color="#F8F9FA")
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(12, weight=1)  # 底部留白

        self.logo_label = ctk.CTkLabel(
            self.sidebar,
            text="♻️ 垃圾分类助手",
            font=ctk.CTkFont(family=self.font_family, size=20, weight="bold"),
            text_color="#1A1A1A"
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(30, 20))

        # 操作按钮组
        self.btn_select = ctk.CTkButton(
            self.sidebar, text="📁 选择图片",
            font=self.font_bold, height=40, corner_radius=8,
            command=self.select_image, fg_color="#007AFF", hover_color="#005BB5"
        )
        self.btn_select.grid(row=1, column=0, padx=20, pady=8, sticky="ew")

        self.btn_video = ctk.CTkButton(
            self.sidebar, text="🎥 视频识别",
            font=self.font_bold, height=40, corner_radius=8,
            command=self.toggle_video_recognition, fg_color="#34C759", hover_color="#28A745",
            state="disabled"
        )
        self.btn_video.grid(row=2, column=0, padx=20, pady=8, sticky="ew")

        # 将开始分类按钮下移并增加间距
        self.btn_classify = ctk.CTkButton(
            self.sidebar, text="🔍 开始分类分析",
            font=self.font_bold, height=45, corner_radius=8,
            command=self.classify_image, state="disabled",
            fg_color="#5856D6", hover_color="#4846B4"
        )
        self.btn_classify.grid(row=4, column=0, padx=20, pady=(40, 10), sticky="ew")

        # 分隔线
        self.sep1 = ctk.CTkFrame(self.sidebar, height=1, fg_color="#E5E5E7")
        self.sep1.grid(row=5, column=0, padx=20, pady=20, sticky="ew")

        # 模型设置
        self.model_label = ctk.CTkLabel(self.sidebar, text="模型配置", font=self.font_bold, text_color="#1A1A1A")
        self.model_label.grid(row=6, column=0, padx=20, pady=(0, 5), sticky="w")

        # 修复更换模型按钮显示和颜色
        self.btn_change_model = ctk.CTkButton(
            self.sidebar, text="更换模型文件",
            font=(self.font_family, 12), height=32, corner_radius=6,
            command=self.select_model,
            fg_color="#E5E5E7", text_color="#1A1A1A", hover_color="#D1D1D6"
        )
        self.btn_change_model.grid(row=7, column=0, padx=20, pady=5, sticky="ew")

        self.model_path_label = ctk.CTkLabel(
            self.sidebar, text="", font=(self.font_family, 10),
            text_color="#8E8E93", wraplength=200
        )
        self.model_path_label.grid(row=8, column=0, padx=20, pady=(2, 10), sticky="ew")

        # 介绍模式
        self.mode_label = ctk.CTkLabel(self.sidebar, text="垃圾介绍生成模式", font=self.font_bold, text_color="#1A1A1A")
        self.mode_label.grid(row=9, column=0, padx=20, pady=(10, 5), sticky="w")

        self.description_mode_var = ctk.StringVar(value="api")
        self.radio_api = ctk.CTkRadioButton(
            self.sidebar, text="在线 API (推荐)", variable=self.description_mode_var,
            value="api", command=self.on_description_mode_change, font=(self.font_family, 12),
            fg_color="#007AFF", border_color="#8E8E93"
        )
        self.radio_api.grid(row=10, column=0, padx=30, pady=5, sticky="w")

        self.radio_local = ctk.CTkRadioButton(
            self.sidebar, text="本地模型 (离线)", variable=self.description_mode_var,
            value="local", command=self.on_description_mode_change, font=(self.font_family, 12),
            fg_color="#007AFF", border_color="#8E8E93"
        )
        self.radio_local.grid(row=11, column=0, padx=30, pady=5, sticky="w")

        # 状态栏 (底部)
        self.status_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.status_frame.grid(row=13, column=0, padx=20, pady=20, sticky="ew")

        self.status_label = ctk.CTkLabel(
            self.status_frame, text="🔄 正在初始化...",
            font=(self.font_family, 11), text_color="#8E8E93", wraplength=200
        )
        self.status_label.pack(side="bottom", fill="x")

        # --- 右侧主内容区 ---
        self.main_content = ctk.CTkFrame(self.root, fg_color="#FFFFFF")
        self.main_content.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        self.main_content.grid_columnconfigure(0, weight=3)
        self.main_content.grid_columnconfigure(1, weight=2)
        self.main_content.grid_rowconfigure(0, weight=1)

        # 图像显示卡片
        self.image_card = ctk.CTkFrame(self.main_content, corner_radius=12, fg_color="#F2F2F7")
        self.image_card.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)

        self.image_label = ctk.CTkLabel(
            self.image_card, text="请选择图片或开启视频识别",
            font=(self.font_family, 15), text_color="#8E8E93"
        )
        self.image_label.pack(fill="both", expand=True, padx=20, pady=20)

        # 结果显示卡片
        self.result_card = ctk.CTkFrame(self.main_content, corner_radius=12, fg_color="#F2F2F7")
        self.result_card.grid(row=0, column=1, sticky="nsew", padx=(0, 20), pady=20)

        self.res_title = ctk.CTkLabel(self.result_card, text="分析结果", font=self.font_title, text_color="#1A1A1A")
        self.res_title.pack(pady=(30, 15))

        # 类别结果
        self.class_result_label = ctk.CTkLabel(
            self.result_card, text="等待分析...",
            font=(self.font_family, 22, "bold"), text_color="#007AFF"
        )
        self.class_result_label.pack(pady=10)

        # 置信度
        self.conf_frame = ctk.CTkFrame(self.result_card, fg_color="transparent")
        self.conf_frame.pack(fill="x", padx=40, pady=10)

        self.confidence_bar = ctk.CTkProgressBar(self.conf_frame, height=10, corner_radius=5, fg_color="#E5E5E7")
        self.confidence_bar.pack(fill="x", pady=(0, 5))
        self.confidence_bar.set(0)

        self.confidence_text = ctk.CTkLabel(self.conf_frame, text="置信度: 0%", font=(self.font_family, 12),
                                            text_color="#1A1A1A")
        self.confidence_text.pack()

        # 详细介绍
        self.desc_label = ctk.CTkLabel(self.result_card, text="处理建议", font=self.font_bold, text_color="#1A1A1A")
        self.desc_label.pack(anchor="w", padx=30, pady=(20, 5))

        self.description_textbox = ctk.CTkTextbox(
            self.result_card, font=(self.font_family, 13),
            corner_radius=10, border_width=0, fg_color="#FFFFFF", text_color="#1A1A1A",
            spacing2=6,  # 行间距
            spacing3=12  # 段落间距
        )
        self.description_textbox.pack(fill="both", expand=True, padx=25, pady=(0, 25))
        self.description_textbox.insert("1.0", "识别完成后，这里将显示详细的垃圾分类处理建议...")
        self.description_textbox.configure(state="disabled")

        # 设备信息
        # device_text = f"🖥️ 设备: {'GPU' if torch.cuda.is_available() else 'CPU'}"
        # self.device_label = ctk.CTkLabel(
        #     self.right_panel,
        #     text=device_text,
        #     font=ctk.CTkFont(family=self.font_family, size=10),
        #     text_color="gray"
        # )
        # self.device_label.pack(side="bottom", pady=(0, 5))

    # 添加窗口大小变化处理方法
    def on_window_resize(self, event=None):
        """处理窗口大小变化事件"""
        # 只有当事件源是主窗口时才处理，避免子组件事件干扰
        if event and event.widget != self.root:
            return

        # 当窗口大小变化时，调整图像显示大小
        if self.current_image_display and self.current_image_path:
            self.display_image(self.current_image_path)

    # 修改display_image方法，使其能够根据当前窗口大小调整图像
    def display_image(self, image_path):
        """显示选中的图片，根据容器大小自动调整"""
        try:
            # 打开并调整图片大小
            image = Image.open(image_path)
            image = image.convert("RGB")

            # 获取 image_card 的实际大小
            self.root.update_idletasks()  # 确保尺寸已更新
            width = self.image_card.winfo_width()
            height = self.image_card.winfo_height()

            # 计算可用显示区域（减去 padding）
            available_width = max(width - 40, 100)
            available_height = max(height - 40, 100)

            # 设置最大显示尺寸
            display_size = (available_width, available_height)
            image.thumbnail(display_size, Image.Resampling.LANCZOS)

            # 转换为PhotoImage
            photo = ImageTk.PhotoImage(image)

            # 更新显示
            self.image_label.configure(image=photo, text="")
            self.current_image_display = photo  # 保持引用

        except Exception as e:
            messagebox.showerror("错误", f"无法加载图片:\n{str(e)}")

    def _shorten_path(self, path, max_length=30):
        """将路径缩短为显示友好的格式"""
        if len(path) <= max_length:
            return path
        parts = path.split(os.sep)
        if len(parts) <= 2:
            return path[-max_length:]
        return "...\\" + os.sep.join(parts[-2:])

    def detect_model_type(self, model_path: str) -> str:
        """根据模型路径检测模型类型"""
        path_lower = model_path.lower()
        if 'vit' in path_lower or 'vision_transformer' in path_lower:
            return 'vit'
        elif 'mobilenet' in path_lower:
            return 'mobilenet'
        elif 'resnet' in path_lower:
            return 'resnet'

        try:
            state_dict = torch.load(model_path, map_location='cpu')
            keys = list(state_dict.keys())
            if any('encoder' in k and 'blocks' in k for k in keys):
                return 'vit'
            elif any('conv2d' in k.lower() and 'bottleneck' in str(keys) for k in keys):
                return 'mobilenet'
            else:
                return 'resnet'
        except:
            return 'resnet'

    def create_model(self, model_type: str, num_classes: int):
        """根据模型类型创建相应的模型架构"""
        if model_type == 'vit':
            model = models.vit_b_16(pretrained=False)
            in_features = model.heads.head.in_features
            model.heads.head = nn.Linear(in_features, num_classes)
            return model
        elif model_type == 'mobilenet':
            model = models.mobilenet_v2(pretrained=False)
            model.classifier[1] = nn.Linear(model.last_channel, num_classes)
            return model
        else:
            model = models.resnet50(pretrained=False)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
            return model

    def load_model_on_startup(self):
        """应用启动时加载模型"""

        def load_model_worker():
            try:
                self.update_status("📚 加载类别名称...")
                class_file = "./class_name.txt"
                with open(class_file, 'r', encoding='utf-8') as f:
                    self.class_names = [line.strip() for line in f if line.strip()]
                    self.class_names.sort()

                num_classes = len(self.class_names)
                self.update_status("🔧 初始化预处理器...")
                self.transform = ImagePreprocessor([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                self.update_status("🔍 检测模型类型...")
                self.model_type = self.detect_model_type(self.model_path)
                self.update_status(f"🤖 加载 {self.model_type.upper()} 模型...")
                self.model = self.create_model(self.model_type, num_classes)
                self.model.to(self.device)
                self.update_status("⚡ 加载模型权重...")
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                self.model.eval()
                self.is_model_loaded = True
                self.update_status(f"✅ {self.model_type.upper()} 模型加载完成")
                self.root.after(0, lambda: self.model_path_label.configure(text=self._shorten_path(self.model_path)))
                self.root.after(0, lambda: self.btn_video.configure(state="normal"))

            except Exception as e:
                self.update_status(f"❌ 模型加载失败: {str(e)}")
                messagebox.showerror("错误", f"模型加载失败:\n{str(e)}")

        threading.Thread(target=load_model_worker, daemon=True).start()

    def update_status(self, message):
        """更新状态显示"""
        self.root.after(0, lambda: self.status_label.configure(text=message))

    def select_image(self):
        """选择图片文件"""
        file_types = [
            ("图像文件", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
            ("所有文件", "*.*")
        ]
        file_path = filedialog.askopenfilename(
            title="选择要分类的图片",
            filetypes=file_types
        )
        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            self.btn_classify.configure(state="normal")
            self.reset_results()

    def reset_results(self):
        """重置分类结果显示"""
        self.class_result_label.configure(text="等待分析...", text_color="#007AFF")
        self.confidence_bar.set(0)
        self.confidence_text.configure(text="置信度: 0%", text_color="#1A1A1A")
        self.description_textbox.configure(state="normal")
        self.description_textbox.delete("1.0", "end")
        self.description_textbox.insert("1.0", "识别完成后，这里将显示详细的垃圾分类处理建议...")
        self.description_textbox.configure(state="disabled")

    def classify_image(self):
        """执行图像分类"""
        if not self.current_image_path and not self.last_video_prediction:
            messagebox.showwarning("警告", "请先选择一张图片或进行视频识别!")
            return
        if not self.is_model_loaded:
            messagebox.showwarning("警告", "模型还未加载完成，请稍候!")
            return
        self.btn_classify.configure(state="disabled", text="🔍 分类中...")
        self.update_status("🔄 正在分析图片...")

        def classify_worker():
            try:
                if self.current_image_path:
                    image = Image.open(self.current_image_path).convert("RGB")
                    image_tensor = self.transform(image).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        outputs = self.model(image_tensor)
                        probability = F.softmax(outputs, dim=1)
                        max_prob, predicted = torch.max(probability, 1)
                        predicted_class = self.class_names[predicted.item()]
                        confidence = max_prob.item() * 100
                else:
                    predicted_class = self.last_video_prediction
                    confidence = self.last_video_confidence

                self.root.after(0, lambda: self.display_results(predicted_class, confidence))

            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("错误", f"分类过程中出错:\n{str(e)}"))
                self.root.after(0, lambda: self.btn_classify.configure(state="normal", text="🔍 开始分类"))

        threading.Thread(target=classify_worker, daemon=True).start()

    def display_results(self, predicted_class, confidence):
        """显示分类结果"""
        self.class_result_label.configure(text=f"{predicted_class}")
        confidence_ratio = confidence / 100
        self.confidence_bar.set(confidence_ratio)
        self.confidence_text.configure(text=f"置信度: {confidence:.2f}%")

        if confidence >= 80:
            color = "#34C759"  # Green
        elif confidence >= 60:
            color = "#FF9500"  # Orange
        else:
            color = "#FF3B30"  # Red

        self.confidence_bar.configure(progress_color=color)
        self.confidence_text.configure(text_color=color)

        self.btn_classify.configure(state="normal", text="🔍 开始分类")
        self.update_status("✅ 分类完成")
        self.get_trash_info(predicted_class)

    def on_description_mode_change(self):
        """处理介绍生成方式变更事件"""
        self.description_mode = self.description_mode_var.get()
        # 如果当前已有分类结果，自动重新获取介绍
        if hasattr(self, 'class_result_label'):
            current_text = self.class_result_label.cget("text")
            if current_text and current_text != "等待分析...":
                self.get_trash_info(current_text)

    def get_trash_info(self, predicted_class):
        """获取垃圾处理介绍 - 根据选择的模式调用不同的方法"""
        if self.description_mode == "api":
            self.get_trash_info_api(predicted_class)
        else:
            self.get_trash_info_local(predicted_class)

    def get_trash_info_api(self, predicted_class):
        """获取垃圾处理介绍 - 调用LLM API"""

        def fetch_description():
            self.root.after(0, lambda: self.update_status("🤖 正在获取AI介绍..."))
            self.root.after(0, lambda: self.update_description("🤖 正在调用API获取垃圾处理介绍，请稍候..."))

            if not SILICONFLOW_API_KEY:
                self.root.after(0, lambda: self.update_description(
                    "错误：未配置API密钥，请设置环境变量 SILICONFLOW_API_KEY"))
                self.root.after(0, lambda: self.update_status("❌ 未配置API密钥"))
                return

            try:
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {SILICONFLOW_API_KEY}"
                }

                garbage = predicted_class
                if '_' in predicted_class:
                    parts = predicted_class.split('_', 1)
                    garbage = parts[1]

                prompt = f"""
你是一位环保科普专家。请注意，接下来你需要针对{garbage}进行科普。
请用平实易懂的语言，为我讲解一下这种东西应该如何妥善处理，以及为什么需要这样做。
你的讲解应包含以下几个方面，请分段但不要分点作答，结构清晰明确，总字数控制在200字以内：
1.{garbage}是什么，属于什么类别的垃圾？
2.丢弃前的准备：在丢弃它之前，我需要做什么准备工作？（例如：是否需要清空、冲洗、保持干燥或完整等）
3.它的"后续旅程"：它被收走以后，会经历怎样的处理流程？它有什么潜在的回收利用价值吗？
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

                response = requests.post(url=SILICONFLOW_API_URL, headers=headers, data=json.dumps(data), timeout=20)
                response.raise_for_status()
                result_json = response.json()
                description = result_json['choices'][0]['message']['content'].strip()

                self.root.after(0, lambda: self.update_description(description))
                self.root.after(0, lambda: self.update_status("✅ 垃圾处理介绍获取完成!"))

            except Exception as e:
                self.root.after(0, lambda: self.update_description(f"获取垃圾处理介绍失败: {str(e)}"))
                self.root.after(0, lambda: self.update_status("❌ 获取垃圾处理介绍失败"))

        threading.Thread(target=fetch_description, daemon=True).start()

    def get_trash_info_local(self, predicted_class):
        """使用本地LLM模型生成垃圾分类介绍（直接集成）"""

        def generate_description():
            try:
                # 更新状态
                self.root.after(0, lambda: self.update_description("🤖 正在加载本地模型..."))

                # 初始化LLM模型（如果尚未初始化）
                if not self.llm_initialized:
                    success = self.init_llm_model()
                    if not success:
                        self.root.after(0, lambda: self.update_status("❌ 本地模型初始化失败"))
                        self.root.after(0,
                                        lambda: self.update_description("无法初始化本地LLM模型，请检查模型文件是否存在"))
                        return

                # 更新状态
                self.root.after(0, lambda: self.update_description("🤖 正在调用本地模型生成垃圾分类介绍..."))

                # 构建prompt
                garbage = predicted_class
                if '_' in predicted_class:
                    parts = predicted_class.split('_', 1)
                    garbage = parts[1]
                prompt = f"""请你从垃圾分类的角度介绍一下{garbage}，以及如何正确处理它。"""

                # 生成回答
                response = self.generate_with_prompt(prompt)

                # 更新UI
                self.root.after(0, lambda: self.update_description(response))
                self.root.after(0, lambda: self.update_status("✅ 本地模型生成介绍完成!"))

            except Exception as e:
                error_msg = f"本地模型生成失败: {str(e)}"
                self.root.after(0, lambda: self.update_description(error_msg))
                self.root.after(0, lambda: self.update_status("❌ 本地模型生成失败"))
                print(f"本地模型错误详情: {str(e)}")

        threading.Thread(target=generate_description, daemon=True).start()

    def init_llm_model(self):
        """初始化LLM模型"""
        try:
            # 设置LLM模型参数
            load_from = 'model'
            save_dir = 'out'
            weight = 'full_sft'
            lora_weight = 'lora_garbage'
            hidden_size = 768
            num_hidden_layers = 16
            use_moe = 0
            inference_rope_scaling = False

            # 初始化tokenizer
            tokenizer = AutoTokenizer.from_pretrained(load_from)

            # 初始化模型
            if 'model' in load_from:
                model = MiniMindForCausalLM(MiniMindConfig(
                    hidden_size=hidden_size,
                    num_hidden_layers=num_hidden_layers,
                    use_moe=bool(use_moe),
                    inference_rope_scaling=inference_rope_scaling
                ))
                moe_suffix = '_moe' if use_moe else ''
                ckp = f'./{save_dir}/{weight}_{hidden_size}{moe_suffix}.pth'
                model.load_state_dict(torch.load(ckp, map_location=self.llm_device), strict=True)
                if lora_weight != 'None':
                    apply_lora(model)
                    load_lora(model, f'./{save_dir}/lora/{lora_weight}_{hidden_size}.pth')
            else:
                model = AutoModelForCausalLM.from_pretrained(load_from, trust_remote_code=True)

            # 移至设备并设置为eval模式
            model = model.eval().to(self.llm_device)

            # 保存模型和tokenizer
            self.llm_model = model
            self.llm_tokenizer = tokenizer
            self.llm_initialized = True

            print(f'LLM模型初始化成功: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M(illion)')
            return True
        except Exception as e:
            print(f'LLM模型初始化失败: {str(e)}')
            self.llm_initialized = False
            return False

    def generate_with_prompt(self, prompt):
        """使用LLM模型生成回答"""
        if not self.llm_initialized:
            return "LLM模型未初始化"
        try:
            setup_seed(2026)

            # 构建对话
            conversation = [{"role": "user", "content": prompt}]

            # 应用chat template
            weight = 'full_sft'  # 与初始化时保持一致
            templates = {"conversation": conversation, "tokenize": False, "add_generation_prompt": True}
            if weight == 'reason':
                templates["enable_thinking"] = True

            # 处理输入
            inputs = self.llm_tokenizer.apply_chat_template(**templates) if weight != 'pretrain' else (
                    self.llm_tokenizer.bos_token + prompt)
            inputs = self.llm_tokenizer(inputs, return_tensors="pt", truncation=True).to(self.llm_device)

            # 生成参数
            max_new_tokens = 8192
            temperature = 0.85
            top_p = 0.85

            # 生成回答
            with torch.no_grad():
                generated_ids = self.llm_model.generate(
                    inputs=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    pad_token_id=self.llm_tokenizer.pad_token_id,
                    eos_token_id=self.llm_tokenizer.eos_token_id,
                    top_p=top_p,
                    temperature=temperature,
                    repetition_penalty=1.0
                )

            # 解码结果
            response = self.llm_tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):],
                                                 skip_special_tokens=True)
            return response
        except Exception as e:
            print(f'生成回答失败: {str(e)}')
            return f"生成回答时出错: {str(e)}"

    def update_description(self, text):
        """更新描述文本框内容，优化分段显示效果"""
        self.description_textbox.configure(state="normal")
        self.description_textbox.delete("1.0", "end")

        # 预处理文本：确保段落之间有统一的换行符，并去除多余空行
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        formatted_text = "\n".join(paragraphs)

        self.description_textbox.insert("1.0", formatted_text)
        self.description_textbox.configure(state="disabled")

    def select_model(self):
        """选择模型文件"""
        file_path = filedialog.askopenfilename(
            title="选择模型文件",
            filetypes=[("模型文件", "*.pth *.pt"), ("所有文件", "*.*")]
        )
        if file_path:
            self.model_path = file_path
            self.update_status("🔄 正在加载新模型...")
            self.load_model_on_startup()

    def toggle_video_recognition(self):
        """切换视频识别状态"""
        if not self.is_video_running:
            self.start_video_recognition()
        else:
            self.stop_video_recognition()

    def start_video_recognition(self):
        """开始视频识别"""
        self.is_video_running = True
        self.btn_video.configure(text="⏹️ 停止识别", fg_color="#F56C6C", hover_color="#D9534F")
        self.btn_select.configure(state="disabled")
        self.btn_change_model.configure(state="disabled")
        self.btn_classify.configure(state="disabled")
        self.update_status("🎥 正在开启摄像头...")

        # 重置结果显示
        self.reset_results()

        # 在新线程中运行视频识别
        self.video_thread = threading.Thread(target=self._video_recognition_worker, daemon=True)
        self.video_thread.start()

    def stop_video_recognition(self):
        """停止视频识别"""
        self.is_video_running = False
        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join(timeout=1.0)

        if self.cap and self.cap.isOpened():
            self.cap.release()

        self.btn_video.configure(text="🎥 视频识别", fg_color="#2FA572", hover_color="#288E62")
        self.btn_select.configure(state="normal")
        self.btn_change_model.configure(state="normal")
        if self.current_image_path or self.last_video_prediction:
            self.btn_classify.configure(state="normal")

        if self.current_image_display:
            self.image_label.configure(image=self.current_image_display)
        else:
            self.image_label.configure(image="", text="请选择图片或开启视频识别")

        # 如果有视频识别的最后结果，直接更新垃圾信息介绍
        if self.last_video_prediction:
            self.update_status("🔄 正在获取处理建议...")
            self.get_trash_info(self.last_video_prediction)

        self.update_status("✅ 视频识别已停止")

    # 修改_video_recognition_worker方法，避免重复调用stop_video_recognition
    def _video_recognition_worker(self):
        """视频识别工作线程"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.root.after(0, lambda: messagebox.showerror("错误", "无法打开摄像头"))
            self.root.after(0, self.stop_video_recognition)
            return

        data_preprocessor = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # 加载字体
        font = None
        font_path = r"C:\Windows\Fonts\simhei.ttf"
        if os.path.exists(font_path):
            try:
                font = ImageFont.truetype(font_path, 28)
            except:
                print("无法加载字体，将使用默认字体")

        # 动态计算显示尺寸的函数
        def calculate_display_size():
            # 获取 image_card 的实际大小
            width = self.image_card.winfo_width()
            height = self.image_card.winfo_height()
            # 减去 padding
            return (max(width - 40, 100), max(height - 40, 100))

        while self.is_video_running:
            ret, frame = self.cap.read()
            if not ret:
                break

            try:
                # 预处理图像用于模型推理
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                input_tensor = data_preprocessor(pil_image).unsqueeze(0).to(self.device)

                # 模型推理
                with torch.no_grad():
                    outputs = self.model(input_tensor)
                    probability = F.softmax(outputs, dim=1)
                    max_prob, predicted = torch.max(probability, 1)
                    predicted_class = self.class_names[predicted.item()]
                    confidence = max_prob.item() * 100

                # 存储最后预测结果
                self.last_video_prediction = predicted_class
                self.last_video_confidence = confidence

                # 在视频帧上绘制结果
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_img)

                display_text = f"预测: {predicted_class}"
                confidence_text = f"置信度: {confidence:.1f}%"

                if font:
                    draw.text((10, 10), display_text, font=font, fill=(0, 255, 0))
                    draw.text((10, 50), confidence_text, font=font, fill=(0, 255, 0))
                else:
                    draw.text((10, 10), display_text, fill=(0, 255, 0))
                    draw.text((10, 30), confidence_text, fill=(0, 255, 0))

                # 转换回BGR用于显示
                frame_with_text = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

                # 使用动态计算的显示尺寸
                display_size = calculate_display_size()
                frame_resized = cv2.resize(frame_with_text, display_size)

                # 将OpenCV图像转换为Tkinter PhotoImage
                rgb_img = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_img)
                photo = ImageTk.PhotoImage(pil_img)

                # 更新显示
                self.current_image_display = photo  # 保持引用
                self.root.after(0, lambda p=photo: self.image_label.configure(image=p, text=""))

                # 更新右侧面板的分类结果
                self.root.after(0, lambda: self.class_result_label.configure(text=f"{predicted_class}"))
                self.root.after(0, lambda: self.confidence_bar.set(confidence / 100))
                self.root.after(0, lambda: self.confidence_text.configure(text=f"置信度: {confidence:.2f}%"))

                # 根据置信度设置颜色
                if confidence >= 80:
                    color = "#34C759"
                elif confidence >= 60:
                    color = "#FF9500"
                else:
                    color = "#FF3B30"
                self.root.after(0, lambda c=color: self.confidence_bar.configure(progress_color=c))
                self.root.after(0, lambda c=color: self.confidence_text.configure(text_color=c))

            except Exception as e:
                print(f"视频处理错误: {e}")
                break

        # 清理资源
        if self.cap and self.cap.isOpened():
            self.cap.release()

        # 确保UI状态恢复
        self.root.after(0, lambda: self._cleanup_after_video())

    # 添加一个新的清理方法，只处理资源释放和UI恢复，不调用get_trash_info
    def _cleanup_after_video(self):
        """视频结束后的清理工作，不包括生成介绍"""
        if not self.is_video_running:  # 确保只在视频确实停止的情况下执行
            if self.cap and self.cap.isOpened():
                self.cap.release()

            self.btn_video.configure(text="🎥 视频识别", fg_color="#2FA572", hover_color="#288E62")
            self.btn_select.configure(state="normal")
            self.btn_change_model.configure(state="normal")
            if self.current_image_path or self.last_video_prediction:
                self.btn_classify.configure(state="normal")

            if self.current_image_display:
                self.image_label.configure(image=self.current_image_display)
            else:
                self.image_label.configure(image="", text="请选择图片或开启视频识别")

            self.update_status("✅ 视频识别已停止")


def main():
    """主函数"""
    app = ImageClassifierGUI()
    app.root.mainloop()


if __name__ == "__main__":
    main()
