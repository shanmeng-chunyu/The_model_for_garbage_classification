import argparse
import base64
import json
import os

import requests
import torch

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


SILICONFLOW_API_KEY = "sk-yjntmvfcbgfrdojoyrmukreyszxrshxprlctsbqfobalzcye"
SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/chat/completions"


def encode_image_to_base64(image_path):
    """将图片文件编码为Base64字符串"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None


def get_class(args):
    if not SILICONFLOW_API_KEY:
        return "错误"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}"
    }
    prompt = f"""
        你是一位环保科普专家，请注意，接下来你需要对图片中的垃圾进行分类,具体的，你需要把垃圾分为以下类别：厨余垃圾，可回收垃圾，其他垃圾，有害垃圾。
        请注意，你只需要按照以下格式输出：{{垃圾类别}}_{{垃圾名称}}，不要有多余输出，也不要回答“好的”等客套话。
        请直接开始输出
        """
    image_name = os.listdir(args.source)[0]
    base64_image = encode_image_to_base64(os.path.join(args.source, image_name))
    data = {
        "model": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "low"
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ],
            }
        ],
        "max_tokens": 10,
        "temperature": 0.4,
    }
    try:
        response = requests.post(url=SILICONFLOW_API_URL, headers=headers, data=json.dumps(data), timeout=20)
        response.raise_for_status()
        result_json = response.json()
        predicted_class = result_json['choices'][0]['message']['content'].strip()
        return predicted_class, base64_image
    except requests.exceptions.RequestException as e:
        print(f"调用LLM API时发生网络错误: {e}")
        return "抱歉，获取详细介绍时遇到了网络问题，请稍后再试。"
    except (KeyError, IndexError) as e:
        print(f"解析LLM API响应时出错: {e}")
        return "抱歉，无法解析来自服务器的详细介绍。"


def get_description(predicted_class):
    if not SILICONFLOW_API_KEY:
        return "错误"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}"
    }

    prompt = f"""
    你是一位环保科普专家。请注意，接下来你需要针对 {predicted_class} 进行科普。
    请用平实易懂的语言，为我讲解一下这种东西应该如何妥善处理，以及为什么需要这样做。
    你的讲解应包含以下几个方面，请分段作答，结构清晰明确，总字数控制在200字以内：
    1.是什么：首先，用一两句话简单介绍一下这个物品的材质或用途。
    2.丢弃前的准备：在丢弃它之前，我需要做什么准备工作？（例如：是否需要清空、冲洗、保持干燥或完整等）
    3.它的“后续旅程”：它被收走以后，会经历怎样的处理流程？它有什么潜在的回收利用价值吗？
    4.不当处理的后果：如果我没有正确处理它，可能会带来什么不好的后果？
    请直接开始讲解，不要说任何无关的客套话。
"""
    data = {
        "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "max_tokens": 200,
        "temperature": 0.7,
    }
    try:
        response = requests.post(url=SILICONFLOW_API_URL, headers=headers, data=json.dumps(data), timeout=20)
        response.raise_for_status()
        result_json = response.json()
        description = result_json['choices'][0]['message']['content'].strip()
        return description
    except requests.exceptions.RequestException as e:
        print(f"调用LLM API时发生网络错误: {e}")
        return "抱歉，获取详细介绍时遇到了网络问题，请稍后再试。"
    except (KeyError, IndexError) as e:
        print(f"解析LLM API响应时出错: {e}")
        return "抱歉，无法解析来自服务器的详细介绍。"


def get_DIY(predicted_class, base64_image):
    if not SILICONFLOW_API_KEY:
        return "错误"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}"
    }

    prompt = f"""
        你是一位环保科普专家兼创意专家。请注意，接下来你需要针对如图所示的 {predicted_class} 进行科普。
        请用平实易懂的语言，为我讲解一下这种东西除了丢进垃圾桶之外，还有什么方法可以变废为宝。
        你需要分段作答，内容在300字以内。
        请至少提供两个个创意，可以是DIY，可以是用于其他用途等。
        请直接开始讲解，不要说任何无关的客套话,也不要说”好的“等客套话。
    """
    data = {
        "model": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "low"
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ],
            }
        ],
        "max_tokens": 300,
        "temperature": 0.4,
    }
    try:
        response = requests.post(url=SILICONFLOW_API_URL, headers=headers, data=json.dumps(data), timeout=20)
        response.raise_for_status()
        result_json = response.json()
        DIY = result_json['choices'][0]['message']['content'].strip()
        return DIY
    except requests.exceptions.RequestException as e:
        print(f"调用LLM API时发生网络错误: {e}")
        return "抱歉，获取详细介绍时遇到了网络问题，请稍后再试。"
    except (KeyError, IndexError) as e:
        print(f"解析LLM API响应时出错: {e}")
        return "抱歉，无法解析来自服务器的详细介绍。"



def main(args):
    predicted_class, base64_img = get_class(args)
    print(predicted_class)
    description = get_description(predicted_class)
    print(description)
    DIY = get_DIY(predicted_class, base64_img)
    print(DIY)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="./image", type=str, help='需要分类的图片所在目录')
    parser.add_argument('--output_dir', default='./run/classify/exp', type=str, help='分类结果保存目录')
    parser.add_argument('--model_path', default='./run/train/exp_ViT3.0_best_highlr/best_model.pth', type=str,
                        help='模型所在目录')
    parser.add_argument('--class_file', default='./class_name.txt', type=str, help='分类标签所在文件')
    args = parser.parse_args()
    main(args)
