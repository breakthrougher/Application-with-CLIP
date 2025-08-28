# -*- coding: utf-8 -*-
from transformers import CLIPProcessor, CLIPModel
import torch
import clip
from PIL import Image
import os
import shutil
import time
import gradio as gr
import numpy as np

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"运行设备: {device}")

# 加载CLIP模型
model, preprocess = clip.load("ViT-B/32", device=device)

# 确保中文正常显示
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

# 配置
output_image_folder = "images_onnx"
feature_library_path = "feature_library_onnx.pth"

# 类别标签（中文）
word_list = ["a fox", "a cat", " a crane", "a bird", "a rabbit", "a monkey", "a panda", "a elephant", "a lion", "a dog"]
texts = clip.tokenize(word_list).to(device)

# 确保输出文件夹存在
if not os.path.exists(output_image_folder):
    os.makedirs(output_image_folder)


def load_feature_library():
    """加载特征库，如果不存在则创建空库"""
    if os.path.exists(feature_library_path):
        try:
            # 尝试加载特征库
            feature_library = torch.load(feature_library_path, weights_only=True).to(device)
            print(f"已加载特征库，包含 {len(feature_library)} 个特征")
            return feature_library
        except Exception as e:
            print(f"加载特征库失败: {e}")
            print("创建新的空特征库")
    return torch.empty((0, model.visual.output_dim), device=device)


def save_feature_library(feature_library):
    """保存特征库"""
    try:
        torch.save(feature_library, feature_library_path)
        print(f"特征库已保存，包含 {len(feature_library)} 个特征")
    except Exception as e:
        print(f"保存特征库失败: {e}")


def classify_image(input_image_path):
    """对输入图像进行分类，并将新图像添加到特征库"""
    if not input_image_path:
        return "请上传图片"

    try:
        # 加载特征库
        feature_library = load_feature_library()

        # 开始计时
        t1 = time.time()

        # 预处理图像
        image = preprocess(Image.open(input_image_path)).unsqueeze(0).to(device)

        # 提取图像特征
        with torch.no_grad():
            image_features = model.encode_image(image).float()

            # 计算与类别文本的相似度
            logits_per_image, _ = model(image, texts)
            probs = logits_per_image.softmax(dim=-1).cpu().detach().numpy()

        # 获取最可能的类别
        max_probs_index = probs.argmax(axis=1)[0]
        confidence = probs[0, max_probs_index] * 100

        # 结束计时
        t2 = time.time()
        inference_time = t2 - t1

        # 输出结果
        result_text = f"分类结果: {word_list[max_probs_index]} ({confidence:.2f}%)"
        print(f"{result_text}，推理时间: {inference_time:.2f}秒")

        # 检查是否为新图像（避免重复添加）
        is_new_image = not any(torch.allclose(image_features, existing_feature) for existing_feature in feature_library)

        if is_new_image:
            # 保存图像到库中
            output_image_idx = len(feature_library)
            output_image_path = os.path.join(output_image_folder, f"image_{output_image_idx}.jpg")
            shutil.copy(input_image_path, output_image_path)

            # 更新特征库
            feature_library = torch.cat((feature_library, image_features), dim=0)
            save_feature_library(feature_library)

            result_text += "\n（已添加到图片库）"
        else:
            result_text += "\n（图片已存在于库中）"

        return result_text

    except Exception as e:
        print(f"分类过程出错: {e}")
        return f"错误: {str(e)}"


def search_image(word):
    """根据关键词搜索最匹配的图像"""
    if not word:
        return None, "请输入搜索关键词"

    try:
        # 加载特征库
        feature_library = load_feature_library()

        if len(feature_library) == 0:
            return None, "图片库为空，请先分类一些图片"

        # 编码搜索文本
        text = clip.tokenize([word]).to(device)

        with torch.no_grad():
            text_features = model.encode_text(text).float()

        # 计算相似度
        similarities = (feature_library @ text_features.T).squeeze().softmax(dim=-1)*100.0
        best_match_idx = similarities.argmax().item()
        confidence = similarities[best_match_idx].item()

        # 获取最佳匹配的图片路径
        best_image_path = os.path.join(output_image_folder, f"image_{best_match_idx}.jpg")

        # 确保图片存在
        if not os.path.exists(best_image_path):
            return None, f"未找到匹配的图片（索引错误: {best_match_idx}）"

        print(f"最佳匹配: {best_image_path}，相似度: {confidence:.2f}%")
        return best_image_path, f"搜索结果: {word}（相似度: {confidence:.2f}%）"

    except Exception as e:
        print(f"搜索过程出错: {e}")
        return None, f"错误: {str(e)}"


def get_image_library_preview():
    """获取图片库预览"""
    if not os.path.exists(output_image_folder):
        return []

    image_files = [f for f in os.listdir(output_image_folder)
                   if f.endswith(('.jpg', '.jpeg', '.png'))]

    return [os.path.join(output_image_folder, f) for f in image_files][-12:]  # 取最近的12张图片


# 创建Gradio界面
with gr.Blocks(title="图像分类与搜索应用", theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 图像分类与搜索应用")

    with gr.Row():
        # 左侧：图像分类功能
        with gr.Column(scale=1):
            gr.Markdown("### 图像分类")
            with gr.Group():
                upload = gr.Image(label="上传图片", type="filepath")
                classify_btn = gr.Button("🔍 开始分类", variant="primary")
                classify_output = gr.Textbox(label="分类结果", lines=3)

        # 右侧：图像搜索功能
        with gr.Column(scale=1):
            gr.Markdown("### 图像搜索")
            with gr.Group():
                category_input = gr.Textbox(label="搜索关键词",
                                            placeholder="输入关键词，如: 猫、鸟...")
                search_btn = gr.Button("🔎 搜索图片", variant="primary")
                with gr.Row():
                    with gr.Column(scale=3):
                        image_output = gr.Image(label="搜索结果", type="filepath")
                    with gr.Column(scale=1):
                        search_output = gr.Textbox(label="", lines=2)

    # 图片库预览
    gr.Markdown("### 图片库预览")
    gallery = gr.Gallery(label="图片库", columns=[4, 4, 4, 4],
                         height=200, show_label=False)

    # 设置事件处理
    classify_btn.click(
        classify_image,
        inputs=upload,
        outputs=classify_output
    )

    search_btn.click(
        search_image,
        inputs=category_input,
        outputs=[image_output, search_output]
    )

    # 应用加载时更新图片库预览
    demo.load(
        get_image_library_preview,
        inputs=None,
        outputs=gallery
    )

    # 分类后更新图片库预览
    classify_btn.click(
        get_image_library_preview,
        inputs=None,
        outputs=gallery
    )

# 启动应用
if __name__ == "__main__":
    demo.launch(server_name='0.0.0.0', server_port=7870, inbrowser=True)