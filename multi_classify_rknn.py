from transformers import CLIPProcessor, CLIPModel
import torch
import clip
from PIL import Image
import os
import shutil
from rknnlite.api import RKNNLite
import numpy as np
from torchvision import transforms
import time

rknn_lite = RKNNLite()
rknn_lite.load_rknn("visual.rknn")
rknn_lite.init_runtime()

device = "cuda" if torch.cuda.is_available() else "cpu"

model, process = clip.load("ViT-B/32", device=device)

preprocess = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),  # 缩放到 224x224
    transforms.CenterCrop(224),  # 中心裁剪确保尺寸
    transforms.ToTensor(),  # 转为张量 [0,1] 范围 (C, H, W)
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],  # CLIP 专用均值
        std=[0.26862954, 0.26130258, 0.27577711]  # CLIP 专用方差
    )
])

input_image_folder = "image_test"
output_image_folder = "images_rknn"

word_list = ["a fox", "a cat", "crane", "a bird", "a rabbit", "a monkey", "a panda", "a elephant", "a lion", "a dog"]
texts = clip.tokenize(word_list).to(device)

if not os.path.exists(output_image_folder):
  os.mkdir(output_image_folder)

for filename in os.listdir(input_image_folder):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        # type:pytorch tensor (N,D) quantity and dimension
        if os.path.exists("feature_library_rknn.pth"):
            feature_library = torch.load("feature_library_rknn.pth").to(device)
        else:
            feature_library = torch.empty((0, model.visual.output_dim), device=device)

        input_image_path = os.path.join(input_image_folder, filename)
        t1=time.time()
        image = Image.open(input_image_path).convert('RGB')
        img_tensor = preprocess(image)
        img_numpy = img_tensor.unsqueeze(0)
        out = rknn_lite.inference(inputs=[img_numpy.numpy().astype(np.float32)])
        image_features = torch.from_numpy(out[0])
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # float16
        logits_per_image, logits_per_text = model(image, texts)
        probs = logits_per_image.softmax(dim=-1).cpu().detach().numpy()
        max_probs_index = probs.argmax(axis=1)[0]
        t2=time.time()
        print(f"Image: {filename}", "Most likely class:", word_list[max_probs_index])
        print("Label probs:", probs)
        print("Inference time:", t2-t1)
        print("")

        if not any(torch.allclose(image_features, existing_feature) for existing_feature in feature_library):
            output_image_path = os.path.join(output_image_folder, f"image_{len(feature_library)}.jpg")
            shutil.copy(input_image_path, output_image_path)
            feature_library_tensor = torch.cat((feature_library, image_features), dim=0) if feature_library.size(0)>0 else image_features
            print("current feature library length:", len(feature_library_tensor))
            torch.save(feature_library_tensor, "feature_library_rknn.pth")