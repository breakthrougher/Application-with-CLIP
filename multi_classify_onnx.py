from transformers import CLIPProcessor, CLIPModel
import torch
import clip
from PIL import Image
import os
import shutil
import time
device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-B/32", device=device)
# model, preprocess = clip.load("RN50x64", device=device)

input_image_folder = "image_test"
output_image_folder = "images_onnx"

word_list = ["a fox", "a cat", "crane", "a bird", "a rabbit", "a monkey", "a panda", "a elephant", "a lion", "a dog"]
texts = clip.tokenize(word_list).to(device)

if not os.path.exists(output_image_folder):
  os.mkdir(output_image_folder)

for filename in os.listdir(input_image_folder):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        ## type:pytorch_tensor (N,D) means quantity and dimension
        if os.path.exists("feature_library_onnx.pth"):
            feature_library = torch.load("feature_library_onnx.pth", map_location=device, weights_only=True)
        else:
            feature_library = torch.empty((0, model.visual.output_dim), device=device)

        input_image_path = os.path.join(input_image_folder, filename)
        t1=time.time()
        image = preprocess(Image.open(input_image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            # 原本为float16，转为float32
            image_features = model.encode_image(image).float()
            # one image size:(1,512)

        logits_per_image, logits_per_text = model(image, texts)
        probs = logits_per_image.softmax(dim=-1).cpu().detach().numpy()
        max_probs_index = probs.argmax(axis=1)[0]
        t2=time.time()
        print(f"Image: {filename}", "Most likely class:", word_list[max_probs_index])
        print("Label probs:", probs)
        print("Inference time:", t2-t1)
        print("")

        if not any(torch.allclose(image_features, existing_feature) for existing_feature in feature_library):
            output_image_path=os.path.join(output_image_folder,f"image_{len(feature_library)}.jpg")
            shutil.copy(input_image_path, output_image_path)
            feature_library_tensor = torch.cat((feature_library, image_features), dim=0) if feature_library.size(0)>0 else image_features
            print("current feature library length:",len(feature_library_tensor))
            torch.save(feature_library_tensor, "feature_library_onnx.pth")