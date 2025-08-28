from transformers import CLIPProcessor, CLIPModel
import torch
import clip
from PIL import Image
import os

model, preprocess = clip.load("ViT-B/32", device="cpu")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("current use device:",device)

if not os.path.exists("feature_library_onnx.pth"):
    print("System doesn't have feature libary!")

feature_library = torch.load("feature_library_onnx.pth",weights_only=True).to(device).float()
print(feature_library.size())
print("curent feature library length:",len(feature_library))

text = clip.tokenize(["a photo of a lion"]).to(device)
model.to(device)

with torch.no_grad():
    text_features = model.encode_text(text)

# Calculate similarity
similarities = (feature_library @ text_features.T).squeeze().softmax(dim=-1)*100.0
best_match_idx = similarities.argmax().item()
print(f"Best matching image is image_{best_match_idx}.jpg")

best_image_path=f"images_onnx/image_{best_match_idx}.jpg"
best_image = Image.open(best_image_path)
best_image.show()