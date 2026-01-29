import os
import numpy as np
from PIL import Image
from pathlib import Path
import torch
from torchvision import models, transforms
import torch.nn as nn
from tqdm import tqdm

# ========= 参数 =========
query_image_path = Path(r"C:\Users\Admin\Documents\GitHub\kaggle_\csiro-biomass\item\image.jpg")
search_folder = Path(r"D:\DATA_hao\Kaggle_\csiro-biomass\train")
top_k = 5  # 返回最相似的前几张

# ========= 加载模型 (ResNet50 去掉最后一层) =========
device = "cpu"

model = models.resnet50(pretrained=True)
model = nn.Sequential(*list(model.children())[:-1])  # 移除分类层
model.to(device)
model.eval()

# ========= 图像预处理 =========
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# 提取特征的函数
def extract_feature(img_path):
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = model(img).squeeze().cpu().numpy()

    # 归一化，让相似度更稳定
    feat = feat / np.linalg.norm(feat)
    return feat

# ========= 提取查询图像特征 =========
query_feat = extract_feature(query_image_path)

# ========= 遍历文件夹中的图像并计算相似度 =========
results = []
img_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# 遍历文件夹中的每一张图片
for img_path in tqdm(search_folder.iterdir()):
    if img_path.suffix.lower() not in img_extensions:
        continue

    try:
        # 提取当前图片的特征
        feat = extract_feature(img_path)
        
        # 计算查询图像和当前图像的余弦相似度
        sim = np.dot(query_feat, feat)  # 余弦相似度 = 向量点积
        results.append((img_path, sim))
    except Exception as e:
        print(f"跳过 {img_path}, 错误: {e}")
        continue

# ========= 找出最相似的 top_k =========
results = sorted(results, key=lambda x: x[1], reverse=True)
top_matches = results[:top_k]

# ========= 输出最相似的图片 =========
print("\n===== 最相似的图片 =====")
for path, sim in top_matches:
    print(f"{sim:.4f}   {path}")
