# 📦
import os
import gc
import cv2
import h5py
import json
import time
import shutil
import socket
import psutil
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from transformers import AutoImageProcessor, AutoModel
import math
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from tabulate import tabulate

from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import r2_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import _LRScheduler
from torch.amp import autocast, GradScaler       

import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import random
from PIL import Image, ImageDraw, ImageFilter

import timm
from torchvision import models, transforms
from torchvision.models import get_model_weights


# 🌱 Path Initialization
if socket.gethostname() == 'hao-2':
    dir = Path('D:/DATA_hao/Kaggle_/csiro-biomass/')
    DIRS = {
        "dir"  : dir,
        "train": Path(dir, "train"),
        "test" : Path(dir, "test"),
        "model": Path(dir, "DualStream_multihead"),
        "data" : Path(dir),
    }
elif socket.gethostname() == 'simon-MS-7D94':
    dir = Path('/home/simon/simondisk2/simon/CSIRO/csiro-biomass')
    DIRS = {
        "dir"  : dir,
        "train": Path(dir, "train"),
        "test" : Path(dir, "test"),
        "model": Path("/home/simon/simondisk2/simon/CSIRO/Train_75", "DualStream_multihead"),
        "data" : Path(dir),
    }
elif socket.gethostname() == 'user-PowerEdge-XE9680':
    dir = Path('/data4/huangweigang/gh/csiro-biomass')
    DIRS = {
        "dir"  : dir,
        "train": Path(dir, "train"),
        "test" : Path(dir, "test"),
        "model": Path(dir, "DualStream_multihead"),
        "data" : Path(dir),
    }
elif socket.gethostname() == 'I24fece71c700101943':
    dir = Path('/hy-tmp/CSIRO/data/csiro-biomass')
    DIRS = {
        "dir"  : dir,
        "train": Path(dir, "train"),
        "test" : Path(dir, "test"),
        "model": Path(dir, "biomasshead"),
        "data" : Path(dir),
    }
else:
    dir = Path('/kaggle/input/csiro-biomass')
    DIRS = {
        "dir"  : dir,
        "train": Path(dir, "train"),
        "test" : Path(dir, "test"),
        "model": Path('/kaggle/input/single-flow-swa'),
        "data" : Path("/kaggle/working/"),
    }


def show_df_info(df, name: str):
    """
    Print the shape and column names of a single DataFrame.

    Args:
        df   : pandas.DataFrame
        name : Display name (string)
    """
    print(f"📊 {name:<16} shape: {str(df.shape):<16}  列名: {df.columns.tolist()}")


def move_column_first(df, col_name):
    if col_name not in df.columns:
        raise ValueError(f"列 '{col_name}' 不存在于 DataFrame 中。")

    cols = [col_name] + [c for c in df.columns if c != col_name]
    return df[cols]



def select_free_gpu(aimID = None, threshold_mem_MB = 500, threshold_util = 20):
    """
    Automatically select an available GPU (works in both .py and Jupyter environments).
    """
    def get_gpu_info():
        """Retrieve GPU information using nvidia-smi."""
        query = (
            "index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw"
        )
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )

        gpus = []
        for line in result.stdout.strip().split("\n"):
            idx, name, mem_used, mem_total, util, temp, power = [x.strip() for x in line.split(",")]
            gpus.append({
                "index": int(idx),
                "name": name,
                "mem_used_MB": int(mem_used),
                "mem_total_MB": int(mem_total),
                "util_%": int(util),
                "temp_C": int(temp),
                "power_W": float(power),
            })
        return gpus

    gpus = get_gpu_info()

    free_gpus = [
        g for g in gpus
        if g["mem_used_MB"] < threshold_mem_MB and g["util_%"] <= threshold_util
    ]

    if not free_gpus:
        gpus.sort(key=lambda x: x["mem_used_MB"])
        selected = gpus[0]
        reason = "(No fully idle GPU found — selected the one with lowest memory usage)"
    else:
        selected = free_gpus[0]
        reason = "(Idle GPU detected)"

    print(tabulate(
        [[g["index"], g["name"], f"{g['mem_used_MB']}/{g['mem_total_MB']} MB",
          f"{g['util_%']}%", f"{g['temp_C']}°C", f"{g['power_W']}W"]
         for g in gpus],
        headers=["GPU", "Name", "Memory", "Util", "Temp", "Power"],
        tablefmt="grid"
    ))
    if aimID == None:
        idx = selected["index"]
        device_name = f"cuda:{idx}"
    else:
        print(f"\n⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️ 指定GPU:{aimID} ⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️\n")
        idx = aimID
        device_name = f"cuda:{idx}"


    try:
        from IPython import get_ipython
        in_notebook = get_ipython() is not None
    except Exception:
        in_notebook = False

    if not in_notebook:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n✅ Selected GPU {idx} {reason}")
        print(f"Current device: {device} (logical GPU {idx})\n")
    else:
        device = torch.device(device_name if torch.cuda.is_available() else "cpu")
        print(f"\n⚠️ Detected Jupyter environment — not modifying CUDA_VISIBLE_DEVICES.")
        print(f"✅ Using device: {device_name} {reason}\n")

    return idx, device



def setup_seed(seed, deterministic=True):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False
    
# 打印清单
def config_to_str(config, indent = 0):
    """递归生成配置字符串"""
    prefix = "     " * indent
    lines = []
    for key, value in config.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}🔹 {key}:")
            lines.append(config_to_str(value, indent + 1))  # 递归拼接子字典
        else:
            lines.append(f"{prefix}- {key:<20}: {value}")
    return "\n".join(lines)



# Model 
class MySingleStreamModel(nn.Module):
    def __init__(self, 
                backbone_name="convnext_tiny", 
                pretrained=True, 
                config = None):
        """
        Args:
            backbone_name: timm model name (e.g., convnext_tiny, resnet50)
            pretrained: whether to load ImageNet pretrained weights
            freeze_ratio: ratio of backbone layers to freeze (0~1)
            weights_dict: per-target weights (dict) for WeightedSmoothL1Loss
        """
        super().__init__()
        print("Current backbone:", backbone_name)

        img_size = (config["img_size"], config.get("img_width", config["img_size"]))

        # 1️⃣ 创建 Backbone 并处理本地权重加载
        if socket.gethostname() in ["user-PowerEdge-XE9680", "hao-2"]:            
            model_path = f"/data4/huangweigang/gh/timm_model/{backbone_name}.pth"
            self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
            try:
                state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
            except TypeError:
                state_dict = torch.load(model_path, map_location='cpu')
            self.backbone.load_state_dict(state_dict, strict=False)
        else:
            self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        
        
        # 2️⃣ 获取特征维度 (在包装 LoRA 前获取)
        self.feature_dim = self.backbone.num_features
        print(f"📊 Backbone feature dimension: {self.feature_dim}")


        # 3️⃣ 微调策略：LoRA vs. Freeze Ratio
        if config.get("use_lora", False):
            print(f"🚀 Enabling LoRA for {backbone_name}...")
            lora_config = LoraConfig(
                r = config.get("lora_r", 8),
                lora_alpha = config.get("lora_alpha", config.get("lora_r", 8) * 2),
                # 对于 ViT，通常作用于 qkv 映射层
                target_modules=["qkv"] if "vit" in backbone_name else ["q_proj", "v_proj"],
                lora_dropout=config.get("lora_dropout", 0.1),
                bias="none",
            )
            self.backbone = get_peft_model(self.backbone, lora_config)
        else:
            # 原有冻结逻辑
            params = list(self.backbone.parameters())
            freeze_until = int(len(params) * config.get("freeze_ratio", 0.8))
            for i, p in enumerate(params):
                p.requires_grad = i >= freeze_until



        # 4️⃣ 构建 Multi-head 回归头
        def make_head():
            return nn.Sequential(
                nn.Linear(self.feature_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )

        self.head_green  = make_head() 
        self.head_clover = make_head()  
        self.head_dead   = make_head()   
        self.head_gdm    = make_head()
        self.head_total  = make_head()
        self.softplus = nn.Softplus(beta=1.0)

        self.weights = config["weights"]

    def forward(self, img):
        # Extract features
        feat = self.backbone(img)

        # Predict three primary targets
        G = self.softplus(self.head_green(feat))
        C = self.softplus(self.head_clover(feat))
        D = self.softplus(self.head_dead(feat))
        GDM   = self.softplus(self.head_gdm(feat))
        Total = self.softplus(self.head_total(feat))

        preds = torch.cat([G, C, D, GDM, Total], dim=1)  # [B, 5]
        return preds


    def compute_loss(self, preds, targets):
        """
        Loss aligned with Kaggle's official Global Weighted R² metric:
        loss = ss_res / ss_tot
        (smaller is better — same direction as the competition metric)
        """
        preds = preds.view(-1, 5)
        targets = targets.view(-1, 5)

        weights = torch.tensor([0.1, 0.1, 0.1, 0.2, 0.5], device=preds.device)

        y_true_flat = targets.view(-1)
        y_pred_flat = preds.view(-1)

        w_flat = torch.cat([
            torch.full_like(targets[:, i], weights[i], device=preds.device)
            for i in range(5)
        ])
        y_mean = torch.sum(w_flat * y_true_flat) / torch.sum(w_flat)

        ss_res = torch.sum(w_flat * (y_true_flat - y_pred_flat) ** 2)
        ss_tot = torch.sum(w_flat * (y_true_flat - y_mean) ** 2)

        loss = ss_res / ss_tot

        return loss

class SingleStreamDataset(Dataset):
    def __init__(self, df, image_dir, config, pre_transform=None, transform=None):
        """
        Args:
            df: DataFrame containing an 'image_path' column.
            image_dir: Directory where images are stored.
            target_cols: Target columns (used for training datasets).
            transform: Albumentations transform pipeline.
        """
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.target_cols = config["target_cols"]
        self.pre_transform = pre_transform
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = Path(self.image_dir, str(row["image_path"]))

        if not img_path.exists():
            print(f"⚠️ Image not found: {img_path}")
            image = np.zeros((1000, 2000, 3), dtype=np.uint8)
        else:
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"⚠️ Failed to read image: {img_path} ({e})")
                image = np.zeros((1000, 2000, 3), dtype=np.uint8)

        image = np.array(image)
        if self.pre_transform:
            image = self.pre_transform(image=image)["image"]
        
        if self.transform:
            image = self.transform(image=image)["image"]

        if self.target_cols is not None:
            targets = torch.tensor(
                row[self.target_cols].astype(float).values,
                dtype=torch.float32
            )
            return image, targets
        else:
            return image


def get_pre_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=(0.8, 1.2), 
                contrast_limit=(0.8, 1.2),   
                p=0.3
            ),
            A.ColorJitter(p=0.7)
        ], p=0.4),
    ])


def _get_resize_dims(config):
    height = config["img_size"]
    width = int(config.get("img_width", config["img_size"]))
    return height, width


def get_train_transforms(config):
    height, width = _get_resize_dims(config)
    return A.Compose([
        A.Resize(height, width),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_valid_transforms(config):
    height, width = _get_resize_dims(config)
    return A.Compose([
        A.Resize(height, width),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_tta_transforms(config):
    height, width = _get_resize_dims(config)
    return {
        "base": A.Compose([
            A.Resize(height, width),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        "hflip": A.Compose([
            A.HorizontalFlip(p=1.0)
        ]),
        # "vflip": A.Compose([
        #     A.VerticalFlip(p=1.0)
        # ])
    }


def compute_cv_score(all_preds, all_targets):
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    target_cols = ["Dry_Green_g", "Dry_Clover_g", "Dry_Dead_g", "GDM_g", "Dry_Total_g"]
    weights = np.array([0.1, 0.1, 0.1, 0.2, 0.5])
  
    y_true_flat = np.concatenate([targets[i, :] for i in range(targets.shape[0])])
    y_pred_flat = np.concatenate([preds[i, :] for i in range(preds.shape[0])])
    w_flat = np.concatenate([np.full_like(targets[:, i], weights[i]) for i in range(5)])

    y_mean = np.sum(w_flat * y_true_flat) / np.sum(w_flat)

    ss_res = np.sum(w_flat * (y_true_flat - y_pred_flat) ** 2)
    ss_tot = np.sum(w_flat * (y_true_flat - y_mean) ** 2)

    r2_global = 1 - ss_res / ss_tot
    return r2_global


class WarmupCosineStepLR(_LRScheduler):
    def __init__(
        self,
        optimizer,
        total_steps=50,
        n_constant=12,
        n_cosine_1=12,
        n_constant_2=13,
        n_cosine_2=13,
        min_lr=1e-6,
        last_epoch=-1,
    ):
        self.total_steps = total_steps
        self.n_constant = n_constant
        self.n_cosine_1 = n_cosine_1
        self.n_constant_2 = n_constant_2
        self.n_cosine_2 = n_cosine_2
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        if step <= self.n_constant:                       # 1. 线性 warmup
            return [base_lr * step / self.n_constant for base_lr in self.base_lrs]
        elif step <= self.n_constant + self.n_cosine_1:   # 2. 第一次 cosine
            s = step - self.n_constant
            return [0.5 * bl + (bl - 0.5 * bl) * (1 + math.cos(math.pi * s / self.n_cosine_1)) / 2
                    for bl in self.base_lrs]
        elif step <= self.n_constant + self.n_cosine_1 + self.n_constant_2:  # 3. 常数
            return [0.5 * bl for bl in self.base_lrs]
        else:                                             # 4. 第二次 cosine
            s = step - (self.n_constant + self.n_cosine_1 + self.n_constant_2)
            return [self.min_lr + (0.5 * bl - self.min_lr) * (1 + math.cos(math.pi * s / self.n_cosine_2)) / 2
                    for bl in self.base_lrs]


def mixup_batch(images, targets, alpha=0.4):
    if alpha <= 0:
        return images, targets, targets, 1.0

    lam = np.random.beta(alpha, alpha)
    indices = torch.randperm(images.size(0), device=images.device)
    shuffled_images = images[indices]
    shuffled_targets = targets[indices]

    mixed_images = lam * images + (1 - lam) * shuffled_images
    mixed_images = mixed_images.contiguous(memory_format=torch.channels_last)
    return mixed_images, targets, shuffled_targets, lam


def cutmix_batch(images, targets, alpha=1.0):
    if alpha <= 0:
        return images, targets, targets, 1.0

    lam = np.random.beta(alpha, alpha)
    indices = torch.randperm(images.size(0), device=images.device)
    shuffled_images = images[indices]
    shuffled_targets = targets[indices]

    _, _, H, W = images.size()
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    images[:, :, y1:y2, x1:x2] = shuffled_images[:, :, y1:y2, x1:x2]

    lam_adjusted = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    images = images.contiguous(memory_format=torch.channels_last)
    return images, targets, shuffled_targets, lam_adjusted


ifPrint = True

def train_one_epoch(model, dataloader, optimizer, device, scaler, config):
    model.train()
    running_loss = []

    start_epoch = time.time()
    prev_end = start_epoch  

    for step, (images, targets) in enumerate(dataloader):
        t_load = time.time()  
        data_load_time = t_load - prev_end

        t0 = time.time()
        images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        targets = targets.to(device, non_blocking=True)
        t1 = time.time()

        # ====== Forward + Backward ======
        optimizer.zero_grad(set_to_none=True)

        use_mixup = config["mixup"] and np.random.rand() < 0.5
        use_cutmix = (not use_mixup) and config["cutmix"] and np.random.rand() < 0.5

        if use_mixup:
            mixed_images, targets_a, targets_b, lam = mixup_batch(
                images, targets, config.get("mixup_alpha", 0.4)
            )
            with torch.amp.autocast(device_type='cuda'):
                preds = model(mixed_images)
                loss = lam * model.compute_loss(preds, targets_a) + (1 - lam) * model.compute_loss(preds, targets_b)
        elif use_cutmix:
            mixed_images, targets_a, targets_b, lam = cutmix_batch(
                images, targets, config.get("cutmix_alpha", 1.0)
            )
            with torch.amp.autocast(device_type='cuda'):
                preds = model(mixed_images)
                loss = lam * model.compute_loss(preds, targets_a) + (1 - lam) * model.compute_loss(preds, targets_b)
        else:
            with torch.amp.autocast(device_type='cuda'):
                preds = model(images)
                loss = model.compute_loss(preds, targets)
                
        t2 = time.time()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        t3 = time.time()

        running_loss.append(loss.item())
        prev_end = t3  
        if ifPrint and step % 20  == 0:
            print(
                f"[TRAIN] Step {step:4d} | "
                f"data load: {data_load_time*1000:8.1f} ms | "
                f"to(device): {(t1-t0)*1000:.1f} ms | "
                f"forward+loss: {(t2-t1)*1000:7.1f} ms | "
                f"backward+opt: {(t3-t2)*1000:.1f} ms | "
                f"total: {(t3-t_load)*1000:7.1f} ms"
            )



    end_epoch = time.time()
    epoch_time = end_epoch - start_epoch
    avg_batch_time = epoch_time / len(dataloader)

    if ifPrint : print(f"[TRAIN] Epoch total time: {epoch_time:.2f}s | "
          f"{len(dataloader)} batches | {avg_batch_time:.3f}s/batch")
    

    return float(np.mean(running_loss))


def validate_one_epoch(model, dataloader, device):
    model.eval()
    val_losses, all_preds, all_targets = [], [], []

    start_epoch = time.time()
    prev_end = start_epoch 

    with torch.no_grad():
        for step, (images, targets) in enumerate(dataloader):
            t_load = time.time()
            data_load_time = t_load - prev_end

            t0 = time.time()
            images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            targets = targets.to(device, non_blocking=True)
            t1 = time.time()

            preds = model(images)
            val_loss = model.compute_loss(preds, targets).item()
            t2 = time.time()

            val_losses.append(val_loss)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

            prev_end = t2 

            if ifPrint and step % 6  == 0:
                print(
                    f"[VAL  ] Step {step:4d} | "
                    f"data load: {data_load_time*1000:8.1f} ms | "
                    f"to(device): {(t1 - t0)*1000:.1f} ms | "
                    f"forward+loss: {(t2 - t1)*1000:7.1f} ms | "
                    f"total: {(t2 - t_load)*1000:7.1f} ms"
                )
            

    end_epoch = time.time()
    epoch_time = end_epoch - start_epoch
    avg_val_loss = float(np.mean(val_losses))
    r2_global = compute_cv_score(all_preds, all_targets)

    if ifPrint : print(
        f"[VAL  ] Epoch total time: {epoch_time:.2f}s | "
        f"{len(dataloader)} batches | {epoch_time / len(dataloader):.3f}s/batch"
    )

    return avg_val_loss, r2_global



def get_fold_loaders(df, fold, groups, config, device):
    """为单个 fold 构建训练和验证 DataLoader（各自独立 transform）"""
    train_df = df[df["fold"] != fold].reset_index(drop=True)
    valid_df = df[df["fold"] == fold].reset_index(drop=True)

    train_dataset = SingleStreamDataset(
        train_df,
        DIRS["dir"], 
        config, 
        pre_transform=get_pre_transforms(), 
        transform=get_train_transforms(config)
    )
    valid_dataset = SingleStreamDataset(
        valid_df, 
        DIRS["dir"], 
        config, 
        pre_transform=None, 
        transform=get_valid_transforms(config)
    )

    num_workers = config["num_workers"]
    prefetch_factor = config["prefetch_factor"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=num_workers,           # ✅ 启用多核加载
        pin_memory=True,                   # ✅ 加速 CPU→GPU 拷贝
        prefetch_factor=prefetch_factor,   # ✅ 每个 worker 预加载 3 个 batch
        persistent_workers=True            # ✅ 保持 worker 常驻
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=max(1, num_workers // 2),
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=True
    )

    return train_loader, valid_loader


def should_stop(cv_scores, window=20, threshold=0.001):
    """Check if recent CV score fluctuations are stable enough to trigger early stopping."""

    if len(cv_scores) < window :  
        return False
    window_scores = cv_scores[-window:]
    diff = max(window_scores) - min(window_scores)
    if diff < threshold:
        print("diff =", diff, f"< {threshold}")
    return diff < threshold


def fetch_scheduler(optimizer, config):
    if config["scheduler_type"] == "warmup_cosine":
        print("use custom warmup_cosine scheduler!!")
        scheduler = WarmupCosineStepLR(optimizer,
                                       total_steps=int(config["epochs"]),  # 总共训练的epoch数
                                       n_constant=int(config["n_constant"]),  # warmup阶段的epoch数
                                       n_cosine_1=int(config["n_cosine_1"]),  # 第一个Cosine退火阶段的epoch数
                                       n_constant_2=int(config["n_constant_2"]),  # Constant阶段的epoch数
                                       n_cosine_2=int(config["n_cosine_2"]),  # 第二个Cosine退火阶段的epoch数
                                       min_lr=float(config["min_lr"]),  # 最小学习率
                                       )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=config["lr"]/100)

    return scheduler


def save_checkpoint(model, path_prefix, score, epoch, config):
    # A. 保存全量 state_dict (包含 Heads 和 LoRA)
    torch.save(model.state_dict(), f"{path_prefix}.pt")
    
    # B. 如果开启了 LoRA，额外保存一份适配器权重
    if config.get("use_lora", False):
        # lora_path = Path(str(path_prefix) + "_lora_adapter")
        # model.backbone.save_pretrained(lora_path)
        pass

def update_top_models(model, current_score, epoch, fold, model_path_prefix, top_list, config, max_keep=3, reverse=True):
    """
    通用 Top 模型管理函数
    reverse=True: CV 越大越好
    reverse=False: Loss 越小越好
    """
    # 1. 检查是否足以进入 Top 列表
    should_save = len(top_list) < max_keep or (
        current_score > min(top_list, key=lambda x: x[0])[0] if reverse else 
        current_score < max(top_list, key=lambda x: x[0])[0]
    )
    
    if should_save:
        # 定义完整的文件路径名（用于存入 top_list）
        full_pt_path = f"{model_path_prefix}.pt"
        
        # 2. 执行保存 (调用你之前定义的适配 LoRA 的 save_checkpoint)
        save_checkpoint(model, model_path_prefix, current_score, epoch, config)
        
        # 3. 更新列表
        top_list.append((current_score, epoch, full_pt_path))
        top_list.sort(key=lambda x: x[0], reverse=reverse)
        
        # 4. 剔除多余模型
        if len(top_list) > max_keep:
            removed = top_list.pop()
            removed_path = removed[2]
            
            # 删除 .pt 文件
            if os.path.exists(removed_path):
                os.remove(removed_path)
            
            # 删除 LoRA 文件夹 (如果存在)
            lora_dir = removed_path.replace(".pt", "_lora_adapter")
            if os.path.exists(lora_dir):
                shutil.rmtree(lora_dir)
                
            print(f"🗑️ 已清理 Fold {fold} 旧模型: {os.path.basename(removed_path)}")
        
        type_str = "CV  " if reverse else "Loss"
        print(f"🌟 更新 {type_str} Top-{len(top_list)}: Fold {fold} | Epoch {epoch} | Score={current_score:.4f}")
    
    return should_save, top_list


def train_one_fold(fold, df, groups, save_dir, config, device):
    """Train a single fold and return performance metrics for that fold."""
    if config["mixup"]:
        print("use mixup!!!")
        
    train_loader, valid_loader = get_fold_loaders(df, fold, groups, config, device)

    model = MySingleStreamModel(config["backbone_name"], pretrained=True, config=config)
    model = model.to(device).to(memory_format=torch.channels_last)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config["lr"], weight_decay=2e-5)
    scheduler = fetch_scheduler(optimizer, config)
    scaler = torch.amp.GradScaler(device="cuda")

    train_losses, val_losses, cv_scores, LR_records = [], [], [], []
    epoch_times = [] 
    all_progress = config["epochs"] * config["n_splits"]

    top_cv_models = []  # (cv_score, epoch, model_path)
    top_loss_models = []  # (loss, epoch, model_path)
    
    best_cv = -float("inf") 
    best_loss = 100

    for epoch in range(config["epochs"]):
        epoch_start = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler, config)
        val_loss, cv = validate_one_epoch(model, valid_loader, device)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        cv_scores.append(cv)
        LR_records.append(scheduler.get_last_lr()[0])

        epoch_time = time.time() - epoch_start
        if epoch > 0:
            epoch_times.append(epoch_time)
            if len(epoch_times) > 10:
                epoch_times.pop(0)
        if len(epoch_times) > 0:
            avg_epoch_time = np.mean(epoch_times)
        else:
            avg_epoch_time = epoch_time

        progress = fold * config["epochs"] + (epoch + 1)
        remaining_epochs = all_progress - progress

        eta_seconds = avg_epoch_time * remaining_epochs
        if not np.isnan(eta_seconds) and not np.isinf(eta_seconds):
            eta_time = datetime.now() + timedelta(seconds=float(eta_seconds))
            eta_time = eta_time.replace(microsecond=0)
            days_diff = (eta_time.date() - datetime.now().date()).days
            eta_str = (
                f"T+{days_diff} " + eta_time.strftime("%H:%M:%S")
                if days_diff > 0 else eta_time.strftime("%H:%M:%S")
            )
        else:
            eta_str = "--:--:--"

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"[{now_str}]🧩[{progress/all_progress*100:6.2f}%] "
            f"Fold {fold+1}/{config['n_splits']} | "
            f"Epoch {epoch+1:03d}/{config['epochs']} | "
            f"Train={train_loss:.4f} | "
            f"Val={val_loss:.4f} | "
            f"CV={cv:.4f} | "
            f"LR={scheduler.get_last_lr()[0]:.6f} | "
            f"{epoch_time:6.2f}s/it | "
            f"ETA≈{eta_str}"
        )

        # --- 模型保存 ---
        if epoch >= config.get("min_epoch", 0):
            # 处理 CV Top3
            cv_prefix = save_dir / f"model_cv_epoch{epoch+1}_fold{fold+1}"
            _, top_cv_models = update_top_models(
                model, cv, epoch+1, fold+1, cv_prefix, top_cv_models, config, reverse=True
            )
            
            # 处理 Loss Top3
            loss_prefix = save_dir / f"model_loss_epoch{epoch+1}_fold{fold+1}"
            _, top_loss_models = update_top_models(
                model, val_loss, epoch+1, fold+1, loss_prefix, top_loss_models, config, reverse=False
            )


        top_models_info = {
            "fold": fold,
            "cv_top3": [
                {"rank": i+1, "cv": float(score), "epoch": ep, "path": path}
                for i, (score, ep, path) in enumerate(top_cv_models)
            ],
            "loss_top3": [
                {"rank": i+1, "loss": float(score), "epoch": ep, "path": path}
                for i, (score, ep, path) in enumerate(top_loss_models)
            ]
        }

        with open(save_dir / f"model_top3_fold{fold+1}.json", "w") as f:
            json.dump(top_models_info, f, indent=4)

        if should_stop(cv_scores, threshold=config["cv_stability_stop_threshold"]):
            print(f"🛑 Early Stop at Epoch {epoch+1}")
            break

    del train_loader, valid_loader, model, optimizer, scheduler, scaler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return train_losses, val_losses, cv_scores, LR_records


def save_metrics_to_excel(all_metrics, save_dir):
    """Combine metrics from all folds and save them into an Excel file."""
    max_epochs = max(len(m[0]) for m in all_metrics)
    df_out = pd.DataFrame({"Epoch": range(1, max_epochs + 1)})

    for i, (train, val, cv, lr) in enumerate(all_metrics, 1):
        df_out[f"Train_Fold{i}"] = train + [None] * (max_epochs - len(train))
        df_out[f"Val_Fold{i}"]   = val   + [None] * (max_epochs - len(val))
        df_out[f"CV_Fold{i}"]    = cv    + [None] * (max_epochs - len(cv))
        df_out[f"LR_Fold{i}"]    = lr    + [None] * (max_epochs - len(lr))

    out_path = Path(save_dir) / "fold_metrics.xlsx"
    df_out.to_excel(out_path, index=False)
    print(f"✅ Training log saved to: {out_path}")


def load_and_prepare_train_df():
    df_file_path = Path(DIRS["dir"]) / "train_with_folds_singleflow.csv"
    df = pd.read_csv(df_file_path)

    return df



def allocate_gpu_memory(memory_size_in_gb, device=None):
      
    # 计算张量的大小，假设每个元素是 float32（4 字节）
    tensor_size = memory_size_in_gb * 1024 * 1024 * 1024 // 4  # 每个 float32 元素占用 4 字节

    # 在 GPU 上创建一个全是零的大张量
    tensor = torch.zeros(tensor_size, dtype=torch.float32, device=device)

    # 打印当前 GPU 内存使用情况
    if device.type == 'cuda':
        print(f"已分配 {memory_size_in_gb}GB 的内存，当前 GPU 内存使用情况：")
        print(torch.cuda.memory_allocated(device) / 1024 / 1024 / 1024, "GB")
    
    return tensor



def train_with_groupkfold(df_train, save_dir, config, device):
    df = df_train.copy()
    
    if socket.gethostname() in ["user-PowerEdge-XE9680"]:
        tensor = allocate_gpu_memory(GPU_MEMORT, device)


    unique_groups = np.random.permutation(df["Sampling_Date"].unique())
    group_map = {g: i for i, g in enumerate(unique_groups)}
    df["GroupID"] = df["Sampling_Date"].map(group_map)

    all_metrics = []
    for fold in range(config["n_splits"]):
        metrics = train_one_fold(fold, df, df["GroupID"], save_dir, config, device)
        all_metrics.append(metrics)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_metrics_to_excel(all_metrics, save_dir)
    print("🎯 All folds training completed.")


config = {
    # ⚙️ basic train params
    "seed"                           : 43,  
    "epochs"                         : 200,
    "freeze_ratio"                   : 0.8,
    # "freeze_start_layer"             : 9, 
    "num_workers"                    : 4,
    "prefetch_factor"                : 3,

    "batch_size"                     : 4,
    
    "lr"                             : 1e-4,
    "n_splits"                       : 5,
    "img_size"                       : 1024,
    "img_ratio"                      : 2.0,  # width = img_size * img_ratio
    "backbone_name"                  : "vit_large_patch16_dinov3_qkvb.lvd1689m", #"vit_base_patch16_dinov3.lvd1689m",
    "cv_stability_stop_threshold"    : 0.005,
    "min_epoch"                      : 0,
    
    "mixup"                          : False,
    "cutmix"                         : False,
    "mixup_alpha"                    : 0.4,
    "cutmix_alpha"                   : 1.0,
    
    "scheduler_type"                 : "cosine",
    "n_constant"                     : 5,
    "n_cosine_1"                     : 25,
    "n_constant_2"                   : 5,
    "n_cosine_2"                     : 80,
    "min_lr"                         : 5e-6,

    "use_lora"                       : True,    # 开启 LoRA 微调
    "lora_r"                         : 8,       # 秩大小。样本少，建议 4 或 8；显存多且效果一般可升至 16
    "lora_alpha"                     : None,      # 通常设为 r 的 2 倍
    "lora_dropout"                   : 0.1,     # 增加正则化能力

    
    "weights": {
        "Dry_Green_g" : 0.1,
        "Dry_Clover_g": 0.1,
        "Dry_Dead_g"  : 0.1,
        "GDM_g"       : 0.2,
        "Dry_Total_g" : 0.5
    },


    "target_cols": [
        "Dry_Green_g",
        "Dry_Clover_g",
        "Dry_Dead_g",
        "GDM_g",
        "Dry_Total_g"
    ]
}

config["lora_alpha"] = int(config["lora_r"] * 2)
config["img_width"] = int(config["img_size"] * config.get("img_ratio", 1.0))


isTRAIN     = True
isPREDICT   = True
DEBUG       = True
# aimGPU_ID   = 6
aimGPU_ID   = None
GPU_MEMORT  = 0


if (
    __name__ == "__main__"
    and isTRAIN
    and socket.gethostname() in ["user-PowerEdge-XE9680", "hao-2", "I24fece71c700101943","simon-MS-7D94"]
):
    torch.multiprocessing.freeze_support()       
    setup_seed(config["seed"])        
    idx, device = select_free_gpu(aimID = aimGPU_ID)

    time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model_root = DIRS["model"]
    history_DIR = Path(model_root, time_str)

    os.makedirs(model_root, exist_ok=True)
    os.makedirs(history_DIR, exist_ok=True)

    if DEBUG:
        config["epochs"] = 1
        print(f"⚠️ DEBUG mode enabled — running only {config['epochs']} epochs\n")

    config["time_str"] = time_str
    config_path = history_DIR / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    print(f"✅ Config saved to: {config_path}")
    print("\n")
    print(config_to_str(config))
    print("\n")

    try:
        source_path = Path(__file__).resolve()
        target_path = history_DIR / source_path.name
        shutil.copy2(source_path, target_path)
        print(f"✅ Source code {source_path.name} copied to: {target_path}")
    except Exception as e:
        print(f"⚠️ Failed to copy source code: {e}")

    print("\n🚀🚀🚀 Starting training process... 🚀🚀🚀\n")

    df_train = load_and_prepare_train_df()
    if DEBUG:
        df_train = df_train.iloc[:30].copy()
        print(f"⚠️ DEBUG mode enabled — Using first {len(df_train)} training data ")
    
    train_with_groupkfold(
        df_train             = df_train,
        save_dir             = history_DIR,
        config               = config,
        device               = device
    )

    print(f"\n✅ Training completed! Results saved in: {history_DIR}")
    print("✅"*65)



# Infer
def load_and_prepare_test_df():
    df_file_path = Path(DIRS["dir"]) / "test.csv"
    df = pd.read_csv(df_file_path)

    df["ID"] = df["sample_id"].str.split("__").str[0]
    df = move_column_first(df, "ID")
    df["target"] = 0  

    df_targets = (
        df.pivot_table(index="ID", columns="target_name", values="target", aggfunc="first")
        .reset_index()
    )
    df_targets.columns.name = None

    df_meta = df[["ID", "image_path"]].drop_duplicates(subset="ID")
    df_test = pd.merge(df_meta, df_targets, on="ID", how="left")
    show_df_info(df_test, "df_test")

    print(f"✅ Test set loaded: {df_test.shape}")
    return df_test


def load_model_for_inference(model_path, device, config):
    model = MySingleStreamModel(config["backbone_name"], pretrained=False, config=config) # , pretrained=False
    model = model.to(device).to(memory_format=torch.channels_last)

    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    except Exception as e:
        print(f"🔄 切换到 weights_only=False 模式加载: {e}")
        state_dict = torch.load(model_path, map_location=device, weights_only=False)

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def predict_with_model(model, dataloader, device):
    preds_list = []
    model.eval()

    with torch.no_grad():
        for step, (images, targets) in enumerate(dataloader):
            images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            targets = targets.to(device, non_blocking=True)

            with torch.amp.autocast(device_type='cuda'):
                preds = model(images)
            
            preds_list.append(preds.cpu().numpy())
    return np.concatenate(preds_list, axis=0)


def predict_ensemble(df_test, pre_transform, transform, model_dir, device, config):
    model_paths = sorted(Path(model_dir).glob("*.pt"))
    assert len(model_paths) > 0, f"❌ No model files found in: {model_dir}"

    print(f"✅ Detected {len(model_paths)} 个模型:")
    for p in model_paths:
        print("   -", p.name)

    test_dataset = SingleStreamDataset(
        df_test,
        DIRS["dir"],
        config,
        pre_transform=pre_transform,
        transform=transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    fold_preds = []
    for fold, model_path in enumerate(model_paths):
        print(f"第 {fold+1}/{len(model_paths)} 模型，推理: {model_path.name}")
        model = load_model_for_inference(model_path, device, config)
        fold_pred = predict_with_model(model, test_loader, device)
        fold_preds.append(fold_pred)

    preds_mean = np.mean(fold_preds, axis=0)
    df_pred = pd.DataFrame(preds_mean, columns=config["target_cols"])
    
    df_pred["ID"] = df_test["ID"]
    df_pred = df_pred[["ID"] + config["target_cols"]]
    show_df_info(df_pred, "df_pred")

    return df_pred


def generate_submission(df_pred_final):
    print("\n\n\nGenerating Kaggle submission file ------------------------------------")

    print("📦 [Step 1] Original prediction DataFrame info:")
    print(df_pred_final.head(5))
    print(f"🔹 Data shape: {df_pred_final.shape}")

    ordered_target_cols = [
        "Dry_Green_g",  # 1️⃣
        "Dry_Clover_g", # 2️⃣
        "Dry_Dead_g",   # 3️⃣
        "GDM_g",        # 4️⃣
        "Dry_Total_g"   # 5️⃣
    ]
    print(f"\n📋 [Step 2] Target column order: {ordered_target_cols}")

    print("\n🔒 [Step 2.5] Clipping negative values to 0")
    for col in ordered_target_cols:
        if col in df_pred_final.columns:
            negative_count = (df_pred_final[col] < 0).sum()
            if negative_count > 0:
                print(f"   - {col}: {negative_count} negative values clipped to 0")
            df_pred_final[col] = df_pred_final[col].clip(lower=0)
        else:
            print(f"   ⚠️ Warning: Column {col} not found in DataFrame")

    print("\n📊 [Step 3] Preview after clipping:")
    print(df_pred_final[ordered_target_cols].head(5))

    df_submit = (
        df_pred_final.melt(
            id_vars="ID",
            value_vars=ordered_target_cols,
            var_name="target_name",
            value_name="target"
        )
    )
    print("\n📊 [Step 4] Preview of DataFrame after melt:")
    print(df_submit.head(10))
    print(f"🔹 Shape after melt: {df_submit.shape}")

    df_submit["sample_id"] = df_submit["ID"] + "__" + df_submit["target_name"]

    print("\n🔧 [Step 5] Preview after adding sample_id column:")
    print(df_submit[["ID", "target_name", "sample_id"]].head(10))

    df_submit = df_submit[["sample_id", "target"]]
    df_submit = df_submit.sort_values("sample_id").reset_index(drop=True)
    print("\n📋 [Step 6] Preview after reordering & sorting by sample_id:")
    print(df_submit.head(10))
    print(f"🔹 Final output shape: {df_submit.shape}")

    negative_in_final = (df_submit["target"] < 0).sum()
    if negative_in_final > 0:
        print(f"⚠️  Warning: Still found {negative_in_final} negative values in final submission!")
    else:
        print("✅ All values are non-negative in final submission")

    df_submit.to_csv("submission.csv", index=False)
    print("\n✅ Submission file generated: submission.csv")


def run_tta_prediction(df_test, model_dir, device, config):
    tta_transforms = get_tta_transforms(config)

    tta_names = list(tta_transforms.keys())
    print(f"\n✅ Detected {len(tta_names)} TTA modes: {tta_names}\n")

    all_preds = []
    for name, _ in tta_transforms.items():
        print(f"\n🚀 TTA mode: {name}")
        if name == "base":
            print("base no pre_transform")
            pre_transform = None
            transform = tta_transforms[name]
        else:
            pre_transform = tta_transforms[name]
            transform = tta_transforms["base"]
        df_pred = predict_ensemble(df_test, pre_transform, transform, model_dir, device, config)
        all_preds.append(df_pred[config["target_cols"]].values)

        # ✅ Show intermediate results
        print(f"Preview of predictions for TTA mode [{name}]:")
        print(df_pred.head())

        print(f"Number of collected TTA results: {len(all_preds)}")
        print(f"Current cumulative shape: {np.array(all_preds).shape}")
        print("\n\n")

    
    print("\nAggregating all TTA predictions:")
    print(f"Total {len(all_preds)} sets of predictions collected.")
    for i, arr in enumerate(all_preds):
        print(f"  └─ Prediction set {i+1}: \n{arr}")

    mean_preds = np.mean(all_preds, axis=0)
    print(f"Mean computation completed. Shape: {mean_preds.shape}, values:")
    print(mean_preds)

    df_pred_final = df_pred.copy()
    df_pred_final[config["target_cols"]] = mean_preds

    print("\n✅ TTA aggregation completed. Preview of final predictions (df_pred_final):")
    print(df_pred_final.head())

    return df_pred_final


if __name__ == "__main__" and isPREDICT:
    print("\n🧠 Starting prediction pipeline...")
    idx, device = select_free_gpu()
    print(idx,device)

    if socket.gethostname() in ["user-PowerEdge-XE9680", "hao-2","simon-MS-7D94"]:
        if DEBUG:
            df_test = load_and_prepare_train_df().iloc[:7]
        else:
            df_test = load_and_prepare_train_df()
    else:
        df_test = load_and_prepare_test_df()

    print(f"\nCurrent dataset shape: {df_test.shape}\n")

    if socket.gethostname() == "hao-2":
        model_dir = Path(DIRS["model"], "2025-11-06 17-05-25")
        if isTRAIN:
            model_dir = Path(DIRS["model"], time_str)
    elif socket.gethostname() == "user-PowerEdge-XE9680":
        model_dir = Path(
            "/data4/huangweigang/gh/csiro-biomass/Model_History",
            "dualstream-multihead_v14_A100_v2_maxepoch"
        )
        if isTRAIN:
            model_dir = Path(DIRS["model"], time_str)
    
    elif socket.gethostname() == "simon-MS-7D94":
        model_dir = Path("/home/simon/simondisk2/simon/CSIRO/Train_base_0/DualStream_multihead", "2025-11-12 17-32-38")
        if isTRAIN:
            model_dir = Path(DIRS["model"], time_str)
    else:
        model_dir = DIRS["model"]

    print(f"Model directory loaded: {model_dir}")

    df_pred_final = run_tta_prediction(df_test, model_dir, device, config)

    if socket.gethostname() in ["user-PowerEdge-XE9680", "hao-2","simon-MS-7D94"]:
        out_xlsx = Path(model_dir) / "df_pred_final.xlsx"
        df_pred_final.to_excel(out_xlsx, index=False)
        print(f"✅ Prediction results saved to: {out_xlsx}")

    generate_submission(df_pred_final)
    print("🎯 Prediction pipeline completed.")