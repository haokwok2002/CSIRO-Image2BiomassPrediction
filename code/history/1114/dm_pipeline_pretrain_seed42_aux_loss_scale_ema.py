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
from copy import deepcopy
from pathlib import Path
from datetime import datetime, timedelta
import random
import numpy as np
import pandas as pd

from tqdm import tqdm
from tabulate import tabulate

from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import r2_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler       

import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

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
        "model": Path("/home/simon/simondisk2/simon/CSIRO/train_Simon"),
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

else:
    dir = Path('/kaggle/input/csiro-biomass')
    DIRS = {
        "dir"  : dir,
        "train": Path(dir, "train"),
        "test" : Path(dir, "test"),
        "model": Path('/kaggle/input', "dm-pipeline-1114fix1"),
        "data" : Path("/kaggle/working/"),
    }







    # print("✅ file path ：")
    # for key, path in DIRS.items():
    #     print(f"{key:<12} : {path}")


# Helper functions
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

def select_free_gpu(threshold_mem_MB = 500, threshold_util = 20):
    """
    Automatically select an available GPU (works in both .py and Jupyter environments).
    """

    # === Internal function: query GPU info from nvidia-smi ===
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

    # === Main logic ===
    gpus = get_gpu_info()

    # Select GPUs with low memory and utilization
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

    # Print GPU information table
    print(tabulate(
        [[g["index"], g["name"], f"{g['mem_used_MB']}/{g['mem_total_MB']} MB",
          f"{g['util_%']}%", f"{g['temp_C']}°C", f"{g['power_W']}W"]
         for g in gpus],
        headers=["GPU", "Name", "Memory", "Util", "Temp", "Power"],
        tablefmt="grid"
    ))

    idx = selected["index"]
    device_name = f"cuda:{idx}"

    # Detect if running inside a Jupyter Notebook
    try:
        from IPython import get_ipython
        in_notebook = get_ipython() is not None
    except Exception:
        in_notebook = False

    if not in_notebook:
        # ✅ Safe to set environment variable in normal Python script
        os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n✅ Selected GPU {idx} {reason}")
        print(f"Current device: {device} (logical GPU {idx})\n")
    else:
        # ⚠️ In notebook environments, do not modify environment variables
        device = torch.device(device_name if torch.cuda.is_available() else "cpu")
        print(f"\n⚠️ Detected Jupyter environment — not modifying CUDA_VISIBLE_DEVICES.")
        print(f"✅ Using device: {device_name} {reason}\n")

    return idx, device


# Model 
class MyDualStreamModel(nn.Module):
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

        # 1️⃣ Backbone
        if socket.gethostname() == "user-PowerEdge-XE9680":
            model_path = f"/data4/huangweigang/gh/timm_model/{backbone_name}.pth"
            self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
            try:
                state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
            except TypeError:
                state_dict = torch.load(model_path, map_location='cpu')
            self.backbone.load_state_dict(state_dict, strict=False)
        else:
            self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)

        in_dim = self.backbone.num_features

        # 2️⃣ Freeze partial backbone parameters
        params = list(self.backbone.parameters())
        freeze_until = int(len(params) * config["freeze_ratio"])
        for i, p in enumerate(params):
            p.requires_grad = i >= freeze_until     # freeze front part, train later part

        # 3️⃣ Dual-stream feature fusion
        self.fusion_dim = in_dim * 2
        print("feature fusion dim = ", self.fusion_dim)

        # 4️⃣ Three output heads
        def make_head():
            return nn.Sequential(
                nn.Linear(self.fusion_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )

        # Model predicts three independent targets
        self.head_green  = make_head()   # Dry_Green_g
        self.head_clover = make_head()   # Dry_Clover_g
        self.head_dead   = make_head()   # Dry_Dead_g

        # 5️⃣ Weights configuration
        self.weights = config["weights"]
        self.target_cols = config["target_cols"]
        # Optional per-target loss scaling to offset target range differences
        self.loss_scales_dict = config.get("loss_scales", {})
        # 🔹 Auxiliary tasks config
        self.aux_enabled = bool(config.get("aux_enabled", False))
        self.aux_class_cols = config.get("aux_class_cols", [])
        self.aux_reg_cols = config.get("aux_reg_cols", [])
        self.aux_cls_num_classes = config.get("aux_cls_num_classes", {})
        self.aux_loss_weights = config.get("aux_loss_weights", {"main": 1.0, "cls": 0.2, "reg": 0.2})
        self.aux_reg_scales = config.get("aux_reg_scales", {})

        # Build auxiliary heads if enabled
        self.cls_heads = nn.ModuleDict()
        self.reg_heads = nn.ModuleDict()
        if self.aux_enabled:
            for col in self.aux_class_cols:
                num_classes = int(self.aux_cls_num_classes.get(col, 0))
                if num_classes > 0:
                    self.cls_heads[col] = nn.Sequential(
                        nn.Linear(self.fusion_dim, 256),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(256, num_classes)
                    )
            for col in self.aux_reg_cols:
                self.reg_heads[col] = nn.Sequential(
                    nn.Linear(self.fusion_dim, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, 1)
                )

    def forward(self, img_left, img_right):
        # Extract features
        feat_left  = self.backbone(img_left)
        feat_right = self.backbone(img_right)
        fused = torch.cat([feat_left, feat_right], dim=1)

        # Predict three primary targets
        G = self.head_green(fused)
        C = self.head_clover(fused)
        D = self.head_dead(fused)

        # Derived targets
        GDM   = G + C
        Total = G + C + D

        preds = torch.cat([G, C, D, GDM, Total], dim=1)  # [B, 5]
        return preds

    def forward_with_aux(self, img_left, img_right):
        """Return main preds and auxiliary outputs (if enabled)."""
        feat_left  = self.backbone(img_left)
        feat_right = self.backbone(img_right)
        fused = torch.cat([feat_left, feat_right], dim=1)

        G = self.head_green(fused)
        C = self.head_clover(fused)
        D = self.head_dead(fused)
        GDM   = G + C
        Total = G + C + D
        preds = torch.cat([G, C, D, GDM, Total], dim=1)  # [B,5]

        aux_outputs = {"cls": {}, "reg": {}}
        if self.aux_enabled:
            for col, head in self.cls_heads.items():
                aux_outputs["cls"][col] = head(fused)  # logits [B, num_classes]
            for col, head in self.reg_heads.items():
                aux_outputs["reg"][col] = head(fused)  # [B,1]
        return preds, aux_outputs


    def compute_loss(self, preds, targets, aux_outputs=None, aux_labels=None):
        """
        方法 2：loss scaling 抵消目标范围差异
        loss = Σ w_t * ((y_t - ŷ_t) / scale_t)^2  （对 batch 求均值）
        """
        preds = preds.view(-1, 5)
        targets = targets.view(-1, 5)

        # Build weights/scales tensors in consistent target order
        device = preds.device
        weight_list = [float(self.weights.get(name, 1.0)) for name in self.target_cols]
        scale_list = [float(self.loss_scales_dict.get(name, 1.0)) for name in self.target_cols]

        weights = torch.tensor(weight_list, dtype=torch.float32, device=device).view(1, -1)
        scales  = torch.tensor(scale_list,  dtype=torch.float32, device=device).view(1, -1)
        scales  = torch.clamp(scales, min=1e-6)

        norm_err_sq = ((preds - targets) / scales) ** 2
        weighted = weights * norm_err_sq
        main_loss = torch.mean(torch.sum(weighted, dim=1))

        # Auxiliary losses
        if not (self.aux_enabled and aux_outputs is not None and aux_labels is not None):
            return main_loss

        total_loss = self.aux_loss_weights.get("main", 1.0) * main_loss

        # Classification aux: CrossEntropyLoss per column
        if "cls" in aux_outputs and "cls" in aux_labels:
            ce = nn.CrossEntropyLoss()
            for col, logits in aux_outputs["cls"].items():
                if col in aux_labels["cls"]:
                    y = aux_labels["cls"][col].to(device, non_blocking=True).view(-1).long()
                    total_loss = total_loss + self.aux_loss_weights.get("cls", 0.2) * ce(logits, y)

        # Regression aux: scaled MSE per column
        if "reg" in aux_outputs and "reg" in aux_labels:
            for col, pred in aux_outputs["reg"].items():
                if col in aux_labels["reg"]:
                    y = aux_labels["reg"][col].to(device, non_blocking=True).view(-1, 1).float()
                    scale = float(self.aux_reg_scales.get(col, 1.0))
                    scale = max(scale, 1e-6)
                    total_loss = total_loss + self.aux_loss_weights.get("reg", 0.2) * torch.mean(((pred - y) / scale) ** 2)

        return total_loss

# ========= EMA =========
class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999, device=None):
        self.decay = float(decay)
        self.ema = deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        if device is not None:
            self.ema.to(device).to(memory_format=torch.channels_last)

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        ema_params = dict(self.ema.named_parameters())
        model_params = dict(model.named_parameters())
        for name, param in model_params.items():
            if name in ema_params:
                ema_param = ema_params[name]
                ema_param.data.mul_(d).add_(param.data, alpha=1.0 - d)
        # Keep buffers (e.g., BatchNorm stats) in sync
        for ema_buf, buf in zip(self.ema.buffers(), model.buffers()):
            ema_buf.copy_(buf)

# Dataset 
class DualStreamDataset(Dataset):
    def __init__(self, df, image_dir, config, transform=None):
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
        self.transform = transform
        # Aux setup
        self.aux_enabled = bool(config.get("aux_enabled", False))
        self.aux_class_cols = config.get("aux_class_cols", [])
        self.aux_reg_cols = config.get("aux_reg_cols", [])
        # Mapping dicts for classes: {col: {label_str: idx}}
        self.aux_cls_label_maps = config.get("aux_cls_label_maps", {})

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = Path(self.image_dir, str(row["image_path"]))
        
        # ====== 1️⃣ Safe image loading ======
        if not img_path.exists():
            print(f"⚠️ Image not found: {img_path}")
            image = np.zeros((1000, 2000, 3), dtype=np.uint8)
        else:
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"⚠️ Failed to read image: {img_path} ({e})")
                image = np.zeros((1000, 2000, 3), dtype=np.uint8)

        # ====== 2️⃣ Ensure conversion to NumPy array ======
        image = np.array(image)
        h, w, _ = image.shape
        mid = w // 2
        
        # Split image into left and right patches
        img_left = image[:, :mid]
        img_right = image[:, mid:]


        # ====== 4️⃣ Apply Albumentations transforms ======
        if self.transform:
            img_left = self.transform(image=img_left)["image"]
            img_right = self.transform(image=img_right)["image"]


        # ====== 5️⃣ Return result ======
        if self.target_cols is not None:
            targets = torch.tensor(
                row[self.target_cols].astype(float).values,
                dtype=torch.float32
            )
            if not self.aux_enabled:
                return img_left, img_right, targets
            # Build auxiliary labels
            aux_labels = {"cls": {}, "reg": {}}
            # Classification aux: map string labels to indices (unknown → 'UNK')
            for col in self.aux_class_cols:
                val = row.get(col, None)
                map_dict = self.aux_cls_label_maps.get(col, {})
                if pd.isna(val):
                    val = "UNK"
                idx = int(map_dict.get(str(val), map_dict.get("UNK", 0)))
                aux_labels["cls"][col] = torch.tensor(idx, dtype=torch.long)
            # Regression aux: raw floats
            for col in self.aux_reg_cols:
                v = float(row.get(col, np.nan))
                if np.isnan(v):
                    v = 0.0
                aux_labels["reg"][col] = torch.tensor(v, dtype=torch.float32)
            return img_left, img_right, targets, aux_labels
        else:
            return img_left, img_right


# Albumentations
def get_train_transforms(size):
    
    return A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_valid_transforms(size):
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_tta_transforms(size):
    return {
        "base": A.Compose([
            A.Resize(size, size),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        "hflip": A.Compose([
            A.Resize(size, size),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        "vflip": A.Compose([
            A.Resize(size, size),
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        # "rot90": A.Compose([
        #     A.Resize(size, size),
        #     A.RandomRotate90(p=1.0),
        #     A.Normalize(mean=[0.485, 0.456, 0.406],
        #                 std=[0.229, 0.224, 0.225]),
        #     ToTensorV2()
        # ]),
        # "brightness": A.Compose([
        #     A.Resize(size, size),
        #     A.RandomBrightnessContrast(brightness_limit=0.1,
        #                                contrast_limit=0.1, p=1.0),
        #     A.Normalize(mean=[0.485, 0.456, 0.406],
        #                 std=[0.229, 0.224, 0.225]),
        #     ToTensorV2()
        # ]),
        # "gamma": A.Compose([
        #     A.Resize(size, size),
        #     A.RandomGamma(gamma_limit=(90, 110), p=1.0),
        #     A.Normalize(mean=[0.485, 0.456, 0.406],
        #                 std=[0.229, 0.224, 0.225]),
        #     ToTensorV2()
        # ]),
    }




# 🔹 Compute Weighted R² Score 
def compute_cv_score(all_preds, all_targets):
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Five target columns and their corresponding official weights
    target_cols = ["Dry_Green_g", "Dry_Clover_g", "Dry_Dead_g", "GDM_g", "Dry_Total_g"]
    weights = np.array([0.1, 0.1, 0.1, 0.2, 0.5])

    # Flatten all target values and predictions    
    y_true_flat = np.concatenate([targets[i, :] for i in range(targets.shape[0])])
    y_pred_flat = np.concatenate([preds[i, :] for i in range(preds.shape[0])])
    w_flat = np.concatenate([np.full_like(targets[:, i], weights[i]) for i in range(5)])

    # Compute global weighted mean
    y_mean = np.sum(w_flat * y_true_flat) / np.sum(w_flat)

    # Compute weighted residual and total sum of squares
    ss_res = np.sum(w_flat * (y_true_flat - y_pred_flat) ** 2)
    ss_tot = np.sum(w_flat * (y_true_flat - y_mean) ** 2)

    # Final global weighted R² score (Kaggle-style)
    r2_global = 1 - ss_res / ss_tot
    return r2_global

# 🔹 Single-epoch training
def train_one_epoch(model, dataloader, optimizer, device, scaler, model_ema: "ModelEMA" = None):
    model.train()
    running_loss = []

    start_epoch = time.time()

    # End time of the previous batch (used to measure data loading time)
    prev_end = start_epoch  

    for step, batch in enumerate(dataloader):
        t_load = time.time()  # Time right after fetching a batch from dataloader
        data_load_time = t_load - prev_end

        # ====== Move data to GPU ======
        t0 = time.time()
        if len(batch) == 3:
            img_left, img_right, targets = batch
            aux_labels = None
        else:
            img_left, img_right, targets, aux_labels = batch
        img_left = img_left.to(device, non_blocking=True)
        img_right = img_right.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        # move aux dict tensors to device lazily in loss
        t1 = time.time()

        # ====== Forward + Backward ======
        optimizer.zero_grad(set_to_none=True)
        # ✅ AMP mixed precision context
        with torch.amp.autocast(device_type='cuda'):
            # Use aux-aware forward if available
            if hasattr(model, "forward_with_aux") and model.aux_enabled:
                preds, aux_outputs = model.forward_with_aux(img_left, img_right)
                loss = model.compute_loss(preds, targets, aux_outputs=aux_outputs, aux_labels=aux_labels)
            else:
                preds = model(img_left, img_right)
                loss = model.compute_loss(preds, targets)
        t2 = time.time()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # ==== EMA update after optimizer step ====
        if model_ema is not None:
            model_ema.update(model)
        t3 = time.time()

        running_loss.append(loss.item())
        # Used to calculate data loading time for the next batch
        prev_end = t3  

        # Print timing breakdown every N steps
        # if step  == 0  or step  == 1:
        # print(
        #     f"[TRAIN] Step {step:4d} | "
        #     f"data load: {data_load_time*1000:.1f} ms | "
        #     f"to(device): {(t1-t0)*1000:.1f} ms | "
        #     f"forward+loss: {(t2-t1)*1000:.1f} ms | "
        #     f"backward+opt: {(t3-t2)*1000:.1f} ms | "
        #     f"total: {(t3-t_load)*1000:.1f} ms"
        # )

    end_epoch = time.time()
    epoch_time = end_epoch - start_epoch
    avg_batch_time = epoch_time / len(dataloader)

    # print(f"[TRAIN] Epoch total time: {epoch_time:.2f}s | "
    #       f"{len(dataloader)} batches | {avg_batch_time:.3f}s/batch")

    return float(np.mean(running_loss))

# 🔹 Single-epoch validation + local CV
def validate_one_epoch(model, dataloader, device):
    model.eval()
    val_losses, all_preds, all_targets = [], [], []
    mae_sum = np.zeros(5, dtype=np.float64)
    n_samples = 0

    start_epoch = time.time()
    # End time of the previous batch (for measuring data loading time)
    prev_end = start_epoch 

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            t_load = time.time()  # Time when dataloader provides the current batch
            data_load_time = t_load - prev_end

            # ====== Move data to GPU ======
            t0 = time.time()
            if len(batch) == 3:
                img_left, img_right, targets = batch
                aux_labels = None
            else:
                img_left, img_right, targets, aux_labels = batch
            img_left = img_left.to(device, non_blocking=True)
            img_right = img_right.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            t1 = time.time()

            # ====== Forward inference + compute loss ======
            if hasattr(model, "forward_with_aux") and model.aux_enabled:
                preds, aux_outputs = model.forward_with_aux(img_left, img_right)
                vloss = model.compute_loss(preds, targets, aux_outputs=aux_outputs, aux_labels=aux_labels)
                val_loss = float(vloss.item())
            else:
                preds = model(img_left, img_right)
                val_loss = model.compute_loss(preds, targets).item()
            t2 = time.time()

            val_losses.append(val_loss)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            # accumulate per-target absolute errors for MAE
            abs_err = torch.abs(preds - targets).detach().cpu().numpy()  # [B,5]
            mae_sum += abs_err.sum(axis=0)
            n_samples += abs_err.shape[0]

            # Used to compute data_load_time for the next batch
            prev_end = t2 

            # Print timing breakdown every N steps
            # if step  == 0  or step  == 1:
            # print(
            #     f"[VAL] Step {step:4d} | "
            #     f"data load: {data_load_time*1000:.1f} ms | "
            #     f"to(device): {(t1 - t0)*1000:.1f} ms | "
            #     f"forward+loss: {(t2 - t1)*1000:.1f} ms | "
            #     f"total: {(t2 - t_load)*1000:.1f} ms"
            # )

    end_epoch = time.time()
    epoch_time = end_epoch - start_epoch
    avg_val_loss = float(np.mean(val_losses))
    r2_global = compute_cv_score(all_preds, all_targets)

    # 🔎 MAE per target (raw)
    if n_samples > 0:
        mae_per_target = mae_sum / float(n_samples)
        names = config["target_cols"]
        mae_str = ", ".join([f"{names[i]}={mae_per_target[i]:.3f}" for i in range(5)])
        print(f"[VAL] MAE per target: {mae_str}")

    # print(
    #     f"[VAL] Epoch total time: {epoch_time:.2f}s | "
    #     f"{len(dataloader)} batches | {epoch_time / len(dataloader):.3f}s/batch"
    # )

    return avg_val_loss, r2_global

# 🔹 Compute MAE per target on an arbitrary dataloader
def compute_mae_per_target(model, dataloader, device):
    model.eval()
    mae_sum = np.zeros(5, dtype=np.float64)
    n_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                img_left, img_right, targets = batch
            else:
                img_left, img_right, targets, _ = batch
            img_left = img_left.to(device, non_blocking=True)
            img_right = img_right.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            preds = model(img_left, img_right)
            abs_err = torch.abs(preds - targets).detach().cpu().numpy()
            mae_sum += abs_err.sum(axis=0)
            n_samples += abs_err.shape[0]
    if n_samples == 0:
        return np.zeros(5, dtype=np.float64)
    return mae_sum / float(n_samples)

# 🔹 基于 GroupKFold 的 DataLoader 构建
def get_fold_loaders(df, fold, groups, config, device):
    """为单个 fold 构建训练和验证 DataLoader（各自独立 transform）"""
    gkf = GroupKFold(n_splits=config["n_splits"])
    train_idx = val_idx = None
    for f, (tr_idx, va_idx) in enumerate(gkf.split(df, groups=groups)):
        if f == fold:
            train_idx, val_idx = tr_idx, va_idx
            break
    assert train_idx is not None, "Fold 索引越界或分割失败"

    train_df = df.iloc[train_idx].reset_index(drop=True)
    valid_df = df.iloc[val_idx].reset_index(drop=True)


    # make Dataset and DataLoader
    train_dataset = DualStreamDataset(train_df, DIRS["dir"], config, transform=get_train_transforms(config["img_size"]))
    valid_dataset = DualStreamDataset(valid_df, DIRS["dir"], config, transform=get_valid_transforms(config["img_size"]))


    # get CPU core
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
        num_workers=max(1, num_workers // 2),  # 验证集线程少一点即可
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=True
    )

    return train_loader, valid_loader

# 🔹 Tianchi: fold loaders (absolute image paths supported)
def get_tianchi_fold_loaders(df, fold, config, device):
    """
    Build DataLoaders for Tianchi pretraining using GroupKFold on
    (acquisition_year, seasonal_harvest_no) derived groups.
    Assumes df['image_path'] is absolute path strings.
    """
    n_splits = int(config.get("tianchi_n_splits", 5))
    gkf = GroupKFold(n_splits=n_splits)
    train_idx = val_idx = None
    groups = df["TianchiGroupID"]
    for f, (tr_idx, va_idx) in enumerate(gkf.split(df, groups=groups)):
        if f == fold:
            train_idx, val_idx = tr_idx, va_idx
            break
    assert train_idx is not None, "Tianchi fold index out of range"

    train_df = df.iloc[train_idx].reset_index(drop=True)
    valid_df = df.iloc[val_idx].reset_index(drop=True)

    # absolute paths → use root "/" as base; Path(base, abs_path) resolves to abs_path
    base_dir = Path("/")
    train_dataset = DualStreamDataset(train_df, base_dir, config, transform=get_train_transforms(config["img_size"]))
    valid_dataset = DualStreamDataset(valid_df, base_dir, config, transform=get_valid_transforms(config["img_size"]))

    num_workers = config["num_workers"]
    prefetch_factor = config["prefetch_factor"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=True
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

# 🔹 Early stopping
def should_stop(cv_scores, window=20, threshold=0.001):
    """Check if recent CV score fluctuations are stable enough to trigger early stopping."""
    
    # Apply early stopping only after at least 100 epochs
    if len(cv_scores) < config["min_epoch"] :  
        return False
    window_scores = cv_scores[-window:]
    diff = max(window_scores) - min(window_scores)
    if diff < threshold:
        print("diff =", diff, f"< {threshold}")
    return diff < threshold

# 🔹 Single-Fold Training
def train_one_fold(fold, df, groups, save_dir, config, device):
    """Train a single fold and return performance metrics for that fold."""
    # === DataLoader ===
    train_loader, valid_loader = get_fold_loaders(df, fold, groups, config, device)

    # === Model and Optimizer ===
    model = MyDualStreamModel(config["backbone_name"], pretrained=True, config=config)
    model = model.to(device).to(memory_format=torch.channels_last)
    # === EMA wrapper ===
    use_ema = bool(config.get("use_ema", False))
    ema_decay = float(config.get("ema_decay", 0.999))
    model_ema = ModelEMA(model, decay=ema_decay, device=device) if use_ema else None
    # Optionally load Tianchi pretrain
    if config.get("is_load_pretrain_tianchi", False) and config.get("pretrain_tianchi_ckpt_path"):
        ckpt_path = config.get("pretrain_tianchi_ckpt_path")
        try:
            state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
        except TypeError:
            state_dict = torch.load(ckpt_path, map_location=device)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"🔁 Loaded Tianchi pretrain from: {ckpt_path}")
        if missing:
            print(f"   Missing keys: {len(missing)}")
        if unexpected:
            print(f"   Unexpected keys: {len(unexpected)}")

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=config["lr"]/100)
    scaler = torch.amp.GradScaler(device="cuda")      # version:2.xx


    # === Metric Caches ===
    train_losses, val_losses, cv_scores, LR_records = [], [], [], []
    epoch_times = []  # record recent 10 epoch durations
    all_progress = config["epochs"] * config["n_splits"]
    
    # Initialize global best metric
    best_cv = -float("inf") 
    best_model_path = None

    # === Main Training Loop ===
    for epoch in range(config["epochs"]):
        epoch_start = time.time()
        
        # --- Train and Validate ---
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler, model_ema=model_ema)
        # choose eval model (EMA or raw)
        eval_with_ema = bool(config.get("eval_with_ema", use_ema))
        model_for_eval = model_ema.ema if (use_ema and eval_with_ema) else model
        val_loss, cv = validate_one_epoch(model_for_eval, valid_loader, device)
        scheduler.step()

        # --- Record Metrics ---
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        cv_scores.append(cv)
        LR_records.append(scheduler.get_last_lr()[0])


        # --- ETA Calculation (safe) ---
        epoch_time = time.time() - epoch_start
        if epoch > 0:
            epoch_times.append(epoch_time)
            if len(epoch_times) > 10:
                epoch_times.pop(0)
        if len(epoch_times) > 0:
            avg_epoch_time = np.mean(epoch_times)
        else:
            avg_epoch_time = epoch_time

        if np.isnan(avg_epoch_time) or np.isinf(avg_epoch_time):
            avg_epoch_time = epoch_time

        progress = fold * config["epochs"] + (epoch + 1)
        remaining_epochs = all_progress - progress

        eta_seconds = avg_epoch_time * remaining_epochs
        if not np.isnan(eta_seconds) and not np.isinf(eta_seconds):
            eta_time = datetime.now() + timedelta(seconds=float(eta_seconds))
            eta_time = eta_time.replace(microsecond=0)
            # 若跨天则在前缀中标注
            days_diff = (eta_time.date() - datetime.now().date()).days
            eta_str = (
                f"T+{days_diff} " + eta_time.strftime("%H:%M:%S")
                if days_diff > 0 else eta_time.strftime("%H:%M:%S")
            )
        else:
            eta_str = "--:--:--"

        now_str = datetime.now().strftime("%H:%M:%S")



        # --- Logging ---
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


        min_epoch = config["min_epoch"] 

        # --- Save Best Model ---
        if epoch >= min_epoch and cv > best_cv:
            best_cv = cv
            best_epoch = epoch + 1
            best_model_path = save_dir / f"model_best_fold{fold+1}.pt"
            # save evaluated weights (EMA or raw)
            torch.save(model_for_eval.state_dict(), best_model_path)

            # Save best score record
            best_info = {
                "fold": fold,
                "epoch": best_epoch,
                "cv": round(float(cv), 6),
                "path": str(best_model_path),
                "evaluated_with_ema": bool(use_ema and eval_with_ema),
                "ema_decay": float(ema_decay) if use_ema else None
            }
            with open(save_dir / f"model_best_fold{fold+1}.json", "w") as f:
                json.dump(best_info, f, indent=4)

            print(f"🌟 更新最佳模型！Fold {fold+1} | Epoch {best_epoch} | r2_global={cv:.4f}")
            # ➕ 输出训练集 target MAE
            train_mae = compute_mae_per_target(model_for_eval, train_loader, device)
            names = config["target_cols"]
            train_mae_str = ", ".join([f"{names[i]}={train_mae[i]:.3f}" for i in range(5)])
            print(f"[TRAIN] MAE per target (best update): {train_mae_str}")

        # --- Early Stopping ---
        if should_stop(cv_scores, threshold=config["cv_stability_stop_threshold"]):
            print(f"🛑 Early Stop at Epoch {epoch+1}")
            break

        # --- Periodic Checkpoint Saving ---
        if epoch >= min_epoch and (epoch+1) % config["save_interval"] == 0:
            torch.save(model_for_eval.state_dict(), save_dir / f"fold{fold+1}_epoch{epoch+1}.pt")

    # === Save Final Model ===
    final_eval_model = model_ema.ema if (use_ema and bool(config.get("eval_with_ema", use_ema))) else model
    torch.save(final_eval_model.state_dict(), save_dir / f"fold{fold+1}_epoch{epoch+1}_final.pt")

    # === Cleanup GPU Memory ===
    del train_loader, valid_loader, model, optimizer, scheduler, scaler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return train_losses, val_losses, cv_scores, LR_records

# 🔹 Save training logs
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

# Load and preprocess training data (CSIRO)
def load_and_prepare_train_df():
    # 1️⃣ Read raw CSV file
    df_file_path = Path(DIRS["dir"]) / "train.csv"
    df = pd.read_csv(df_file_path)

    # 2️⃣ Extract unique ID (e.g., "ID1011485656__Dry_Green_g" → "ID1011485656")
    df["ID"] = df["sample_id"].str.split("__").str[0]

    # 3️⃣ Move ID column to the front
    
    df = move_column_first(df, "ID")
    # show_df_info(df, "df")

    # 4️⃣ Pivot target values (long → wide format)
    df_targets = (
        df.pivot_table(
            index="ID",
            columns="target_name",
            values="target",
            aggfunc="first"
        )
        .reset_index()
    )
    df_targets.columns.name = None  # remove multi-index column names

    # 5️⃣ Extract metadata (one row per ID)
    meta_cols = [
        "ID", "image_path", "Sampling_Date", "State",
        "Species", "Pre_GSHH_NDVI", "Height_Ave_cm"
    ]
    df_meta = df[meta_cols].drop_duplicates(subset="ID")
    # show_df_info(df_meta, "df_meta")

    # 6️⃣ Merge metadata with target values
    df_train = pd.merge(df_meta, df_targets, on="ID", how="left")
    show_df_info(df_train, "df_train")

    return df_train

# 🔹 Compute per-target loss scales from training data
def compute_loss_scales_from_df(df, target_cols, eps=1e-6):
    """
    Compute scale_t for each target to offset range differences.
    Default: population std; fallback to 1.0 if too small.
    """
    scales = {}
    for col in target_cols:
        arr = df[col].astype(float).values
        std = float(np.std(arr)) if arr.size > 0 else 0.0
        scales[col] = std if std > eps else 1.0
    print("✅ Loss scales (per target):", {k: round(v, 3) for k, v in scales.items()})
    return scales

# 🔹 Build auxiliary task config from training dataframe
def setup_aux_config_from_df(df, config):
    """
    Prepare auxiliary tasks:
    - Classification: State, Species
    - Regression: Pre_GSHH_NDVI, Height_Ave_cm
    Will mutate the provided config with aux settings.
    """
    class_cols = ["State", "Species"]
    reg_cols = ["Pre_GSHH_NDVI", "Height_Ave_cm"]

    # Build label maps
    cls_label_maps = {}
    cls_num_classes = {}
    for col in class_cols:
        values = df[col].astype(str).fillna("UNK")
        uniques = sorted(values.unique().tolist())
        if "nan" in uniques:
            uniques = [u for u in uniques if u != "nan"]
        if "UNK" not in uniques:
            uniques.insert(0, "UNK")
        label_map = {str(v): i for i, v in enumerate(uniques)}
        cls_label_maps[col] = label_map
        cls_num_classes[col] = len(uniques)

    # Regression scales
    reg_scales = {}
    for col in reg_cols:
        arr = df[col].astype(float).values
        std = float(np.std(arr)) if arr.size > 0 else 0.0
        reg_scales[col] = std if std > 1e-6 else 1.0

    # Loss weights
    aux_loss_weights = config.get("aux_loss_weights", {"main": 1.0, "cls": 0.2, "reg": 0.2})

    # Update config
    config["aux_enabled"] = True
    config["aux_class_cols"] = class_cols
    config["aux_reg_cols"] = reg_cols
    config["aux_cls_label_maps"] = cls_label_maps
    config["aux_cls_num_classes"] = cls_num_classes
    config["aux_reg_scales"] = reg_scales
    config["aux_loss_weights"] = aux_loss_weights
    print("✅ Aux config prepared:",
          {"class_cols": class_cols, "reg_cols": reg_cols,
           "num_classes": cls_num_classes, "reg_scales": reg_scales})
    return config

# Load and preprocess Tianchi data, mapped to CSIRO targets (absolute image paths)
def load_and_prepare_tianchi_df(config):
    csv_path = Path(config.get("tianchi_csv_path", "/home/simon/simondisk2/simon/CSIRO/material/Tianchi/biomass_train_data.csv"))
    img_dir = Path(config.get("tianchi_image_dir", "/home/simon/simondisk2/simon/CSIRO/material/Tianchi/GrassClover Dataset02_datasets/train"))
    t = pd.read_csv(csv_path, sep=";")

    # Fill clover as sum if needed
    clover_sum = t[["dry_white_clover", "dry_red_clover"]].fillna(0).sum(axis=1)
    dry_clover = t["dry_clover"].fillna(0) + clover_sum.where(t["dry_clover"].isna(), 0)

    dry_green  = t["dry_grass"].fillna(0)
    dry_total  = t["dry_total"].fillna(0)
    # Approximate Dead using weeds (fallback: residual)
    residual = (dry_total - (dry_green + dry_clover)).clip(lower=0)
    dry_dead  = t["dry_weeds"].fillna(residual)

    df = pd.DataFrame({
        "ID": t["image_file_name"].str.replace(".jpg","", regex=False),
        "image_path": (img_dir.as_posix() + "/" + t["image_file_name"]),
        "Dry_Green_g": dry_green,
        "Dry_Clover_g": dry_clover,
        "Dry_Dead_g": dry_dead,
        "GDM_g": dry_green + dry_clover,
        "Dry_Total_g": dry_total,
        "acquisition_year": t.get("acquisition_year", np.nan),
        "seasonal_harvest_no": t.get("seasonal_harvest_no", np.nan),
    })
    # Group for CV split
    df["TianchiGroupID"] = df["acquisition_year"].astype(str) + "_" + df["seasonal_harvest_no"].astype(str)
    show_df_info(df, "df_tianchi_mapped")
    return df

# 🔹 Single-Fold Training for Tianchi (pretraining)
def train_one_fold_tianchi(fold, df, save_dir, config, device):
    """Pretrain a single fold on Tianchi and return metrics."""
    train_loader, valid_loader = get_tianchi_fold_loaders(df, fold, config, device)

    model = MyDualStreamModel(config["backbone_name"], pretrained=True, config=config)
    model = model.to(device).to(memory_format=torch.channels_last)
    # EMA for pretraining if enabled
    use_ema = bool(config.get("use_ema", False))
    ema_decay = float(config.get("ema_decay", 0.999))
    model_ema = ModelEMA(model, decay=ema_decay, device=device) if use_ema else None

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["pretrain_tianchi_epochs"], eta_min=config["lr"]/100)
    scaler = torch.amp.GradScaler(device="cuda")

    train_losses, val_losses, cv_scores, LR_records = [], [], [], []
    epoch_times = []

    best_cv = -float("inf")
    best_model_path = None

    for epoch in range(config["pretrain_tianchi_epochs"]):
        epoch_start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler, model_ema=model_ema)
        eval_with_ema = bool(config.get("eval_with_ema", use_ema))
        model_for_eval = model_ema.ema if (use_ema and eval_with_ema) else model
        val_loss, cv = validate_one_epoch(model_for_eval, valid_loader, device)
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

        now_str = datetime.now().strftime("%H:%M:%S")
        print(
            f"[{now_str}]🧩[Pretrain Tianchi] "
            f"Epoch {epoch+1:03d}/{config['pretrain_tianchi_epochs']} | "
            f"Train={train_loss:.4f} | Val={val_loss:.4f} | CV={cv:.4f} | "
            f"LR={scheduler.get_last_lr()[0]:.6f}"
        )

        min_epoch = min(config["min_epoch"], config["pretrain_tianchi_epochs"]//2)
        if epoch >= min_epoch and cv > best_cv:
            best_cv = cv
            best_epoch = epoch + 1
            best_model_path = save_dir / f"pretrain_tianchi_best.pt"
            torch.save(model_for_eval.state_dict(), best_model_path)
            best_info = {
                "phase": "pretrain_tianchi",
                "fold": fold,
                "epoch": best_epoch,
                "cv": round(float(cv), 6),
                "path": str(best_model_path),
                "evaluated_with_ema": bool(use_ema and eval_with_ema),
                "ema_decay": float(ema_decay) if use_ema else None
            }
            with open(save_dir / f"pretrain_tianchi_best.json", "w") as f:
                json.dump(best_info, f, indent=4)
            print(f"🌟 更新最佳预训练模型！Epoch {best_epoch} | r2_global={cv:.4f}")

    # Save final model for completeness
    final_eval_model = model_ema.ema if (use_ema and bool(config.get("eval_with_ema", use_ema))) else model
    torch.save(final_eval_model.state_dict(), save_dir / f"pretrain_tianchi_final.pt")

    del train_loader, valid_loader, model, optimizer, scheduler, scaler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return train_losses, val_losses, cv_scores, LR_records, best_model_path

# 🔹 Main training
def train_with_groupkfold(df_train, save_dir, config, device):
    
    df = df_train.copy()
    np.random.seed(42)

    # 随机打乱 group
    unique_groups = np.random.permutation(df["Sampling_Date"].unique())
    group_map = {g: i for i, g in enumerate(unique_groups)}
    df["GroupID"] = df["Sampling_Date"].map(group_map)


    # 各折训练
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
    "epochs"                         : 300,
    "freeze_ratio"                   : 0.6,
    "num_workers"                    : 4,
    "prefetch_factor"                : 3,
    "batch_size"                     : 20,
    
    "lr"                             : 1e-4,
    "n_splits"                       : 5,
    "img_size"                       : 768,
    "backbone_name"                  : "convnext_tiny",
    "cv_stability_stop_threshold"    : 0.02,
    "save_interval"                  : 30,
    "min_epoch"                      : 100,
    # EMA
    "use_ema"                        : True,
    "ema_decay"                      : 0.999,
    "eval_with_ema"                  : True,
    "is_pretrain_tianchi"            : False,
    "is_load_pretrain_tianchi"       : False,
    "pretrain_tianchi_epochs"        : 60,
    "tianchi_csv_path"               : "/home/simon/simondisk2/simon/CSIRO/material/Tianchi/biomass_train_data.csv",
    "tianchi_image_dir"              : "/home/simon/simondisk2/simon/CSIRO/material/Tianchi/GrassClover Dataset02_datasets/train/images",
    "tianchi_n_splits"               : 5,
    "tianchi_pretrain_fold"          : 0,


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
    ],

    # 🔹 Auxiliary tasks (enabled for CSIRO training; pretraining keeps it off)
    "aux_enabled"                    : False,
    # defaults; will be filled from df_train by setup_aux_config_from_df
    "aux_class_cols"                 : ["State", "Species"],
    "aux_reg_cols"                   : ["Pre_GSHH_NDVI", "Height_Ave_cm"],
    "aux_cls_num_classes"            : {},
    "aux_cls_label_maps"             : {},
    "aux_reg_scales"                 : {},
    "aux_loss_weights"               : {"main": 1.0, "cls": 0.2, "reg": 0.2},
}


isTRAIN     = True
isPREDICT   = True
DEBUG       = False




# 🧩 Training Entry Point
if (
    __name__ == "__main__"
    and isTRAIN
    and socket.gethostname() in ["user-PowerEdge-XE9680", "hao-2","simon-MS-7D94"]
):
    # 🔧 Initialize environment
    seed_everything(seed=42)
    torch.multiprocessing.freeze_support()       
    torch.backends.cudnn.benchmark = True        
    idx, device = select_free_gpu()



    # Create output directories & save config
    time_str = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    model_root = DIRS["model"]
    history_DIR = Path(model_root, time_str)

    os.makedirs(model_root, exist_ok=True)
    os.makedirs(history_DIR, exist_ok=True)

    # DEBUG mode (quick test)
    if DEBUG:
        config["epochs"] = 1
        print(f"⚠️ DEBUG mode enabled — running only {config['epochs']} epochs\n")


    # Save config file
    config["time_str"] = time_str
    config["host"] = socket.gethostname()
    config_path = history_DIR / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    print(f"✅ Config saved to: {config_path}")


    # Copy source script
    try:
        source_path = Path(__file__).resolve()
        target_path = history_DIR / source_path.name
        shutil.copy2(source_path, target_path)
        print(f"✅ Source code {source_path.name} copied to: {target_path}")
    except Exception as e:
        print(f"⚠️ Failed to copy source code: {e}")


    print("\n🚀🚀🚀 Starting training process... 🚀🚀🚀\n")


    # Optional: Tianchi pretraining (single fold)
    pretrain_ckpt_path = None
    if config.get("is_pretrain_tianchi", False):
        print("🔧 Pretrain on Tianchi enabled.")
        df_tianchi = load_and_prepare_tianchi_df(config)
        if DEBUG:
            df_tianchi = df_tianchi.iloc[:100].copy()
            print(f"⚠️ DEBUG mode — Using first {len(df_tianchi)} Tianchi samples")
        pretrain_dir = history_DIR / "pretrain_tianchi"
        os.makedirs(pretrain_dir, exist_ok=True)
        _, _, _, _, best_path = train_one_fold_tianchi(
            fold=int(config.get("tianchi_pretrain_fold", 0)),
            df=df_tianchi,
            save_dir=pretrain_dir,
            config=config,
            device=device
        )
        if best_path is not None:
            pretrain_ckpt_path = str(best_path)
            config["pretrain_tianchi_ckpt_path"] = pretrain_ckpt_path
            # persist into config.json
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            print(f"✅ Pretrain checkpoint saved at: {pretrain_ckpt_path}")
        else:
            print("⚠️ No pretrain best checkpoint produced.")


    # Load CSIRO training data
    df_train = load_and_prepare_train_df()
    if DEBUG:
        df_train = df_train.iloc[:7].copy()
        print(f"⚠️ DEBUG mode enabled — Using first {len(df_train)} training data ")
    
    # Compute and persist loss scales for targets
    config["loss_scales"] = compute_loss_scales_from_df(df_train, config["target_cols"])
    # Prepare auxiliary tasks (enabled only for CSIRO training)
    setup_aux_config_from_df(df_train, config)
    # update config.json with loss_scales
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        print("✅ Updated config with loss_scales and aux settings.")
    except Exception as e:
        print(f"⚠️ Failed to update config with loss_scales/aux: {e}")

    # Start training
    # If requested, load pretrain before fold training (handled in train_one_fold)
    train_with_groupkfold(
        df_train             = df_train,
        save_dir             = history_DIR,
        config               = config,
        device               = device
    )

    print(f"\n✅ Training completed! Results saved in: {history_DIR}")
    print("✅"*65)



# Infer
# Load and prepare the test DataFrame
def load_and_prepare_test_df():
    df_file_path = Path(DIRS["dir"]) / "test.csv"
    df = pd.read_csv(df_file_path)

    # Extract unique ID (e.g., "ID1011485656__Dry_Green_g" → "ID1011485656")
    df["ID"] = df["sample_id"].str.split("__").str[0]
    df = move_column_first(df, "ID")
    df["target"] = 0  # 占位列

    # Pivot target columns (long → wide format)
    df_targets = (
        df.pivot_table(index="ID", columns="target_name", values="target", aggfunc="first")
        .reset_index()
    )
    df_targets.columns.name = None

    # Extract meta information (one row per ID)
    df_meta = df[["ID", "image_path"]].drop_duplicates(subset="ID")

    # Merge meta information and target table
    df_test = pd.merge(df_meta, df_targets, on="ID", how="left")
    show_df_info(df_test, "df_test")

    print(f"✅ Test set loaded: {df_test.shape}")
    return df_test

# Load model weights for inference
def load_model_for_inference(model_path, device, config):
    # Initialize model (without pretrained weights)
    model = MyDualStreamModel(config["backbone_name"], pretrained=False, config=config)
    model = model.to(device).to(memory_format=torch.channels_last)

    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location=device)

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

# Single-model prediction
def predict_with_model(model, dataloader, device):
    preds_list = []
    model.eval()

    with torch.no_grad():
        for step, (img_left, img_right, targets) in enumerate(dataloader):
            img_left   = img_left.to(device, non_blocking=True)
            img_right  = img_right.to(device, non_blocking=True)
            targets    = targets.to(device, non_blocking=True)

            preds = model(img_left, img_right)
            
            preds_list.append(preds.cpu().numpy())
    return np.concatenate(preds_list, axis=0)

# Multi-fold ensemble prediction
def predict_ensemble(df_test, transform, model_dir, device, config):
    # Collect all trained model files
    model_paths = sorted(Path(model_dir).glob("model_best*.pt"))
    assert len(model_paths) > 0, f"❌ No model files found in: {model_dir}"

    print(f"✅ Detected {len(model_paths)} 个模型:")
    for p in model_paths:
        print("   -", p.name)

    # 测试集
    test_dataset = DualStreamDataset(
        df_test,
        DIRS["dir"],
        config,
        transform=transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # Perform inference for each fold model
    fold_preds = []
    for fold, model_path in enumerate(model_paths):
        print(f"Fold {fold+1}/{len(model_paths)} 推理: {model_path.name}")
        model = load_model_for_inference(model_path, device, config)
        fold_pred = predict_with_model(model, test_loader, device)
        fold_preds.append(fold_pred)

    preds_mean = np.mean(fold_preds, axis=0)
    df_pred = pd.DataFrame(preds_mean, columns=config["target_cols"])
    
    # Average predictions across folds
    df_pred["ID"] = df_test["ID"]
    df_pred = df_pred[["ID"] + config["target_cols"]]
    show_df_info(df_pred, "df_pred")

    return df_pred

# Generate Kaggle submission file
def generate_submission(df_pred_final):
    print("\n\n\nGenerating Kaggle submission file ------------------------------------")

    print("📦 [Step 1] Original prediction DataFrame info:")
    print(df_pred_final.head(5))
    print(f"🔹 Data shape: {df_pred_final.shape}")

    # ✅ 1️⃣ Fix column order (must match sample_submission.csv)
    ordered_target_cols = [
        "Dry_Green_g",  # 1️⃣
        "Dry_Clover_g", # 2️⃣
        "Dry_Dead_g",   # 3️⃣
        "GDM_g",        # 4️⃣
        "Dry_Total_g"   # 5️⃣
    ]
    print(f"\n📋 [Step 2] Target column order: {ordered_target_cols}")

    # ✅ 2️⃣ Melt the DataFrame from wide to long format
    df_submit = (
        df_pred_final.melt(
            id_vars="ID",
            value_vars=ordered_target_cols,
            var_name="target_name",
            value_name="target"
        )
    )
    print("\n📊 [Step 3] Preview of DataFrame after melt:")
    print(df_submit.head(10))
    print(f"🔹 Shape after melt: {df_submit.shape}")

    # ✅ 3️⃣ Create sample_id column
    df_submit["sample_id"] = df_submit["ID"] + "__" + df_submit["target_name"]

    print("\n🔧 [Step 4] Preview after adding sample_id column:")
    print(df_submit[["ID", "target_name", "sample_id"]].head(10))


    # ✅ 4️⃣ Set final column order and sort
    df_submit = df_submit[["sample_id", "target"]]
    df_submit = df_submit.sort_values("sample_id").reset_index(drop=True)
    print("\n📋 [Step 5] Preview after reordering & sorting by sample_id:")
    print(df_submit.head(10))
    print(f"🔹 Final output shape: {df_submit.shape}")


    # ✅ 5️⃣ Save to CSV
    df_submit.to_csv("submission.csv", index=False)
    print("\n✅ Submission file generated: submission.csv")

# TTA (Test-Time Augmentation) Prediction Aggregation
def run_tta_prediction(df_test, model_dir, device, config):
    # Generate all TTA transforms
    tta_transforms = get_tta_transforms(config["img_size"])

    # Print available TTA modes
    tta_names = list(tta_transforms.keys())
    print(f"\n✅ Detected {len(tta_names)} TTA modes: {tta_names}\n")


    
    all_preds = []

    # Run prediction for each TTA mode
    for name, transform in tta_transforms.items():
        print(f"\n🚀 TTA mode: {name}")
        df_pred = predict_ensemble(df_test, transform, model_dir, device, config)
        all_preds.append(df_pred[config["target_cols"]].values)

        # ✅ Show intermediate results
        print(f"Preview of predictions for TTA mode [{name}]:")
        print(df_pred.head())

        print(f"Number of collected TTA results: {len(all_preds)}")
        print(f"Current cumulative shape: {np.array(all_preds).shape}")
        print("\n\n")


    # 4️⃣ Aggregate TTA results and compute mean predictions
    print("\nAggregating all TTA predictions:")
    print(f"Total {len(all_preds)} sets of predictions collected.")
    for i, arr in enumerate(all_preds):
        print(f"  └─ Prediction set {i+1}: \n{arr}")

    mean_preds = np.mean(all_preds, axis=0)
    print(f"Mean computation completed. Shape: {mean_preds.shape}, values:")
    print(mean_preds)


    # Build final prediction DataFrame
    df_pred_final = df_pred.copy()
    df_pred_final[config["target_cols"]] = mean_preds

    print("\n✅ TTA aggregation completed. Preview of final predictions (df_pred_final):")
    print(df_pred_final.head())


    return df_pred_final


if __name__ == "__main__" and isPREDICT:

    print("\n🧠 Starting prediction pipeline...")
    idx, device = select_free_gpu()
    print(idx,device)
    
    # Select test dataset based on environment
    if socket.gethostname() in ["user-PowerEdge-XE9680", "hao-2","simon-MS-7D94"]:
        if DEBUG:
            df_test = load_and_prepare_train_df().iloc[:7]
        else:
            df_test = load_and_prepare_train_df()
    else:
        df_test = load_and_prepare_test_df()

    print(f"\nCurrent dataset shape: {df_test.shape}\n")


    # Automatically select model directory
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

    # Run TTA (Test-Time Augmentation) prediction
    df_pred_final = run_tta_prediction(df_test, model_dir, device, config)

    # Save results to Excel and CSV
    if socket.gethostname() in ["user-PowerEdge-XE9680", "hao-2","simon-MS-7D94"]:
        out_xlsx = Path(model_dir) / "df_pred_final.xlsx"
        df_pred_final.to_excel(out_xlsx, index=False)
        print(f"✅ Prediction results saved to: {out_xlsx}")

    generate_submission(df_pred_final)
    print("🎯 Prediction pipeline completed.")
