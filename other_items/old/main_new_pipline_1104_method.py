# 📦 导入库
import os, json, time, socket, gc, psutil
import numpy as np, pandas as pd, torch, timm, cv2, h5py
import torch.nn as nn
from pathlib import Path
from datetime import datetime, timedelta
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import r2_score
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image


# ⚙️ 全局配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化
if socket.gethostname() == 'hao-2':
    dir = Path('D:/DATA_hao/Kaggle_/csiro-biomass/')
    DIRS = {
    "dir":        dir,                                       
    "train":     Path(dir, "train"),                              
    "test":     Path(dir, "test"),                              
    "model":     Path(dir,"DualStream_multihead"),              
    "data":     Path(dir),   
    }
    
    # # 打印时一行一个地址
    # print("✅ 路径：\n")
    # for key, path in DIRS.items():
    #     print(f"{key:<12} : {path}")
else:
    dir = Path('/kaggle/input/csiro-biomass')
    DIRS = {
    "dir":        dir,                                       
    "train":     Path(dir, "train"),                              
    "test":     Path(dir, "test"),                              
    "model":     Path('/kaggle/input', "dualstream-multihead-model"),              
    "data":     Path("/kaggle/working/"),   
    }

    # # 打印时一行一个地址
    # print("✅ 路径：\n")
    # for key, path in DIRS.items():
    #     print(f"{key:<12} : {path}")

# 小函数
def show_df_info(df, name: str):
    """
    打印单个 DataFrame 的形状与列名信息。
    参数:
        df   : pandas.DataFrame
        name : 显示名称（字符串）
    """
    print(f"📊 {name:<16} shape: {str(df.shape):<16}  列名: {df.columns.tolist()}")

def move_column_first(df, col_name):
    """
    将 DataFrame 中指定列移动到最前面。
    参数:
        df (pd.DataFrame): 原始数据框
        col_name (str): 要移动到最前面的列名
    返回:
        pd.DataFrame: 调整后的新 DataFrame
    """
    if col_name not in df.columns:
        raise ValueError(f"列 '{col_name}' 不存在于 DataFrame 中。")

    cols = [col_name] + [c for c in df.columns if c != col_name]
    return df[cols]

# 🧮 后处理函数（恢复 5 个目标）
def recover_all_targets(df_pred_3):
    df = df_pred_3.copy()
    df["Dry_Clover_g"] = np.maximum(0, df["GDM_g"] - df["Dry_Green_g"])
    df["Dry_Dead_g"] = np.maximum(0, df["Dry_Total_g"] - df["GDM_g"])
    return df[["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]]


# 🧠 MyDualStreamModel：双流 + 多头回归 + 内部训练逻辑
class MyDualStreamModel(nn.Module):
    def __init__(self, 
                backbone_name="convnext_tiny", 
                pretrained=True, 
                config = None):
        """
        参数:
        - backbone_name: timm 模型名称 (如 convnext_tiny, resnet50)
        - pretrained: 是否加载 ImageNet 权重
        - freeze_ratio: 冻结比例（0~1）
        - weights_dict: 各目标权重 (dict), 用于 WeightedSmoothL1Loss
        """
        super().__init__()

        # 1️⃣ Backbone
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        in_dim = self.backbone.num_features

        # 2️⃣ 冻结部分参数
        params = list(self.backbone.parameters())
        freeze_until = int(len(params) * config["freeze_ratio"])
        for i, p in enumerate(params):
            p.requires_grad = i >= freeze_until  # 前部分冻结，后部分可学习

        # 3️⃣ 双流融合
        self.fusion_dim = in_dim * 2

        # 4️⃣ 三个输出 Head
        def make_head():
            return nn.Sequential(
                nn.Linear(self.fusion_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )

        # 模型仅预测三个“独立变量”
        self.head_green  = make_head()   # Dry_Green_g
        self.head_clover = make_head()   # Dry_Clover_g
        self.head_dead   = make_head()   # Dry_Dead_g

        # 5️⃣ 权重
        self.weights = config["weights"]

    # ------------------------------------------------------------
    # 🔁 Forward
    # ------------------------------------------------------------
    def forward(self, img_left, img_right):
        # 提取特征
        feat_left  = self.backbone(img_left)
        feat_right = self.backbone(img_right)
        fused = torch.cat([feat_left, feat_right], dim=1)

        # 三头预测（标准化空间）
        zG = self.head_green(fused)
        zC = self.head_clover(fused)
        zD = self.head_dead(fused)
        preds_3 = torch.cat([zG, zC, zD], dim=1)  # [B, 3]


        # 结构化推导出 GDM 和 Total
        G, C, D = preds_3[:, 0:1], preds_3[:, 1:2], preds_3[:, 2:3]
        GDM = G + C
        Total = G + C + D

        preds_full = torch.cat([G, C, D, GDM, Total], dim=1)  # [B, 5]
        return preds_full

    # ------------------------------------------------------------
    # 🧮 损失计算（内部调用）
    # ------------------------------------------------------------
    def compute_loss(self, preds, targets):
        l1 = nn.SmoothL1Loss(reduction="none")
        w = torch.tensor([
            self.weights["Dry_Green_g"],
            self.weights["Dry_Clover_g"],
            self.weights["Dry_Dead_g"],
            self.weights["GDM_g"],
            self.weights["Dry_Total_g"]
        ], device=preds.device).view(1, 5)

        per_target_loss = l1(preds, targets)
        weighted_loss = (per_target_loss * w).mean()
        return weighted_loss

# 数据集加载定义
class DualStreamDataset(Dataset):
    def __init__(self, df, image_dir, config, transform=None):
        """
        df: DataFrame，包含 image_path 列
        image_dir: 图像目录
        target_cols: 如果是训练集，指定目标列
        transform: Albumentations 变换
        """
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.target_cols = config["target_cols"]
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = Path(self.image_dir, str(row["image_path"]))
        
        # ====== 1️⃣ 安全加载 ======
        if not img_path.exists():
            print(f"⚠️ 图片不存在: {img_path}")
            image = np.zeros((1000, 2000, 3), dtype=np.uint8)
        else:
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"⚠️ 无法读取图片: {img_path} ({e})")
                image = np.zeros((1000, 2000, 3), dtype=np.uint8)

        # ====== 2️⃣ 确保转换为 NumPy 数组 ======
        image = np.array(image)  # 转换为 NumPy 数组
        h, w, _ = image.shape
        mid = w // 2
        
        # 拆分成左右两个 patch
        img_left = image[:, :mid]
        img_right = image[:, mid:]

        # ====== 4️⃣ 应用 Albumentations 变换 ======
        if self.transform:
            img_left = self.transform(image=img_left)["image"]
            img_right = self.transform(image=img_right)["image"]

        # ====== 5️⃣ 返回结果 ======
        if self.target_cols is not None:
            targets = torch.tensor(row[self.target_cols].astype(float).values, dtype=torch.float32)
            return img_left, img_right, targets
        else:
            return img_left, img_right

# Albumentations 变换   训练集、验证集、测试TTA
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
        "rot90": A.Compose([
            A.Resize(size, size),
            A.RandomRotate90(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        "brightness": A.Compose([
            A.Resize(size, size),
            A.RandomBrightnessContrast(brightness_limit=0.1,
                                       contrast_limit=0.1, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        "gamma": A.Compose([
            A.Resize(size, size),
            A.RandomGamma(gamma_limit=(90, 110), p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
    }

# ✅ 计算 Weighted R² 分数（完全与 Kaggle Metric 对齐）
def compute_cv_score(all_preds, all_targets):
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # 五个目标列名与对应权重（与官方相同）
    target_cols = ["Dry_Green_g", "Dry_Clover_g", "Dry_Dead_g", "GDM_g", "Dry_Total_g"]
    weights = np.array([0.1, 0.1, 0.1, 0.2, 0.5])

    # 拼接所有目标
    y_true_flat = np.concatenate([targets[:, i] for i in range(5)])
    y_pred_flat = np.concatenate([preds[:, i] for i in range(5)])
    w_flat = np.concatenate([np.full_like(targets[:, i], weights[i]) for i in range(5)])

    # 全局加权均值
    y_mean = np.sum(w_flat * y_true_flat) / np.sum(w_flat)

    # 计算加权残差平方和与总平方和
    ss_res = np.sum(w_flat * (y_true_flat - y_pred_flat) ** 2)
    ss_tot = np.sum(w_flat * (y_true_flat - y_mean) ** 2)

    # Kaggle 官方全局加权 R²
    r2_global = 1 - ss_res / ss_tot
    return r2_global


# 🔹 单轮训练
def train_one_epoch(model, dataloader, optimizer, device, scaler):
    model.train()
    running_loss = []

    start_epoch = time.time()
    prev_end = start_epoch  # ⏱️ 上一 batch 结束时间，用于统计 data loading time

    for step, (img_left, img_right, targets) in enumerate(dataloader):
        t_load = time.time()  # dataloader 取到 batch 后的时间
        data_load_time = t_load - prev_end

        # ====== 数据拷贝到 GPU ======
        t0 = time.time()
        img_left, img_right, targets = (
            img_left.to(device, non_blocking=True),
            img_right.to(device, non_blocking=True),
            targets.to(device, non_blocking=True),
        )
        t1 = time.time()

        # ====== 前向 + 反向 ======
        optimizer.zero_grad(set_to_none=True)  # ✅ 更高效清空梯度
        # ✅ AMP混合精度上下文
        with autocast():
            preds = model(img_left, img_right)
            loss = model.compute_loss(preds, targets)
        t2 = time.time()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        t3 = time.time()

        running_loss.append(loss.item())
        prev_end = t3  # 下次计算 data_load_time 用

        # 每 N 步打印耗时细分
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

# 🔹 单轮验证 + 本地CV
def validate_one_epoch(model, dataloader, device):
    model.eval()
    val_losses, all_preds, all_targets = [], [], []

    start_epoch = time.time()
    prev_end = start_epoch  # ⏱️ 上一 batch 结束时间（用于统计 data loading time）

    with torch.no_grad():
        for step, (img_left, img_right, targets) in enumerate(dataloader):
            t_load = time.time()  # dataloader 提供当前 batch 的时间
            data_load_time = t_load - prev_end

            # ====== 数据拷贝到 GPU ======
            t0 = time.time()
            img_left, img_right, targets = (
                img_left.to(device, non_blocking=True),
                img_right.to(device, non_blocking=True),
                targets.to(device, non_blocking=True),
            )
            t1 = time.time()

            # ====== 前向推理 + 计算损失 ======
            preds = model(img_left, img_right)
            val_loss = model.compute_loss(preds, targets).item()
            t2 = time.time()

            val_losses.append(val_loss)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

            prev_end = t2  # 用于计算下一个 batch 的 data_load_time

            # 每 N 步打印耗时细分
            
            # if step  == 0  or step  == 1:
            #     print(
            #         f"[VAL] Step {step:4d} | "
            #         f"data load: {data_load_time*1000:.1f} ms | "
            #         f"to(device): {(t1 - t0)*1000:.1f} ms | "
            #         f"forward+loss: {(t2 - t1)*1000:.1f} ms | "
            #         f"total: {(t2 - t_load)*1000:.1f} ms"
            #     )

    end_epoch = time.time()
    epoch_time = end_epoch - start_epoch
    avg_val_loss = float(np.mean(val_losses))
    r2_global = compute_cv_score(all_preds, all_targets)

    # print(
    #     f"[VAL] Epoch total time: {epoch_time:.2f}s | "
    #     f"{len(dataloader)} batches | {epoch_time / len(dataloader):.3f}s/batch"
    # )

    return avg_val_loss, r2_global

# 🔹 主函数：KFold 训练
def train_with_groupkfold(  
    df_train,
    save_dir,
    get_train_transforms,
    get_valid_transforms,
    config,
    device=None,
):
    """使用 GroupKFold 进行交叉验证训练"""

    df = df_train.copy()



    # 固定随机种子
    np.random.seed(42)
    # 打乱分组顺序（只打乱 Sampling_Date 的顺序，不破坏组内结构）
    unique_groups = df["Sampling_Date"].unique()
    shuffled_groups = np.random.permutation(unique_groups)
    # 重建 group 序列（映射打乱顺序）
    group_mapping = {g: i for i, g in enumerate(shuffled_groups)}
    df["GroupID"] = df["Sampling_Date"].map(group_mapping)
    # 重新分组
    # 创建分组 K 折对象（按采样日期分组）
    gkf = GroupKFold(n_splits=config["n_splits"])
    groups = df["GroupID"]
    


    # 保存各折的指标
    fold_train_losses, fold_val_losses, fold_cv_scores, fold_LR_records = [], [], [], []
    epoch_times = []  # ⏱️ 保存最近 10 个 epoch 耗时，用于计算 ETA


    # 🔁 逐折训练
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, groups=groups)):

        # 划分当前折的训练集与验证集
        train_df = df.iloc[train_idx].reset_index(drop=True)
        valid_df = df.iloc[val_idx].reset_index(drop=True)

        # 构建 Dataset 与 DataLoader
        train_dataset = DualStreamDataset(train_df, DIRS["dir"], config, transform=get_train_transforms(config["img_size"]))
        valid_dataset = DualStreamDataset(valid_df, DIRS["dir"], config, transform=get_valid_transforms(config["img_size"]))


        # 自动获取 CPU 核心数（此处手动设定为 4）
        num_workers = 4
        prefetch_factor = 3

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


        # ✅ 模型初始化：channels_last 内存布局 + AMP 兼容
        model = MyDualStreamModel(config["backbone_name"], pretrained=True, config=config)
        model = model.to(device).to(memory_format=torch.channels_last)

        # ✅ 优化器：AdamW（推荐首选）
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config["lr"]          # 主学习率
            # weight_decay=1e-2         # 控制参数规模的 L2 正则（建议 1e-2 ~ 5e-3）
        )

        # ✅ 调度器：余弦退火（一个完整周期）
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["epochs"],         # 学习率退火完整周期
            eta_min=config["lr"] / 100       # 最低学习率比例
        )

        # ✅ 混合精度缩放器（提升速度与显存效率）
        scaler = torch.cuda.amp.GradScaler()

        # 记录当前折指标
        train_losses, val_losses, cv_scores, LR_records = [], [], [], []


        # 🔁 逐 epoch 训练
        for epoch in range(config["epochs"]):
            epoch_start = time.time()

            # --- 单轮训练与验证 ---
            avg_train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler)
            avg_val_loss, r2_global = validate_one_epoch(model, valid_loader, device)
            scheduler.step()

            # --- 记录指标 ---
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            cv_scores.append(r2_global)
            LR_records.append(scheduler.get_last_lr()[0])

            # --- Early Stopping 条件检测 ---
            if epoch >= 20:
                window_scores = cv_scores[-20:]  # 最近20个epoch
                diff = max(window_scores) - min(window_scores)
                if diff < config["cv_stability_stop_threshold"]:
                    print(
                        f"\n🛑 Early stopping triggered on Fold {fold} at Epoch {epoch+1} "
                        f"(CV fluctuation {diff:.4f} < threshold {config['cv_stability_stop_threshold']})"
                    )
                    break

            # --- 定期保存模型 ---
            if (epoch + 1) % config["save_interval"] == 0:
                save_path = save_dir / f"model_weights_fold{fold}_epoch{epoch+1}.pt"
                torch.save(model.state_dict(), save_path)

            # --- 时间统计与 ETA ---
            epoch_time = time.time() - epoch_start
            if epoch > 0:
                epoch_times.append(epoch_time)
                if len(epoch_times) > 10:
                    epoch_times.pop(0)  # 只保留最近 10 个

            now_str = datetime.now().strftime("%H:%M:%S")
            progress = (epoch + 1) + fold * config["epochs"]
            all_progress = config["epochs"] * config["n_splits"]
            remaining_epochs = all_progress - progress

            avg_epoch_time = np.mean(epoch_times) if epoch_times else epoch_time
            eta_seconds = avg_epoch_time * remaining_epochs if epoch_times else float('nan')

            # ====== 预计完成时间 ======
            if not np.isnan(eta_seconds):
                eta_time = datetime.now() + timedelta(seconds=eta_seconds)
                eta_time = eta_time.replace(microsecond=0)
                days_diff = (eta_time.date() - datetime.now().date()).days
                eta_str = f"T+{days_diff} " + eta_time.strftime("%H:%M:%S") if days_diff > 0 else eta_time.strftime("%H:%M:%S")
            else:
                eta_str = "--:--:--"

            # --- 日志输出 ---
            print(
                f"[{now_str}]🧩[{progress/all_progress*100:.2f}%] "
                f"Fold {fold}/{config['n_splits']} | "
                f"Epoch {epoch+1}/{config['epochs']} | "
                f"Train={avg_train_loss:.4f} | "
                f"Val={avg_val_loss:.4f} | "
                f"CV={r2_global:.4f} | "
                f"lr={scheduler.get_last_lr()[0]:.6f} | "
                f"{avg_epoch_time:.2f}s/it | "
                f"ETA≈{eta_str}\n",
                end="\r",
                flush=True
            )







        # 📦 当前 Fold 训练完成
        torch.save(model.state_dict(), save_dir / f"model_weights_fold{fold}_epoch{epoch+1}_final.pt")

        fold_train_losses.append(train_losses)
        fold_val_losses.append(val_losses)
        fold_cv_scores.append(cv_scores)
        fold_LR_records.append(LR_records)

        # 🧹 Fold 结束后清理（更彻底）
        try:
            del train_loader, valid_loader, train_dataset, valid_dataset
        except Exception:
            pass
        try:
            del optimizer, scheduler, scaler
        except Exception:
            pass
        del model

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        time.sleep(2)  # 给 dataloader worker 释放时间


    # 📊 保存整体训练日志
    max_epochs = max(len(x) for x in fold_train_losses)
    df_out = pd.DataFrame({"Epoch": range(1, max_epochs + 1)})

    for i, (train_list, val_list, cv_list, lr_list) in enumerate(
        zip(fold_train_losses, fold_val_losses, fold_cv_scores, fold_LR_records),
        start=1
    ):
        df_out[f"Train_Loss_Fold{i}"] = train_list + [None] * (max_epochs - len(train_list))
        df_out[f"Val_Loss_Fold{i}"]   = val_list   + [None] * (max_epochs - len(val_list))
        df_out[f"CV_Fold{i}"]         = cv_list    + [None] * (max_epochs - len(cv_list))
        df_out[f"LR_Fold{i}"]         = lr_list    + [None] * (max_epochs - len(lr_list))

    out_path = Path(save_dir) / "fold_metrics.xlsx"
    df_out.to_excel(out_path, index=False)
    print(f"\n✅ 训练日志已保存: {out_path}")

# 📘 训练数据读取与预处理
def load_and_prepare_train_df():
    # 1️⃣ 读取原始数据
    df_file_path = Path(DIRS["dir"]) / "train.csv"
    df = pd.read_csv(df_file_path)
    # show_df_info(df, "train.csv")

    # 2️⃣ 提取唯一 ID（例如 "ID1011485656__Dry_Green_g" → "ID1011485656"）
    df["ID"] = df["sample_id"].str.split("__").str[0]

    # 3️⃣ 将 ID 列移动到最前面
    df = move_column_first(df, "ID")
    # show_df_info(df, "df")

    # 4️⃣ 目标值透视（行转列）
    df_targets = (
        df
        .pivot_table(
            index="ID",
            columns="target_name",
            values="target",
            aggfunc="first"
        )
        .reset_index()
    )
    df_targets.columns.name = None  # 去掉多级列名层次
    # show_df_info(df_targets, "df_targets")

    # 5️⃣ 提取元信息（每个 ID 仅保留一行）
    meta_cols = [
        "ID", "image_path", "Sampling_Date", "State",
        "Species", "Pre_GSHH_NDVI", "Height_Ave_cm"
    ]
    df_meta = df[meta_cols].drop_duplicates(subset="ID")
    # show_df_info(df_meta, "df_meta")

    # 6️⃣ 合并元信息与目标数据
    df_train = pd.merge(df_meta, df_targets, on="ID", how="left")
    show_df_info(df_train, "df_train")

    
    return df_train



config = {
    # ⚙️ 基础训练参数
    "epochs"       : 240,
    "freeze_ratio" : 0.0,
    "batch_size"   : 12,
    "lr"           : 1e-4,
    "n_splits"     : 5,
    "save_interval": 20,
    "img_size"     : 500,
    "backbone_name"     : "focalnet_tiny_srf",
    "cv_stability_stop_threshold"     : 0.03,

    # ⚖️ 损失权重（与评分规则对应）
    "weights": {
        "Dry_Green_g" : 0.1,
        "Dry_Clover_g": 0.1,
        "Dry_Dead_g"  : 0.1,
        "GDM_g"       : 0.2,
        "Dry_Total_g" : 0.5
    },

    # 📊 完整目标列（包括计算所得的 GDM、Total）
    "target_cols": [
        "Dry_Green_g",
        "Dry_Clover_g",
        "Dry_Dead_g",
        "GDM_g",
        "Dry_Total_g"
    ]
}


# 训练部分
isTRAIN = True
if __name__ == "__main__" and isTRAIN: 
    torch.multiprocessing.freeze_support()  # ✅ 仅在主进程入口调用一次
    torch.backends.cudnn.benchmark = True  # ✅ 全局启用 cudnn benchmark
    print(f"✅ 使用设备: {device}")


    # 启动训练 🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀


    # 本地机器执行（hao-2）
    if socket.gethostname() == "hao-2":

        # 生成时间戳与结果目录
        time_str = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        print(time_str)

        history_DIR = Path(DIRS["model"], time_str)
        os.makedirs(history_DIR, exist_ok=True)

        # 保存配置文件
        config["time_str"] = time_str
        config_path = history_DIR / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        print(f"✅ 配置文件已保存到: {config_path}")





        # 读取训练数据
        df_train = load_and_prepare_train_df()

        # 🚀 启动 KFold 训练
        train_with_groupkfold(
            df_train             = df_train,
            save_dir             = history_DIR,
            get_train_transforms = get_train_transforms,
            get_valid_transforms = get_valid_transforms,
            config               = config,
            device               = device
        )

        print("\n✅ 全部训练完成！结果保存在：", history_DIR)





# 预测部分
# 📘 数据读取与预处理（测试集）
def load_and_prepare_test_df():
    # 1️⃣ 读取原始数据
    df_file_path = Path(DIRS["dir"]) / "test.csv"
    df = pd.read_csv(df_file_path)
    show_df_info(df, "test.csv")

    # 2️⃣ 提取唯一 ID（例如 "ID1011485656__Dry_Green_g" → "ID1011485656"）
    df["ID"] = df["sample_id"].str.split("__").str[0]

    # 3️⃣ 将 ID 列移动到最前面
    df = move_column_first(df, "ID")

    # 4️⃣ 初始化目标列（test 集无目标值）
    df["target"] = 0
    show_df_info(df, "df")

    # 5️⃣ 目标列透视（行转列结构保持一致）
    df_targets = (
        df
        .pivot_table(
            index="ID",
            columns="target_name",
            values="target",
            aggfunc="first"
        )
        .reset_index()
    )
    df_targets.columns.name = None  # 去掉多级列名层次
    show_df_info(df_targets, "df_targets")

    # 6️⃣ 提取元信息（每个 ID 仅保留一行）
    meta_cols = [
        "ID",
        "image_path",
    ]
    df_meta = df[meta_cols].drop_duplicates(subset="ID")
    show_df_info(df_meta, "df_meta")

    # 7️⃣ 合并元信息与目标数据
    df_test = pd.merge(df_meta, df_targets, on="ID", how="left")
    show_df_info(df_test, "df_test")

    return df_test

# 基于 model  transform  model_dir  预测
def predict_ensemble_df(df_test, transform, model, model_target_cols, model_dir, device, batch_size=32, img_size=768):

    model_dir = model_dir
    print(f"模型目录: {model_dir}")
    assert model_dir.exists(), f"❌ 模型目录不存在: {model_dir}"

    # 🔍 搜索所有 fold 模型
    model_paths = sorted(model_dir.glob("model_weights_fold*_final.pt"))
    if not model_paths:
        raise FileNotFoundError(f"❌ 未找到模型文件: {model_dir}/model_weights_fold*.pt")

    print(f"🔹 检测到 {len(model_paths)} 个模型:")
    for p in model_paths:
        print("   -", p.name)

    # 3️⃣ 构建测试数据集
    test_dataset = DualStreamDataset(
        df_test, 
        DIRS["dir"], 
        config, 
        transform=transform 
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # 存储每个fold的预测
    fold_preds = []

    for fold, model_path in enumerate(model_paths):
        print(f"🚀 加载模型 {fold+1}/{len(model_paths)}: {model_path.name}")

        # 1️⃣ 加载模型结构
        model = model

        # 2️⃣ 加载权重
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        # 3️⃣ 推理
        preds_list = []
        with torch.no_grad():
            for img_left, img_right, _ in test_loader:
                img_left, img_right = img_left.to(device, non_blocking=True), img_right.to(device, non_blocking=True)
                preds = model(img_left, img_right)
                preds_list.append(preds.cpu().numpy())

        fold_pred = np.concatenate(preds_list, axis=0)
        fold_preds.append(fold_pred)

    # 4️⃣ 多模型平均
    preds_mean = np.mean(fold_preds, axis=0)
    df_pred5 = pd.DataFrame(preds_mean, columns=model_target_cols)





    # 追加样本 ID 并调整列顺序
    df_pred5["ID"] = df_test["ID"]
    df_pred5 = df_pred5[["ID"] + model_target_cols]




    # 打印结果预览
    show_df_info(df_pred5, "final df_pred5")

    return df_pred5

# 📤 5️⃣ 生成 Kaggle 提交文件 submission.csv
def generate_Kaggle_file(df_pred_final):

    df = df_pred_final

    # 按指定顺序展开
    ordered_target_cols = [
        "Dry_Clover_g",  # 1️⃣
        "Dry_Dead_g",    # 2️⃣
        "Dry_Green_g",   # 3️⃣
        "Dry_Total_g",   # 4️⃣
        "GDM_g"          # 5️⃣
    ]

    df_submit = (
        df
        .melt(id_vars="ID", value_vars=ordered_target_cols,
            var_name="target_name", value_name="target")
    )

    # 组合成 Kaggle 所需的 sample_id
    df_submit["sample_id"] = df_submit["ID"] + "__" + df_submit["target_name"]

    df_submit = move_column_first(df_submit, "target")
    df_submit = move_column_first(df_submit, "sample_id")

    # 只保留 Kaggle 要的两列
    df_submit = df_submit[["sample_id", "target"]]
    df_submit
    # 按 sample_id 排序（可选）
    # df_submit = df_submit.sort_values("sample_id").reset_index(drop=True)

    # 保存文件
    df_submit.to_csv("submission.csv", index=False)
    print("✅ 已生成提交文件 submission.csv")

# 🧠 模型加载与 TTA 推理
if __name__ == "__main__" and not isTRAIN: 

    # 1️⃣ 加载模型结构
    # ✅ 模型初始化：channels_last 内存布局 + AMP 兼容
    model = MyDualStreamModel(config["backbone_name"], pretrained=False, config=config)
    model = model.to(device).to(memory_format=torch.channels_last)

    # 2️⃣ 设置模型目录（根据运行环境自动切换）
    if socket.gethostname() == "hao-2":
        model_dir = Path(DIRS["model"] , "2025-11-02 23-23-25")
    else:
        model_dir = DIRS["model"]

    # 3️⃣ 执行 TTA（Test-Time Augmentation）推理
    tta_preds = []
    tta_transforms = get_tta_transforms(config["img_size"])

    for name, tform in tta_transforms.items():
        print(f"\n🚀 Running TTA: {name}")

        transform  = tform
        df_pred5   = predict_ensemble_df(
            df_test           = load_and_prepare_test_df(),
            transform         = transform,
            model             = model,
            model_target_cols = config["target_cols"],
            model_dir         = model_dir,
            device            = device,
            img_size          = config["img_size"]
        )
        
        # ✅ 输出阶段性结果
        print(f"\n📄 当前 TTA 模式 [{name}] 的预测结果预览：")
        print(df_pred5.head())

        tta_preds.append(df_pred5[config["target_cols"]].values)

        print(f"\n📦 当前已收集的 TTA 结果数量：{len(tta_preds)}")
        print(f"📊 当前累计结果形状：{np.array(tta_preds).shape}")
        print("-" * 60)
        print("\n\n\n")


    # 4️⃣ 汇总 TTA 结果并计算平均预测
    print("\n📦 聚合全部 TTA 结果：")
    print(f"共有 {len(tta_preds)} 组预测结果。")
    for i, arr in enumerate(tta_preds):
        print(f"  └─ 第 {i+1} 组预测: {arr}")

    mean_preds = np.mean(tta_preds, axis=0)

    print("\n🧮 计算平均值完成：")
    print(mean_preds)
    print(f"\n✅ 聚合完成，mean_preds 形状：{mean_preds.shape}")


    # 5️⃣ 生成最终预测 DataFrame
    df_pred_final = df_pred5.copy()
    df_pred_final[config["target_cols"]] = mean_preds

    print("\n🧾 最终预测 DataFrame 预览：")
    print(df_pred_final.head())
    show_df_info(df_pred_final, "df_pred_final")

    generate_Kaggle_file(df_pred_final)


