import torch
import torch
import torch.nn as nn
import numpy as np


# 模拟数据：3 个样本 × 5 个目标
# all_preds = torch.tensor([[0.1396, 0.0394, 0.2231, 0.1790, 0.4021],
#                           [0.1501, -0.0556, 0.3123, 0.0945, 0.4067],
#                           [0.1046, -0.0863, 0.2008, 0.0183, 0.2191]], device='cuda:0', dtype=torch.float16)

# all_targets = torch.tensor([[16.2751, 0.0000, 31.9984, 16.2750, 48.2735],
#                             [24.2376, 0.0000, 30.9703, 24.2376, 55.2079],
#                             [32.1910, 23.0755, 2.6135, 55.2665, 57.8800]], device='cuda:0', dtype=torch.float32)


all_preds = torch.tensor([
    [3, 3, 3, 3, 3],   # 样本 1
    [5, 5, 5, 5, 5],   # 样本 2
    [7, 7, 7, 7, 7]    # 样本 3
], device='cuda:0', dtype=torch.float16)
all_targets = torch.tensor([
    [1, 1, 1, 1, 1],   # 第1行（样本1）
    [3, 3, 3, 3, 3],   # 第2行（样本2）
    [5, 5, 5, 5, 5]    # 第3行（样本3）
], device='cuda:0', dtype=torch.float16)




all_preds = all_preds * 2
all_targets = all_targets * 2




# 打印查看这两个张量
print("all_preds:", all_preds)
print("all_targets:", all_targets)



def compute_loss(preds, targets):
    """
    与 Kaggle 官方 Global Weighted R² 对齐的损失：
    loss = ss_res / ss_tot
    （越小越好，与官方评分方向一致）
    """
    # ============================
    # 1️⃣ 打印输入张量的基本信息
    # ============================
    print("【变形前（原始输入）】")
    print(f"预测值 preds:\n{preds}\n形状: {preds.shape}")
    print(f"真实值 targets:\n{targets}\n形状: {targets.shape}")

    # ============================
    # 2️⃣ 调整形状为二维 [B, 5]
    # ============================
    preds = preds.view(-1, 5)
    targets = targets.view(-1, 5)

    print("\n【变形后（统一为二维张量）】")
    print(f"预测值 preds:\n{preds}\n形状: {preds.shape}")
    print(f"真实值 targets:\n{targets}\n形状: {targets.shape}")

    # ============================
    # 3️⃣ 定义每个目标列的权重
    # ============================
    weights = torch.tensor([0.1, 0.1, 0.1, 0.5, 0.2], device=preds.device)
    print("\n【目标权重】")
    print(f"weights: {weights}")

    # ============================
    # 4️⃣ 展平数据（行优先）
    # ============================
    y_true_flat = targets.view(-1)  # 展平真实值
    y_pred_flat = preds.view(-1)    # 展平预测值

    print("\n【展平后的张量】")
    print(f"展平后的真实值 y_true_flat:\n{y_true_flat}\n形状: {y_true_flat.shape}")
    print(f"展平后的预测值 y_pred_flat:\n{y_pred_flat}\n形状: {y_pred_flat.shape}")

    # ============================
    # 5️⃣ 生成对应的权重张量
    # ============================
    # 对每一列复制相应权重，拼接成和展平数据一样长的一维张量
    w_flat = torch.cat([
        torch.full_like(targets[:, i], weights[i], device=preds.device)
        for i in range(5)
    ])

    print("\n【展开后的权重张量】")
    print(f"w_flat:\n{w_flat}\n形状: {w_flat.shape}")

    # ============================
    # 6️⃣ 计算全局加权均值
    # ============================
    y_mean = torch.sum(w_flat * y_true_flat) / torch.sum(w_flat)
    print("\n【全局加权均值】")
    print(f"y_mean: {y_mean}")

    # ============================
    # 7️⃣ 计算残差平方和与总平方和
    # ============================
    ss_res = torch.sum(w_flat * (y_true_flat - y_pred_flat) ** 2)
    ss_tot = torch.sum(w_flat * (y_true_flat - y_mean) ** 2)

    print("\n【加权残差平方和与总平方和】")
    print(f"ss_res（加权残差平方和）: {ss_res}")
    print(f"ss_tot（加权总平方和）: {ss_tot}")

    # ============================
    # 8️⃣ 计算最终损失
    # ============================
    loss = ss_res / ss_tot
    print("\n【最终加权损失（越小越好）】")
    print(f"Loss = ss_res / ss_tot = {loss}")

    return loss



def compute_cv_score(all_preds, all_targets):
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # 确保 targets 和 preds 是二维数组，如果是 1D，就 reshape 成 [B, 5]
    if targets.ndim == 1:
        targets = targets.reshape(-1, 5)
    if preds.ndim == 1:
        preds = preds.reshape(-1, 5)

    
    print("\n【变形后（统一为二维张量）】")
    print(f"预测值 preds:\n{preds}\n形状: {preds.shape}")
    print(f"真实值 targets:\n{targets}\n形状: {targets.shape}")

    # 五个目标列名与对应权重（与官方相同）
    target_cols = ["Dry_Green_g", "Dry_Clover_g", "Dry_Dead_g", "GDM_g", "Dry_Total_g"]
    weights = np.array([0.1, 0.1, 0.1, 0.1, 0.1])

    # 拼接所有目标
    y_true_flat = np.concatenate([targets[i, :] for i in range(targets.shape[0])])
    y_pred_flat = np.concatenate([preds[i, :] for i in range(preds.shape[0])])

    w_flat = np.concatenate([np.full_like(targets[:, i], weights[i]) for i in range(5)])

    print("\n【展平后的张量】")
    print(f"展平后的真实值 y_true_flat:\n{y_true_flat}\n形状: {y_true_flat.shape}")
    print(f"展平后的预测值 y_pred_flat:\n{y_pred_flat}\n形状: {y_pred_flat.shape}")
    print("\n【展开后的权重张量】")
    print(f"w_flat:\n{w_flat}\n形状: {w_flat.shape}")

    # 全局加权均值
    y_mean = np.sum(w_flat * y_true_flat) / np.sum(w_flat)
    print("\n【全局加权均值】")
    print(f"y_mean: {y_mean}")

    # 计算加权残差平方和与总平方和
    ss_res = np.sum(w_flat * (y_true_flat - y_pred_flat) ** 2)
    ss_tot = np.sum(w_flat * (y_true_flat - y_mean) ** 2)

    print("\n【加权残差平方和与总平方和】")
    print(f"ss_res（加权残差平方和）: {ss_res}")
    print(f"ss_tot（加权总平方和）: {ss_tot}")

    # Kaggle 官方全局加权 R²
    r2_global = ss_res / ss_tot
    return r2_global


# 2. 使用 SmoothL1 损失函数的加权损失（compute_loss2）
def compute_loss2(preds, targets):
    """
    计算加权的 Smooth L1 损失
    """
    l1 = nn.SmoothL1Loss(reduction="none")
    # 权重定义（与官方相同）
    weights = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1], device=preds.device).view(1, 5)

    # 计算每个目标的损失
    per_target_loss = l1(preds, targets)
    weighted_loss = (per_target_loss * weights).mean()
    return weighted_loss







# 计算并打印损失
print("Using compute_loss:---------------------------------------------------------------")
loss = compute_loss(all_preds, all_targets)
print(f"Computed loss: {loss.item()}")



# 如果你希望运行 compute_cv_loss，你需要传入 NumPy 张量
print("\nUsing compute_cv_loss (R² style):---------------------------------------------------------------")
# 将 PyTorch 张量转换为 NumPy 数组传入
all_preds_np = all_preds.cpu().numpy()
all_targets_np = all_targets.cpu().numpy()
loss3 = compute_cv_score(all_preds_np, all_targets_np)
print(f"Computed loss: {loss3}")


print("\nUsing compute_loss2 (SmoothL1 weighted):---------------------------------------------------------------")
loss2 = compute_loss2(all_preds, all_targets)
print(f"Computed loss: {loss2.item()}")