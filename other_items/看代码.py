import os
import math
import argparse
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

import cv2
import timm
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2


@dataclass
class Config:
    dropout: float = 0.1
    hidden_ratio: float = 0.35
    
    dino_candidates: Tuple[str, ...] = (
        "vit_base_patch14_dinov2",
        "vit_base_patch14_reg4_dinov2",
        "vit_small_patch14_dinov2",
    )
    small_grid: Tuple[int, int] = (4, 4)
    big_grid: Tuple[int, int] = (2, 2)
    t2t_depth: int = 2
    cross_layers: int = 2
    cross_heads: int = 6
    
    pyramid_dims: Tuple[int, int, int] = (384, 512, 640)
    mobilevit_heads: int = 4
    mobilevit_depth: int = 2
    sra_heads: int = 8
    sra_ratio: int = 2
    mamba_depth: int = 3
    mamba_kernel: int = 5
    aux_head: bool = True
    aux_loss_weight: float = 0.4
    
    base_path: str = "/kaggle/input/csiro-biomass"
    test_csv: str = "/kaggle/input/csiro-biomass/test.csv"
    test_image_dir: str = "/kaggle/input/csiro-biomass/test"
    
    experiment_dir_a: str = "/kaggle/input/csiro/pytorch/default/12"
    ckpt_pattern_fold_x_a: str = "/kaggle/input/csiro/pytorch/default/12/fold_{fold}/checkpoints/best_wr2.pt"
    ckpt_pattern_foldx_a: str = "/kaggle/input/csiro/pytorch/default/12/fold{fold}/checkpoints/best_wr2.pt"
    n_folds_a: int = 5
    
    model_dir_b: str = "/kaggle/input/csiro-mvp-models"
    model_paths_b: List[str] = field(default_factory=lambda: [
        f"/kaggle/input/csiro-mvp-models/model{i}.pth" for i in range(1, 11)
    ])
    weight_a: float = 0.95
    weight_b: float = 0.075
    
    batch_size: int = 1
    num_workers: int = 0
    mixed_precision: bool = True
    use_tta: bool = True
    
    submission_file: str = "submission.csv"
    
    all_target_cols: Tuple[str, ...] = (
        "Dry_Green_g",
        "Dry_Dead_g",
        "Dry_Clover_g",
        "GDM_g",
        "Dry_Total_g",
    )
    
    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @property
    def ckpts_a(self):
        return self.model_paths_b[:5]
    
    @property
    def ckpts_b(self):
        return self.model_paths_b[5:]


CFG = Config()


class FeedForward(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        hid = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class AttentionBlock(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.0, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class MobileViTBlock(nn.Module):
    def __init__(self, dim, heads=4, depth=2, patch=(2, 2), dropout=0.0):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
        )
        self.patch = patch
        self.transformer = nn.ModuleList(
            [AttentionBlock(dim, heads=heads, dropout=dropout, mlp_ratio=2.0) for _ in range(depth)]
        )
        self.fuse = nn.Conv2d(dim * 2, dim, kernel_size=1)

    def forward(self, x: torch.Tensor):
        local_feat = self.local(x)
        B, C, H, W = local_feat.shape
        ph, pw = self.patch
        new_h = math.ceil(H / ph) * ph
        new_w = math.ceil(W / pw) * pw
        if new_h != H or new_w != W:
            local_feat = F.interpolate(local_feat, size=(new_h, new_w), mode="bilinear", align_corners=False)
            H, W = new_h, new_w

        tokens = local_feat.unfold(2, ph, ph).unfold(3, pw, pw)
        tokens = tokens.contiguous().view(B, C, -1, ph, pw)
        tokens = tokens.permute(0, 2, 3, 4, 1).reshape(B, -1, C)

        for blk in self.transformer:
            tokens = blk(tokens)

        feat = tokens.view(B, -1, ph * pw, C).permute(0, 3, 1, 2)
        nh = H // ph
        nw = W // pw
        feat = feat.view(B, C, nh, nw, ph, pw).permute(0, 1, 2, 4, 3, 5)
        feat = feat.reshape(B, C, H, W)

        if feat.shape[-2:] != x.shape[-2:]:
            feat = F.interpolate(feat, size=x.shape[-2:], mode="bilinear", align_corners=False)

        out = self.fuse(torch.cat([x, feat], dim=1))
        return out


class SpatialReductionAttention(nn.Module):
    def __init__(self, dim, heads=8, sr_ratio=2, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, hw: Tuple[int, int]):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.heads, C // self.heads).permute(0, 2, 1, 3)

        if self.sr is not None:
            H, W = hw
            feat = x.transpose(1, 2).reshape(B, C, H, W)
            feat = self.sr(feat)
            feat = feat.reshape(B, C, -1).transpose(1, 2)
            feat = self.norm(feat)
        else:
            feat = x

        kv = self.kv(feat)
        k, v = kv.chunk(2, dim=-1)
        k = k.reshape(B, -1, self.heads, C // self.heads).permute(0, 2, 3, 1)
        v = v.reshape(B, -1, self.heads, C // self.heads).permute(0, 2, 1, 3)

        attn = torch.matmul(q, k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)
        out = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(B, N, C)
        out = self.proj(out)
        return out


class PVTBlock(nn.Module):
    def __init__(self, dim, heads=8, sr_ratio=2, dropout=0.0, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.sra = SpatialReductionAttention(dim, heads=heads, sr_ratio=sr_ratio, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x, hw: Tuple[int, int]):
        x = x + self.sra(self.norm1(x), hw)
        x = x + self.ff(self.norm2(x))
        return x


class LocalMambaBlock(nn.Module):
    def __init__(self, dim, kernel_size=5, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim)
        self.gate = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        g = torch.sigmoid(self.gate(x))
        x = (x * g).transpose(1, 2)
        x = self.dwconv(x).transpose(1, 2)
        x = self.proj(x)
        x = self.drop(x)
        return shortcut + x


class T2TRetokenizer(nn.Module):
    def __init__(self, dim, depth=2, heads=4, dropout=0.0):
        super().__init__()
        self.blocks = nn.ModuleList(
            [AttentionBlock(dim, heads=heads, dropout=dropout, mlp_ratio=2.0) for _ in range(depth)]
        )

    def forward(self, tokens: torch.Tensor, grid_hw: Tuple[int, int]):
        B, T, C = tokens.shape
        H, W = grid_hw
        feat_map = tokens.transpose(1, 2).reshape(B, C, H, W)
        seq = feat_map.flatten(2).transpose(1, 2)
        for blk in self.blocks:
            seq = blk(seq)
        seq_map = seq.transpose(1, 2).reshape(B, C, H, W)
        pooled = F.adaptive_avg_pool2d(seq_map, (2, 2))
        retokens = pooled.flatten(2).transpose(1, 2)
        return retokens, seq_map


class CrossScaleFusion(nn.Module):
    def __init__(self, dim, heads=6, dropout=0.0, layers=2):
        super().__init__()
        self.layers_s = nn.ModuleList(
            [AttentionBlock(dim, heads=heads, dropout=dropout, mlp_ratio=2.0) for _ in range(layers)]
        )
        self.layers_b = nn.ModuleList(
            [AttentionBlock(dim, heads=heads, dropout=dropout, mlp_ratio=2.0) for _ in range(layers)]
        )
        self.cross_s = nn.ModuleList(
            [
                nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True, kdim=dim, vdim=dim)
                for _ in range(layers)
            ]
        )
        self.cross_b = nn.ModuleList(
            [
                nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True, kdim=dim, vdim=dim)
                for _ in range(layers)
            ]
        )
        self.norm_s = nn.LayerNorm(dim)
        self.norm_b = nn.LayerNorm(dim)

    def forward(self, tok_s: torch.Tensor, tok_b: torch.Tensor):
        B, Ts, C = tok_s.shape
        Tb = tok_b.shape[1]
        cls_s = tok_s.new_zeros(B, 1, C)
        cls_b = tok_b.new_zeros(B, 1, C)
        tok_s = torch.cat([cls_s, tok_s], dim=1)
        tok_b = torch.cat([cls_b, tok_b], dim=1)

        for ls, lb, cs, cb in zip(self.layers_s, self.layers_b, self.cross_s, self.cross_b):
            tok_s = ls(tok_s)
            tok_b = lb(tok_b)
            q_s = self.norm_s(tok_s[:, :1])
            q_b = self.norm_b(tok_b[:, :1])
            cls_s_upd, _ = cs(
                q_s,
                torch.cat([tok_b, q_b], dim=1),
                torch.cat([tok_b, q_b], dim=1),
                need_weights=False,
            )
            cls_b_upd, _ = cb(
                q_b,
                torch.cat([tok_s, q_s], dim=1),
                torch.cat([tok_s, q_s], dim=1),
                need_weights=False,
            )
            tok_s = torch.cat([tok_s[:, :1] + cls_s_upd, tok_s[:, 1:]], dim=1)
            tok_b = torch.cat([tok_b[:, :1] + cls_b_upd, tok_b[:, 1:]], dim=1)

        tokens = torch.cat([tok_s[:, :1], tok_b[:, :1], tok_s[:, 1:], tok_b[:, 1:]], dim=1)
        return tokens


class TileEncoder(nn.Module):
    def __init__(self, backbone: nn.Module, input_res: int):
        super().__init__()
        self.backbone = backbone
        self.input_res = input_res

    def forward(self, x: torch.Tensor, grid: Tuple[int, int]):
        B, C, H, W = x.shape
        r, c = grid
        hs = torch.linspace(0, H, steps=r + 1, device=x.device).round().long()
        ws = torch.linspace(0, W, steps=c + 1, device=x.device).round().long()
        tiles = []
        for i in range(r):
            for j in range(c):
                rs, re = hs[i].item(), hs[i + 1].item()
                cs, ce = ws[j].item(), ws[j + 1].item()
                xt = x[:, :, rs:re, cs:ce]
                if xt.shape[-2:] != (self.input_res, self.input_res):
                    xt = F.interpolate(xt, size=(self.input_res, self.input_res), mode="bilinear", align_corners=False)
                tiles.append(xt)
        tiles = torch.stack(tiles, dim=1)
        flat = tiles.view(-1, C, self.input_res, self.input_res)
        feats = self.backbone(flat)
        feats = feats.view(B, -1, feats.shape[-1])
        return feats


class PyramidMixer(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dims: Tuple[int, int, int],
        mobilevit_heads: int = 4,
        mobilevit_depth: int = 2,
        sra_heads: int = 6,
        sra_ratio: int = 2,
        mamba_depth: int = 3,
        mamba_kernel: int = 5,
        dropout: float = 0.0,
    ):
        super().__init__()
        c1, c2, c3 = dims
        self.proj1 = nn.Linear(dim_in, c1)
        self.mobilevit = MobileViTBlock(c1, heads=mobilevit_heads, depth=mobilevit_depth, dropout=dropout)
        self.proj2 = nn.Linear(c1, c2)
        self.pvt = PVTBlock(c2, heads=sra_heads, sr_ratio=sra_ratio, dropout=dropout, mlp_ratio=3.0)
        self.mamba_local = LocalMambaBlock(c2, kernel_size=mamba_kernel, dropout=dropout)
        self.proj3 = nn.Linear(c2, c3)
        self.mamba_global = nn.ModuleList(
            [LocalMambaBlock(c3, kernel_size=mamba_kernel, dropout=dropout) for _ in range(mamba_depth)]
        )
        self.final_attn = AttentionBlock(c3, heads=min(8, c3 // 64 + 1), dropout=dropout, mlp_ratio=2.0)

    def _tokens_to_map(self, tokens: torch.Tensor, target_hw: Tuple[int, int]):
        B, N, C = tokens.shape
        H, W = target_hw
        need = H * W
        if N < need:
            pad = tokens.new_zeros(B, need - N, C)
            tokens = torch.cat([tokens, pad], dim=1)
        tokens = tokens[:, :need, :]
        feat_map = tokens.transpose(1, 2).reshape(B, C, H, W)
        return feat_map

    @staticmethod
    def _fit_hw(n_tokens: int) -> Tuple[int, int]:
        h = int(math.sqrt(n_tokens))
        w = h
        while h * w < n_tokens:
            w += 1
            if h * w < n_tokens:
                h += 1
        return h, w

    def forward(self, tokens: torch.Tensor):
        B, N, C = tokens.shape
        map_hw = (3, 4)
        feat_map = self._tokens_to_map(tokens, map_hw)

        t1 = self.proj1(tokens)
        m1 = self._tokens_to_map(t1, map_hw)
        m1 = self.mobilevit(m1)
        t1_out = m1.flatten(2).transpose(1, 2)[:, :N]

        t2 = self.proj2(t1_out)
        new_len = max(4, N // 2)
        t2 = t2[:, :new_len] + F.adaptive_avg_pool1d(t2.transpose(1, 2), new_len).transpose(1, 2)
        hw2 = self._fit_hw(t2.size(1))
        if t2.size(1) < hw2[0] * hw2[1]:
            pad = t2.new_zeros(B, hw2[0] * hw2[1] - t2.size(1), t2.size(2))
            t2 = torch.cat([t2, pad], dim=1)
        t2 = self.pvt(t2, hw2)
        t2 = self.mamba_local(t2)

        t3 = self.proj3(t2)
        pooled = torch.stack([t3.mean(dim=1), t3.max(dim=1).values], dim=1)
        t3 = pooled
        for blk in self.mamba_global:
            t3 = blk(t3)
        t3 = self.final_attn(t3)
        global_feat = t3.mean(dim=1)
        return global_feat, {"stage1_map": m1.detach(), "stage2_tokens": t2.detach(), "stage3_tokens": t3.detach()}


class CrossPVT_T2T_MambaDINO(nn.Module):
    def __init__(self, dropout: float = 0.1, hidden_ratio: float = 0.35):
        super().__init__()
        self.backbone, self.feat_dim, self.backbone_name, self.input_res = self._build_dino_backbone()
        self.tile_encoder = TileEncoder(self.backbone, self.input_res)
        self.t2t = T2TRetokenizer(self.feat_dim, depth=CFG.t2t_depth, heads=CFG.cross_heads, dropout=dropout)
        self.cross = CrossScaleFusion(
            self.feat_dim, heads=CFG.cross_heads, dropout=dropout, layers=CFG.cross_layers
        )
        self.pyramid = PyramidMixer(
            dim_in=self.feat_dim,
            dims=CFG.pyramid_dims,
            mobilevit_heads=CFG.mobilevit_heads,
            mobilevit_depth=CFG.mobilevit_depth,
            sra_heads=CFG.sra_heads,
            sra_ratio=CFG.sra_ratio,
            mamba_depth=CFG.mamba_depth,
            mamba_kernel=CFG.mamba_kernel,
            dropout=dropout,
        )

        combined = CFG.pyramid_dims[-1] * 2
        self.combined_dim = combined
        hidden = max(32, int(combined * hidden_ratio))

        def head():
            return nn.Sequential(
                nn.Linear(combined, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )

        self.head_green = head()
        self.head_clover = head()
        self.head_dead = head()
        self.score_head = nn.Sequential(nn.LayerNorm(combined), nn.Linear(combined, 1))
        self.aux_head = (
            nn.Sequential(nn.LayerNorm(CFG.pyramid_dims[1]), nn.Linear(CFG.pyramid_dims[1], 5))
            if CFG.aux_head
            else None
        )
        self.softplus = nn.Softplus(beta=1.0)

        self.cross_gate_left = nn.Linear(CFG.pyramid_dims[-1], CFG.pyramid_dims[-1])
        self.cross_gate_right = nn.Linear(CFG.pyramid_dims[-1], CFG.pyramid_dims[-1])

    def _build_dino_backbone(self):
        last_err = None
        for name in CFG.dino_candidates:
            for gp in ["token", "avg", "__default__"]:
                try:
                    if gp == "__default__":
                        m = timm.create_model(name, pretrained=False, num_classes=0)
                        gp_str = "default"
                    else:
                        m = timm.create_model(name, pretrained=False, num_classes=0, global_pool=gp)
                        gp_str = gp
                    feat = m.num_features
                    input_res = self._infer_input_res(m)
                    if hasattr(m, "set_grad_checkpointing"):
                        m.set_grad_checkpointing(True)
                    return m, feat, name, int(input_res)
                except Exception as e:
                    last_err = e
                    continue
        raise RuntimeError(f"Cannot create DINO backbone. Last error: {last_err}")

    @staticmethod
    def _infer_input_res(m) -> int:
        if hasattr(m, "patch_embed") and hasattr(m.patch_embed, "img_size"):
            isz = m.patch_embed.img_size
            return int(isz if isinstance(isz, (int, float)) else isz[0])
        if hasattr(m, "img_size"):
            isz = m.img_size
            return int(isz if isinstance(isz, (int, float)) else isz[0])
        dc = getattr(m, "default_cfg", {}) or {}
        ins = dc.get("input_size", None)
        if ins:
            if isinstance(ins, (tuple, list)) and len(ins) >= 2:
                return int(ins[1])
            return int(ins if isinstance(ins, (int, float)) else 224)
        return 518

    def _half_forward(self, x_half: torch.Tensor):
        tiles_small = self.tile_encoder(x_half, CFG.small_grid)
        tiles_big = self.tile_encoder(x_half, CFG.big_grid)
        t2, stage1_map = self.t2t(tiles_small, CFG.small_grid)
        fused = self.cross(t2, tiles_big)
        feat, feat_maps = self.pyramid(fused)
        feat_maps["stage1_map"] = stage1_map
        return feat, feat_maps

    def _merge_heads(self, f_l: torch.Tensor, f_r: torch.Tensor):
        g_l = torch.sigmoid(self.cross_gate_left(f_r))
        g_r = torch.sigmoid(self.cross_gate_right(f_l))
        f_l = f_l * g_l
        f_r = f_r * g_r
        f = torch.cat([f_l, f_r], dim=1)
        green_pos = self.softplus(self.head_green(f))
        clover_pos = self.softplus(self.head_clover(f))
        dead_pos = self.softplus(self.head_dead(f))
        gdm = green_pos + clover_pos
        total = gdm + dead_pos
        return total, gdm, green_pos, f

    def forward(self, *inputs, x_left=None, x_right=None, return_features: bool = False):
        if inputs:
            if len(inputs) == 1:
                first = inputs[0]
                if isinstance(first, (tuple, list)):
                    if len(first) >= 1:
                        x_left = first[0]
                    if len(first) >= 2:
                        x_right = first[1]
                else:
                    x_left = first
            else:
                x_left = inputs[0]
                x_right = inputs[1]

        if x_left is None:
            return {}

        if x_right is None:
            if isinstance(x_left, torch.Tensor):
                if x_left.shape[1] % 2 != 0:
                    raise ValueError("Cannot infer left/right branches from single tensor.")
                x_left, x_right = torch.chunk(x_left, 2, dim=1)
            else:
                raise ValueError("Missing x_right input.")

        feat_l, feats_l = self._half_forward(x_left)
        feat_r, feats_r = self._half_forward(x_right)
        total, gdm, green, f_concat = self._merge_heads(feat_l, feat_r)

        out = {
            "total": total,
            "gdm": gdm,
            "green": green,
            "score_feat": f_concat,
        }
        if self.aux_head is not None:
            aux_tokens = torch.cat([feats_l["stage2_tokens"], feats_r["stage2_tokens"]], dim=1)
            aux_pred = self.softplus(self.aux_head(aux_tokens.mean(dim=1)))
            out["aux"] = aux_pred
        if return_features:
            out["feature_maps"] = {
                "stage1_left": feats_l.get("stage1_map"),
                "stage1_right": feats_r.get("stage1_map"),
                "stage3_left": feats_l.get("stage3_tokens"),
                "stage3_right": feats_r.get("stage3_tokens"),
            }
        return out



class FiLM(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        hidden = max(32, feat_dim // 2)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, feat_dim * 2)
        )

    def forward(self, context):
        gamma_beta = self.mlp(context)
        return torch.chunk(gamma_beta, 2, dim=1)


class BaseDINO(nn.Module):
    def __init__(self, backbone_name):
        super().__init__()
        self.dropout = 0.30
        self.hidden_ratio = 0.25
        self.grid = (2, 2)
        self.backbone_name = backbone_name
        
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        self.feat_dim = self.backbone.num_features
        self.input_size = self._get_input_size(self.backbone)
        self.combined_dim = self.feat_dim * 2
        hidden_size = max(8, int(self.combined_dim * self.hidden_ratio))

        def make_head():
            return nn.Sequential(
                nn.Linear(self.combined_dim, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout),
                nn.Linear(hidden_size, 1),
            )

        self.head_green = make_head()
        self.head_clover = make_head()
        self.head_dead = make_head()
        self.softplus = nn.Softplus(beta=1.0)

    def _get_input_size(self, model):
        if hasattr(model, "patch_embed") and hasattr(model.patch_embed, "img_size"):
            size = model.patch_embed.img_size
            return int(size if isinstance(size, (int, float)) else size[0])
        
        if hasattr(model, "img_size"):
            size = model.img_size
            return int(size if isinstance(size, (int, float)) else size[0])
        
        cfg = getattr(model, "default_cfg", {}) or {}
        input_size = cfg.get("input_size", None)
        
        if input_size:
            if isinstance(input_size, (tuple, list)) and len(input_size) >= 2:
                return int(input_size[1])
            return int(input_size if isinstance(input_size, (int, float)) else 224)
        
        arch = cfg.get("architecture", "") or str(type(model))
        return 518 if "dinov2" in arch.lower() or "dinov3" in arch.lower() else 224

    def merge_features(self, left_feat, right_feat):
        combined = torch.cat([left_feat, right_feat], dim=1)
        green = self.softplus(self.head_green(combined))
        clover = self.softplus(self.head_clover(combined))
        dead = self.softplus(self.head_dead(combined))
        gdm = green + clover
        total = gdm + dead
        return total, gdm, green


class TiledFiLMDINO(BaseDINO):
    def __init__(self, backbone_name):
        super().__init__(backbone_name)
        self.film_left = FiLM(self.feat_dim)
        self.film_right = FiLM(self.feat_dim)

    def _split_dimension(self, length, parts):
        step = length // parts
        segments = []
        start = 0
        
        for _ in range(parts - 1):
            segments.append((start, start + step))
            start += step
        
        segments.append((start, length))
        return segments

    def _extract_tile_features(self, x):
        B, C, H, W = x.shape
        rows, cols = self.grid
        row_segments = self._split_dimension(H, rows)
        col_segments = self._split_dimension(W, cols)
        features = []
        
        for (rs, re) in row_segments:
            for (cs, ce) in col_segments:
                tile = x[:, :, rs:re, cs:ce]
                if tile.shape[-2:] != (self.input_size, self.input_size):
                    tile = F.interpolate(tile, size=(self.input_size, self.input_size), mode="bilinear")
                feat = self.backbone(tile)
                features.append(feat)
        
        return torch.stack(features, dim=0).permute(1, 0, 2)

    def _process_stream(self, x, film_layer):
        tiles = self._extract_tile_features(x)
        context = tiles.mean(dim=1)
        gamma, beta = film_layer(context)
        modulated = tiles * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        return modulated.mean(dim=1)

    def forward(self, left_img, right_img):
        left_feat = self._process_stream(left_img, self.film_left)
        right_feat = self._process_stream(right_img, self.film_right)
        return self.merge_features(left_feat, right_feat)


class BiomassDataset(Dataset):
    def __init__(self, df, transform, img_dir):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.img_dir = img_dir
        self.paths = self.df["image_path"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filename = os.path.basename(self.paths[idx])
        full_path = os.path.join(self.img_dir, filename)
        
        img = cv2.imread(full_path)
        if img is None:
            img = np.zeros((1000, 2000, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w, _ = img.shape
        mid = w // 2
        left_img = img[:, :mid]
        right_img = img[:, mid:]

        left_tensor = self.transform(image=left_img)["image"]
        right_tensor = self.transform(image=right_img)["image"]
        
        return left_tensor, right_tensor


def get_tta_transforms_v4(img_size: int):
    base = [
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]

    transforms = []
    transforms.append(
        A.Compose([
            A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
            *base,
        ])
    )

    transforms.append(
        A.Compose([
            A.HorizontalFlip(p=1.0),
            A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
            *base,
        ])
    )

    transforms.append(
        A.Compose([
            A.VerticalFlip(p=1.0),
            A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),
            *base,
        ])
    )

    return transforms


def get_tta_transforms_mvp(img_size):
    norm = [A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]
    return [
        A.Compose([A.Resize(img_size, img_size), *norm]),
        A.Compose([A.HorizontalFlip(p=1.0), A.Resize(img_size, img_size), *norm]),
        A.Compose([A.VerticalFlip(p=1.0), A.Resize(img_size, img_size), *norm]),
        A.Compose([A.RandomRotate90(p=1.0), A.Resize(img_size, img_size), *norm]),
    ]


def strip_module_prefix(state_dict: dict) -> dict:
    if not state_dict:
        return state_dict

    keys = list(state_dict.keys())
    if all(k.startswith("module.") for k in keys):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def clean_state_dict_mvp(state_dict):
    if not state_dict:
        return state_dict
    
    cleaned_dict = {}
    
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        
        if k.startswith("student."):
            k = k[8:]
        
        skip_prefixes = ("txt_enc.", "img_proj.", "txt_film", "teacher.", "momentum_teacher.")
        if any(k.startswith(prefix) for prefix in skip_prefixes):
            continue
            
        cleaned_dict[k] = v
    
    return cleaned_dict


def load_checkpoint_v4(path: str) -> nn.Module:
    state = torch.load(path, map_location="cpu", weights_only=False)
    cfg_dict = state.get("cfg", {})
    dropout = cfg_dict.get("dropout", CFG.dropout)
    hidden_ratio = cfg_dict.get("hidden_ratio", CFG.hidden_ratio)

    model = CrossPVT_T2T_MambaDINO(dropout=dropout, hidden_ratio=hidden_ratio)

    model_state = state.get("model_state")
    if model_state is None:
        model_state = state

    model_state = strip_module_prefix(model_state)
    model.load_state_dict(model_state, strict=False)
    model.to(CFG.device)
    model.eval()

    return model


def load_checkpoint_mvp(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        return None
    
    try:
        raw_state = torch.load(checkpoint_path, map_location=CFG.device, weights_only=False)
    except Exception:
        return None
    
    if isinstance(raw_state, dict):
        if 'state_dict' in raw_state:
            state_dict = raw_state['state_dict']
        elif 'model' in raw_state:
            state_dict = raw_state['model']
        else:
            state_dict = raw_state
    else:
        state_dict = raw_state
    
    state_dict = clean_state_dict_mvp(state_dict)
    
    if not state_dict:
        return None
    
    # Используем те же backbone, что и в исходном рабочем коде
    backbones = [
        "vit_base_patch14_reg4_dinov2",
        "vit_base_patch14_reg4_dinov3",
        "vit_base_patch14_dinov3",
    ]
    
    for backbone in backbones:
        try:
            model = TiledFiLMDINO(backbone)
            result = model.load_state_dict(state_dict, strict=False)
            
            missing = [k for k in result.missing_keys if not k.startswith('backbone.pos_embed')]
            
            if len(missing) == 0:
                model.to(CFG.device)
                model.eval()
                return model
                
        except Exception:
            continue
    
    return None


@torch.no_grad()
def predict_one_view_v4(models: List[nn.Module], loader: DataLoader) -> np.ndarray:
    preds_list = []
    amp_dtype = "cuda" if CFG.device.type == "cuda" else "cpu"
    
    for xl, xr in tqdm(loader, leave=False):
        xl = xl.to(CFG.device, non_blocking=True)
        xr = xr.to(CFG.device, non_blocking=True)
        x_cat = torch.cat([xl, xr], dim=1)
        
        per_model_preds = []
        
        with torch.amp.autocast(amp_dtype, enabled=CFG.mixed_precision):
            for model in models:
                out = model(x_cat, return_features=False)
                
                total = out["total"]
                gdm = out["gdm"]
                green = out["green"]
                
                dead = total - gdm
                clover = gdm - green
                five = torch.cat([green, dead, clover, gdm, total], dim=1)
                
                per_model_preds.append(five.float().cpu())
        
        stacked = torch.mean(torch.stack(per_model_preds, dim=0), dim=0)
        preds_list.append(stacked.numpy())
    
    return np.concatenate(preds_list, axis=0)


@torch.no_grad()
def predict_one_view_mvp(models: List[nn.Module], loader: DataLoader) -> np.ndarray:
    preds = []
    use_amp = CFG.device.type == "cuda"
    
    for left_imgs, right_imgs in tqdm(loader, leave=False):
        left_imgs = left_imgs.to(CFG.device, non_blocking=True)
        right_imgs = right_imgs.to(CFG.device, non_blocking=True)
        batch_preds = []
        
        with torch.amp.autocast("cuda", enabled=use_amp):
            for model in models:
                total, gdm, green = model(left_imgs, right_imgs)
                dead = torch.clamp(total - gdm, min=0.0)
                clover = torch.clamp(gdm - green, min=0.0)
                pred = torch.cat([green, dead, clover, gdm, total], dim=1)
                batch_preds.append(pred.clamp(0.05, 400.0).cpu())
        
        preds.append(torch.stack(batch_preds).mean(dim=0).numpy())
    
    return np.concatenate(preds, axis=0)


def run_inference_v4(test_df: pd.DataFrame, image_dir: str) -> np.ndarray:
    models = []
    
    for fold in range(CFG.n_folds_a):
        ckpt_path = CFG.ckpt_pattern_fold_x_a.format(fold=fold)
        if not os.path.exists(ckpt_path):
            ckpt_path = CFG.ckpt_pattern_foldx_a.format(fold=fold)
        
        if not os.path.exists(ckpt_path):
            continue
        
        model = load_checkpoint_v4(ckpt_path)
        models.append(model)
    
    if not models:
        raise RuntimeError("No V4 models loaded!")
    
    input_size = getattr(models[0], "input_res", 518)
    
    if CFG.use_tta:
        tta_transforms = get_tta_transforms_v4(input_size)
        per_view_preds = []
        
        for transform in tta_transforms:
            ds = BiomassDataset(test_df, transform, image_dir)
            dl = DataLoader(
                ds,
                batch_size=CFG.batch_size,
                shuffle=False,
                num_workers=CFG.num_workers,
                pin_memory=True,
            )
            view_pred = predict_one_view_v4(models, dl)
            per_view_preds.append(view_pred)
        
        final_pred = np.mean(per_view_preds, axis=0)
    else:
        transform = get_tta_transforms_v4(input_size)[0]
        ds = BiomassDataset(test_df, transform, image_dir)
        dl = DataLoader(
            ds,
            batch_size=CFG.batch_size,
            shuffle=False,
            num_workers=CFG.num_workers,
            pin_memory=True,
        )
        final_pred = predict_one_view_v4(models, dl)
    
    return final_pred


def run_inference_mvp(checkpoint_paths, df, img_dir) -> np.ndarray:
    models = []
    
    for ckpt_path in checkpoint_paths:
        model = load_checkpoint_mvp(ckpt_path)
        if model is not None:
            models.append(model)
    
    if not models:
        raise ValueError("No MVP models loaded!")
    
    input_size = models[0].input_size
    
    tta_preds = []
    for transform in get_tta_transforms_mvp(input_size):
        ds = BiomassDataset(df, transform, img_dir)
        dl = DataLoader(ds, batch_size=CFG.batch_size, shuffle=False, num_workers=0, pin_memory=True)
        tta_preds.append(predict_one_view_mvp(models, dl))
    
    return np.mean(tta_preds, axis=0)


def create_submission(final_pred: np.ndarray, test_long: pd.DataFrame, test_unique: pd.DataFrame) -> pd.DataFrame:
    green = final_pred[:, 0]
    dead = final_pred[:, 1]
    clover = final_pred[:, 2]
    gdm = final_pred[:, 3]
    total = final_pred[:, 4]

    def clean(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return np.maximum(0, x)

    green, dead, clover, gdm, total = map(clean, [green, dead, clover, gdm, total])

    wide = pd.DataFrame(
        {
            "image_path": test_unique["image_path"],
            "Dry_Green_g": green,
            "Dry_Dead_g": dead,
            "Dry_Clover_g": clover,
            "GDM_g": gdm,
            "Dry_Total_g": total,
        }
    )

    long_preds = wide.melt(
        id_vars=["image_path"],
        value_vars=CFG.all_target_cols,
        var_name="target_name",
        value_name="target",
    )

    sub = pd.merge(
        test_long[["sample_id", "image_path", "target_name"]],
        long_preds,
        on=["image_path", "target_name"],
        how="left",
    )[["sample_id", "target"]]

    sub["target"] = np.nan_to_num(sub["target"], nan=0.0, posinf=0.0, neginf=0.0)
    sub.to_csv(CFG.submission_file, index=False)
    
    return sub


test_df = pd.read_csv(CFG.test_csv)
unique_df = test_df.drop_duplicates(subset=["image_path"]).reset_index(drop=True)

try:
    pred_v4 = run_inference_v4(unique_df, CFG.test_image_dir)
except Exception as e:
    print(f"V4 inference failed: {e}")
    pred_v4 = None

try:
    pred_mvp_a = run_inference_mvp(CFG.ckpts_a, unique_df, CFG.test_image_dir)
    pred_mvp_b = run_inference_mvp(CFG.ckpts_b, unique_df, CFG.test_image_dir)
    pred_mvp = CFG.weight_a * pred_mvp_a + CFG.weight_b * pred_mvp_b
except Exception as e:
    print(f"MVP inference failed: {e}")
    pred_mvp = None
    
final_pred = (pred_v4 + pred_mvp) / 2.0
    
submission = create_submission(final_pred, test_df, unique_df)
submission.head()