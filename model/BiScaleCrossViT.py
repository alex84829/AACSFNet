import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import copy
import torch
from torchvision import models
from torchvision.models import ResNet50_Weights
from torch import detach
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from timm.layers import to_2tuple
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import time

import math
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=6, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class CrossAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=6, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        B, N_Q, _ = query.shape
        B, N_K, _ = key.shape
        B, N_V, _ = value.shape

        q = self.q(query).reshape(B, N_Q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(key).reshape(B, N_K, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(value).reshape(B, N_V, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N_Q, self.embed_dim)
        x = self.proj(x)
        x = self.proj_dropout(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=6, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, E = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, E)
        x = self.proj(x)
        x = self.proj_dropout(x)
        return x

class MLP(nn.Module):
    def __init__(self, embed_dim=768, hidden_dim=3072, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MultiScaleVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_sizes=[32, 16], in_channels=3, num_classes=1,  # 修改默认值
                 embed_dim=192, depth=3, num_heads=6, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.patch_embeds = nn.ModuleList([
            PatchEmbedding(img_size, ps, in_channels, embed_dim) for ps in patch_sizes
        ])
        self.cls_tokens = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 1, embed_dim)) for _ in patch_sizes
        ])
        self.pos_embeds = nn.ParameterList([
            nn.Parameter(torch.zeros(1, p_embed.n_patches + 1, embed_dim))
            for p_embed in self.patch_embeds
        ])

        self.blocks = nn.ModuleList([
            nn.ModuleList([
                TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]) for _ in patch_sizes
        ])

        self.cross_attn = CrossAttention(embed_dim, num_heads, dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim * len(patch_sizes), num_classes)  # 自动适配双尺度
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        multi_scale_features = []
        for i, (patch_embed, cls_token, pos_embed, blocks) in enumerate(zip(
                self.patch_embeds, self.cls_tokens, self.pos_embeds, self.blocks)):

            B = x.shape[0]
            x_embed = patch_embed(x)
            cls_tokens = cls_token.expand(B, -1, -1)
            x_embed = torch.cat((cls_tokens, x_embed), dim=1)
            x_embed += pos_embed
            x_embed = self.dropout(x_embed)

            for blk in blocks:
                x_embed = blk(x_embed)

            x_embed = self.norm(x_embed)
            multi_scale_features.append(x_embed[:, 0])

        # 交叉注意力融合（自动适配双尺度）
        cross_outputs = []
        for i in range(len(multi_scale_features)):
            query_feat = multi_scale_features[i].unsqueeze(1)
            other_feats = [feat for j, feat in enumerate(multi_scale_features) if j != i]
            key_val_feats = torch.stack(other_feats, dim=1)
            attn_out = self.cross_attn(query_feat, key_val_feats, key_val_feats)
            cross_outputs.append(attn_out.squeeze(1))

        combined_features = torch.cat(cross_outputs, dim=1)
        output = self.head(combined_features)
        return output


class BiscaleCrossVitnet(nn.Module):
    def __init__(self):
        super(BiscaleCrossVitnet, self).__init__()

        self.MultiScaleVisionTransformer = MultiScaleVisionTransformer(patch_sizes=[32, 16],  # 双尺度
        num_classes=1,
        depth=3,
        embed_dim=192)

    # ------------------------------#
    # resnet50的前向传播函数
    # ------------------------------#
    def forward(self, x):
        x = self.MultiScaleVisionTransformer(x)

        return x




