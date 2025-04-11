import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import copy
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



class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd."
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 平均池化
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 最大池化
        x = torch.cat([avg_out, max_out], dim=1)  # 拼接两个池化结果
        x = self.conv1(x)  # 卷积操作
        return self.sigmoid(x)  # Sigmoid激活


class ChannelAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels // reduction_ratio, num_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))  # 全局平均池化后通过全连接层
        return avg_out.view(b, c, 1, 1)  # 调整形状

d_model=49
nhead=7
num_encoder_layers=3

# 定义TransformerEncorder模型
class TransformerEncorder(nn.Module):
    def __init__(self, d_model=49, nhead=7, num_encoder_layers=3):
        super(TransformerEncorder, self).__init__()
        self.trans = nn.Transformer(d_model=d_model,
                                    nhead=nhead,
                                    num_encoder_layers=num_encoder_layers)

    def forward(self, xb):
        xb = self.trans.encoder(xb)  # 编码器部分

        return xb

class HAM(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16, kernel_size=7):
        super(HAM, self).__init__()
        self.channel_attention = ChannelAttention(num_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
        self.con_1_1 = nn.Sequential(
            nn.Conv2d(2048, 512, 1),  # 1x1卷积降维
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.TransformerEncorder = TransformerEncorder(d_model=49, nhead=7, num_encoder_layers=3)

    def forward(self, x):
        x = x * self.channel_attention(x)  # 通道注意力
        x = x * self.spatial_attention(x)  # 空间注意力
        x = self.con_1_1(x)
        x = x.view(32, 512, -1)
        x = self.TransformerEncorder(x)

        return x

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


class BiScaleCrossViT(nn.Module):
    def __init__(self, img_size=224, patch_sizes=[32, 16, 8], in_channels=3, num_classes=1000,
                 embed_dim=768, depth=3, num_heads=6, mlp_ratio=4.0, dropout=0.1):
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
        self.head = nn.Linear(embed_dim * len(patch_sizes), num_classes)
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

        # 交叉注意力融合
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


class TransformerEncorder_Fuse(nn.Module):
    def __init__(self, d_model=16, nhead=1, num_encoder_layers=2):
        super(TransformerEncorder_Fuse, self).__init__()
        self.trans = nn.Transformer(d_model=d_model,
                                    nhead=nhead,
                                    num_encoder_layers=num_encoder_layers)

    def forward(self, xb):
        xb = self.trans.encoder(xb)  # 编码器部分
        return xb

class DimReduction(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 16)
        )

    def forward(self, x):
        return self.fc(x)



# --------------------------------#
class AACSFNet(nn.Module):
    def __init__(self):
        super(AACSFNet, self).__init__()

        #调用有与训练权重的resnet50模型
        resnet50 = models.resnet50(weights=ResNet50_Weights.DEFAULT)

        # 加载修改后的模型，与训练模型参数保留到layer4（仅保留到 layer4）
        resnet50_modified = torch.nn.Sequential(
            resnet50.conv1,
            resnet50.bn1,
            resnet50.relu,
            resnet50.maxpool,
            resnet50.layer1,
            resnet50.layer2,
            resnet50.layer3,
            resnet50.layer4
        )

        self.resnet50_modified = resnet50_modified

        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(32, 1)
        )

        self.TransformerEncorder_Fuse = TransformerEncorder_Fuse(d_model=16,nhead=1,num_encoder_layers=2)

        self.BiScaleCrossViT = BiScaleCrossViT(
            patch_sizes=[32, 16, 8],
            num_classes=16,
            depth=3,
            embed_dim=192)
        self.DimReduction = DimReduction()
        self.HAM =  HAM(num_channels=2048, reduction_ratio=16, kernel_size=7)
        self.pool = nn.AdaptiveAvgPool1d(1)


    # ------------------------------#
    # AACSFNET的前向传播函数
    # ------------------------------#
    def forward(self, x):
        x = x
        x1 = detach(x).clone()
        x2 = detach(x).clone()

        x1 = self.resnet50_modified(x1)
        x1 = self.HAM(x1)
        x1 = self.pool(x1)
        #调整维度
        x1 = x1.view(32, 512 * 1*1)  # 形状: [32, 25088]
        # 使用全连接降维度
        x1 = self.DimReduction(x1)


        x2 = self.BiScaleCrossViT(x2)

        #将x1特征和x2特征进行堆叠为融合准备
        trans_in = torch.stack([x1, x2], dim=1)  # 形状: [32, 2, 16]
        trans_out = self.TransformerEncorder_Fuse(trans_in)
        trans_residual = torch.cat([trans_in, trans_out], dim=2)
        trans_residual = torch.reshape(trans_residual, [trans_residual.shape[0], -1])
        trans_residual = self.fc(trans_residual)

        return trans_residual




