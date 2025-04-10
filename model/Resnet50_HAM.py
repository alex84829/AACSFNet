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


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd."
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True) 
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)  # 拼接两个池化结果
        x = self.conv1(x)  
        return self.sigmoid(x) 


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


class HAM(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16, kernel_size=7):
        super(HAM, self).__init__()
        self.channel_attention = ChannelAttention(num_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)  # 通道注意力
        x = x * self.spatial_attention(x)  # 空间注意力

        return x

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

        # Transformer编码器
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



class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

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
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(8, 1)
        )

        self.con_1_1 = nn.Sequential(
    nn.Conv2d(2048, 512, 1),  # 1x1卷积降维
    nn.BatchNorm2d(512),
    nn.ReLU()
)

        self.TransformerEncorder = TransformerEncorder(d_model=49,nhead=7,num_encoder_layers=3)
        self.DimReduction = DimReduction()
        self.HAM =  HAM(num_channels=2048, reduction_ratio=16, kernel_size=7)
        self.pool = nn.AdaptiveAvgPool1d(1)


    def forward(self, x):
        x1 = x
        x1 = self.resnet50_modified(x1)
        x1 = self.HAM(x1)
        x1 = self.con_1_1(x1)
        x1 = x1.view(32, 512, -1)
        x1 = self.TransformerEncorder(x1)
        x1 = self.pool(x1)
        #调整维度
        x1 = x1.view(32, 512 * 1*1)  # 形状: [32, 25088]
        # 使用全连接降维度
        x1 = self.DimReduction(x1)
        x1 = self.fc(x1)


        return x1




