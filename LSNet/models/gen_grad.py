# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit_pytorch.py

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from functools import partial
from itertools import repeat
from collections import OrderedDict
from einops import rearrange
from einops.layers.torch import Rearrange
import math
from pygcn.models import GCN
import numpy as np
# import cupy as cp


from models import *
# from torch._six import container_abcs

# From PyTorch internals
def _ntuple(n):
    def parse(x):
        # if isinstance(x, container_abcs.Iterable):
        #     return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
MIN_NUM_PATCHES = 16


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        temp = self.norm(x)
        return self.fn(temp, *args, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.GELU(),
            # nn.Dropout(dropout)
        )

    def forward(self, x):
        # b, n, _ = x.shape

        return self.net(x)


class GenTransformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout, att_size):
        super().__init__()

        self.layers = nn.ModuleList([])
        self.heads = heads
        for _ in range(depth):
            self.layers.append(
                # Residual(PreNorm(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))))


        # self.N = att_size ** 2  # 确保与梯度图的 N 一致
        # 可选：添加线性层调整 AA^T 的尺度（避免数值过大）
        # self.scale_layer = nn.Sequential(
        #     nn.Linear(self.N, self.N),  # 保持 N×N 形状
        #     nn.GELU()
        # )


        self.conv_dim = dim//heads*4
        #使用深度可分离卷积进一步减少FLOPs
        self.conv_layer = nn.Sequential(OrderedDict([
            ('depthwise', nn.Conv2d(
                in_channels=self.conv_dim,
                out_channels=self.conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=self.conv_dim  # 深度可分离卷积
            )),
            ('pointwise', nn.Conv2d(
                in_channels=self.conv_dim,
                out_channels=self.conv_dim,
                kernel_size=1
            )),
            ('bn', nn.BatchNorm2d(self.conv_dim)),
        ]))
        self.att_activation = nn.SELU()

    def forward(self, x, mask=None):

        x = rearrange(x.squeeze(3), 'b p n (h d) -> b p h n d', h=self.heads)
        b, p, h, n, d = x.shape
        w = int(n ** 0.5)
        conv_input = x.reshape(b * h, w, w, d * p).permute(0, 3, 1, 2)
        conv_output = self.conv_layer(conv_input)
        x_heads = rearrange(conv_output, '(b h) (p d) w v -> b p h (w v) d', b=b, p=p)

        # for ff in self.layers:
        #     x = ff(x)
        # dim_per_head = torch.div(x.shape[-1], self.heads, rounding_mode='trunc')
        # x_heads = rearrange(x, "b p n (h d) -> b p h n d ", h=self.heads, d=dim_per_head)

        attn_map = torch.matmul(x_heads, x_heads.transpose(-2, -1))  # 形状：(batch, 8, N, N)
        attn_map = self.att_activation(attn_map)

        # attn_map = self.scale_layer(attn_map)  # 输入输出均为(batch, 8, N, N)

        return attn_map