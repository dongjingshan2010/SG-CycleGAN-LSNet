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


# class Residual(nn.Module):
#     def __init__(self, fn):
#         super().__init__()
#         self.fn = fn
#     def forward(self, x, *args, **kwargs):
#         return self.fn(x, *args, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        temp = self.norm(x)
        return self.fn(temp, *args, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, image_size, patch_size, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            # nn.Linear(dim, 64),
            # nn.Linear(64, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            # nn.Linear(hidden_dim, 64),
            # nn.Linear(64, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, _ = x.shape

        return self.net(x)


class Attention(nn.Module):
    def __init__(self,
                 dim,
                 heads = 8,
                 dropout = 0,
                 qkv_bias = False,
                 attn_drop= 0.,
                 proj_drop=0.,
                 method='dw_bn',
                 kernel_size = 3,
                 stride_kv = 1,
                 stride_q = 1,
                 padding_kv = 1,
                 padding_q = 1,
                 with_cls_token=True
                 ):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        # self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        # self.to_out = nn.Sequential(
        #     nn.Linear(dim, dim),
        #     nn.Dropout(dropout)
        # )
        # 添加预训练阶段标志
        self.is_pretraining = False

        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.dim = dim
        self.num_heads = heads
        self.scale = dim ** -0.5
        self.with_cls_token = with_cls_token

        dim_in = dim
        dim_out = dim
        self.conv_proj_q = self._build_projection(
            dim_in, dim_out, kernel_size, padding_q,
            stride_q, 'linear' if method == 'avg' else method
        )
        self.conv_proj_k = self._build_projection(
            dim_in, dim_out, kernel_size, padding_kv,
            stride_kv, method
        )
        self.conv_proj_v = self._build_projection(
            dim_in, dim_out, kernel_size, padding_kv,
            stride_kv, method
        )

        self.proj_q = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_k = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_v = nn.Linear(dim_in, dim_out, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_out, dim_out)
        # self.proj1 = nn.Linear(dim_out, 64)
        # self.proj2 = nn.Linear(64, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

    def set_pretraining_mode(self, is_pretraining):
        """设置预训练模式标志"""
        self.is_pretraining = is_pretraining

    def _build_projection(self,
                      dim_in,
                      dim_out,
                      kernel_size,
                      padding,
                      stride,
                      method):
        if method == 'dw_bn':
            proj = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(
                dim_in,
                dim_in,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                bias=False,
                groups=dim_in
                )),
                ('bn', nn.BatchNorm2d(dim_in)),
                ('rearrage', Rearrange('b c h w -> b (h w) c')),
            ]))
        elif method == 'avg':
            proj = nn.Sequential(OrderedDict([
                ('avg', nn.AvgPool2d(
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                ceil_mode=True
                )),
                ('rearrage', Rearrange('b c h w -> b (h w) c')),
            ]))
        elif method == 'linear':
            proj = None
        else:
            raise ValueError('Unknown method ({})'.format(method))

        return proj

    def forward_conv(self, x, h, w):
        if self.with_cls_token:
            cls_token, x = torch.split(x, [1, h*w], 1)

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        if self.conv_proj_q is not None:
            q = self.conv_proj_q(x)
        else:
            q = rearrange(x, 'b c h w -> b (h w) c')

        if self.conv_proj_k is not None:
            k = self.conv_proj_k(x)
        else:
            k = rearrange(x, 'b c h w -> b (h w) c')

        if self.conv_proj_v is not None:
            v = self.conv_proj_v(x)
        else:
            v = rearrange(x, 'b c h w -> b (h w) c')

        if self.with_cls_token:
            q = torch.cat((cls_token, q), dim=1)
            k = torch.cat((cls_token, k), dim=1)
            v = torch.cat((cls_token, v), dim=1)

        return q, k, v

    def forward(self, x, h, w, mask=None):

        if (
            self.conv_proj_q is not None
            or self.conv_proj_k is not None
            or self.conv_proj_v is not None
        ):
            q, k, v = self.forward_conv(x, h, w)

        q = rearrange(self.proj_q(q), 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(self.proj_k(k), 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(self.proj_v(v), 'b t (h d) -> b h t d', h=self.num_heads)

        # q = rearrange((q), 'b t (h d) -> b h t d', h=self.num_heads)
        # k = rearrange((k), 'b t (h d) -> b h t d', h=self.num_heads)
        # v = rearrange((v), 'b t (h d) -> b h t d', h=self.num_heads)

        attn_score = torch.einsum('bhlk,bhtk->bhlt', [q, k]) * self.scale

        # 只在预训练阶段保存注意力分数和梯度
        if self.training and self.is_pretraining:
            pass
        else:
            # 非预训练阶段，保存attn_score属性
            self.attn_score = attn_score
            self.attn_score.retain_grad()
        # 使用局部变量进行计算，避免依赖实例属性
        current_attn_score = attn_score if not hasattr(self, 'attn_score') else self.attn_score

        attn = F.softmax(current_attn_score, dim=-1)
        attn = self.attn_drop(attn)

        x = torch.einsum('bhlt,bhtv->bhlv', [attn, v])
        x = rearrange(x, 'b h t d -> b t (h d)')

        x = self.proj(x)
        # x = self.proj1(x)
        # x = self.proj2(x)
        x = self.proj_drop(x)

        out = x

        # qkv = self.to_qkv(x).chunk(3, dim=-1)
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = head_num), qkv)
        #
        # dots = torch.einsum('bhid, bhjd->bhij', q, k) * self.scale
        #
        # if mask is not None:
        #     mask = F.pad(mask.flatten(1), (1, 0), value = True)
        #     assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
        #     mask = mask[:, None, :] * mask[:, :, None]
        #     dots.masked_fill_(~mask, float('-inf'))
        #     del mask

        # attn = dots.softmax(dim=-1)
        #
        # out = torch.einsum('bhij,bhjd->bhid', attn, v)
        # out = rearrange(out, 'b h n d -> b n (h d)')
        # out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout, image_size, patch_size, kernel_size, batch_size, in_chans=3, patch_stride=2,
                 patch_padding=1, norm_layer=nn.LayerNorm):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, image_size, patch_size, dropout = dropout))
            ]))
        self.patch_h = image_size // patch_size
        self.patch_w = image_size // patch_size

    def forward(self, x, mask = None):
        B, N, C = x.size()
        h, w = self.patch_h, self.patch_w

        for attn, ff in self.layers:
            x = attn(x, h, w, mask=mask) + x
            x = ff(x) + x

        # x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)     # used when two transformers are used
        return x


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, kernel_size, batch_size, num_classes, dim, depth, heads, mlp_dim, patch_stride, dropout=0., emb_dropout=0., expansion_factor=1):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        pantchesalow = image_size // patch_size
        num_patches = pantchesalow ** 2

        channels = 3
        patch_dim = channels * patch_size ** 2
        # assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective. try decreasing your patch size'

        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout, image_size, patch_size, kernel_size, batch_size, patch_stride=patch_stride, patch_padding=1)

        self.to_cls_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim*expansion_factor),
            nn.Linear(dim*expansion_factor, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes)
        )

    def set_pretraining_mode(self, is_pretraining):
        """设置所有注意力层为预训练模式"""
        for layer in self.transformer.layers:
            attn_layer = layer[0].fn  # 获取注意力层
            if hasattr(attn_layer, 'set_pretraining_mode'):
                attn_layer.set_pretraining_mode(is_pretraining)

    def forward(self, img, grad=None, mask=None):
        p = self.patch_size
        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x += self.pos_embedding[:, :(n + 1)]

        x = self.dropout(x)
        x = self.transformer(x, mask)

        cls_token_out = x[:, 0, :]  # 提取 cls_token 的输出（第 0 个位置）
        x = self.to_cls_token(cls_token_out)  # 替换原来的 x.mean(dim=1)
        classifier_result = self.mlp_head(x)

        return x, classifier_result

