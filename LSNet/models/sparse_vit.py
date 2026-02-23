import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from functools import partial
from itertools import repeat
from collections import OrderedDict
from einops.layers.torch import Rearrange
import math
# 注释无用/缺失依赖，避免报错
# from pygcn.models import GCN
# import numpy as np
# from models import *
# from torch._six import container_abcs
from models.gen_grad import GenTransformer
from torch import Tensor
from typing import Tuple

# From PyTorch internals
def _ntuple(n):
    def parse(x):
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
        return self.fn(self.norm(x), *args, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, image_size, patch_size, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# class TopkRouting(nn.Module):
#     def __init__(self, diff_routing=False):
#         super().__init__()
#         self.diff_routing = diff_routing
#         self.routing_act = nn.Softmax(dim=-1)
#
#     def forward(self, gen_adj: Tensor, topk: int) -> Tuple[Tensor]:
#         max_possible_topk = gen_adj.size(-1)
#         topk = min(topk, max_possible_topk)  # 避免索引越界
#         topk_attn_logit, topk_index = torch.topk(gen_adj, k=topk, dim=-1)
#         r_weight = self.routing_act(topk_attn_logit)
#         return r_weight, topk_index
class TopkRouting(nn.Module):
    def __init__(self, diff_routing=False):
        super().__init__()
        self.diff_routing = diff_routing
        self.routing_act = nn.Softmax(dim=-1)

    def forward(self, gen_adj: Tensor, topk: Tensor) -> Tuple[Tensor]:
        # 确保topk是形状为[N, 1]的张量
        assert topk.dim() == 2 and topk.size(1) == 1, "topk should be of shape [N, 1]"

        batch_size = gen_adj.size(0)
        max_possible_topk = gen_adj.size(-1)
        gen_adj = torch.abs(gen_adj)  # 基于幅度筛选，忽略正负方向

        # 初始化输出张量
        r_weight_list = []
        topk_index_list = []

        topknum = int(torch.max(topk).item())

        # 对批次中的每个样本单独处理
        for i in range(batch_size):
            # 获取当前样本的topk值
            current_topk = min(topk[i].item(), max_possible_topk)

            # 对当前样本执行topk操作
            topk_attn_logit, topk_index = torch.topk(gen_adj[i], k=current_topk, dim=-1)

            # 应用softmax激活函数
            current_r_weight = self.routing_act(topk_attn_logit)

            # 如果需要填充到最大可能的topk大小
            if current_topk < topknum:
                padding_size = topknum - current_topk
                current_r_weight = F.pad(current_r_weight, (0, padding_size), value=0)
                topk_index = F.pad(topk_index, (0, padding_size), value=0)

            r_weight_list.append(current_r_weight)
            topk_index_list.append(topk_index)

        # 将列表堆叠成张量
        r_weight = torch.stack(r_weight_list, dim=0)
        topk_index = torch.stack(topk_index_list, dim=0)

        return r_weight, topk_index


class KGather(nn.Module):
    def __init__(self, mul_weight='none'):
        super().__init__()
        assert mul_weight in ['none', 'soft', 'hard']
        self.mul_weight = mul_weight

    def forward(self, r_idx: Tensor, r_weight: Tensor, k: Tensor):
        n, p2, w2, c_k = k.size()
        topk = r_idx.size(-1)
        topk_k = torch.gather(
            k.view(n, 1, p2, w2, c_k).expand(-1, p2, -1, -1, -1),
            dim=2,
            index=r_idx.view(n, p2, topk, 1, 1).expand(-1, -1, -1, w2, c_k)
        )
        if self.mul_weight == 'soft':
            topk_k = r_weight.view(n, p2, topk, 1, 1) * topk_k
        elif self.mul_weight == 'hard':
            raise NotImplementedError('differentiable hard routing TBA')
        return topk_k


class Attention(nn.Module):
    def __init__(self,
                 dim,
                 heads=8,
                 dropout=0,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 method='dw_bn',
                 kernel_size=3,
                 stride_kv=1,
                 stride_q=1,
                 padding_kv=1,
                 padding_q=1,
                 with_cls_token=True,
                 min_topk=1,  # 实际topk的最小整数（至少1，否则无路由意义）
                 max_topk=64  # 实际topk的最大整数（建议设为patch总数，如32x32/patch4x4=64）
                 ):
        super().__init__()
        self.scale = dim ** -0.5
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.dim = dim
        self.num_heads = heads
        self.with_cls_token = with_cls_token

        # -------------------------- 核心修改1：topk_param范围约束与梯度保障 --------------------------
        # 1. 初始值设为0.5（在[0,1]中间，给更新留空间）
        # 2. requires_grad=True确保可更新
        self.topk_param = nn.Linear(dim, 1)
        self.non_negative_act = nn.Sigmoid()  # ReLU激活：输出 ≥ 0

        # 实际topk的上下限（根据patch总数设定，如64=8x8 patch）
        self.min_topk = min_topk
        self.max_topk = max_topk

        # 其他模块初始化（不变）
        self.router = TopkRouting(diff_routing=False)
        self.k_gather = KGather(mul_weight='none')
        self.attn_act = nn.Softmax(dim=-1)
        dim_in = dim
        dim_out = dim
        self.conv_proj_q = self._build_projection(
            dim_in, dim_out, kernel_size, padding_q, stride_q,
            'linear' if method == 'avg' else method
        )
        self.conv_proj_k = self._build_projection(
            dim_in, dim_out, kernel_size, padding_kv, stride_kv, method
        )
        self.conv_proj_v = self._build_projection(
            dim_in, dim_out, kernel_size, padding_kv, stride_kv, method
        )
        self.proj_q = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_k = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_v = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_out, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)
        # 定义二维卷积层（可根据需求调整卷积核大小、输出通道数等）
        # self.conv_layer2 = nn.Conv2d(
        #     in_channels=dim//heads*2,  # 输入通道数=特征维度16
        #     out_channels=dim//heads*2,  # 输出通道数保持与输入一致
        #     kernel_size=3,  # 3×3卷积核
        #     stride=2,
        #     padding=1  # 保持8×8尺寸不变
        # ).to(device)

        self.conv_layer = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(
            in_channels=dim*2,  # 输入通道数=特征维度16
            out_channels=dim*2,  # 输出通道数保持与输入一致
            kernel_size=3,  # 3×3卷积核
            stride=2,
            padding=1  # 保持8×8尺寸不变
            )),
            ('bn', nn.BatchNorm2d(dim*2)),
        ]))

    def _build_projection(self, dim_in, dim_out, kernel_size, padding, stride, method):
        if method == 'dw_bn':
            return nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(
                    dim_in, dim_in, kernel_size=kernel_size, padding=padding,
                    stride=stride, bias=False, groups=dim_in
                )),
                ('bn', nn.BatchNorm2d(dim_in)),
                ('rearrage', Rearrange('b c h w -> b (h w) c')),
            ]))
        elif method == 'avg':
            return nn.Sequential(OrderedDict([
                ('avg', nn.AvgPool2d(
                    kernel_size=kernel_size, padding=padding,
                    stride=stride, ceil_mode=True
                )),
                ('rearrage', Rearrange('b c h w -> b (h w) c')),
            ]))
        elif method == 'linear':
            return None
        else:
            raise ValueError('Unknown method ({})'.format(method))

    def forward_conv(self, x, h, w):
        if self.with_cls_token:
            cls_token, x = torch.split(x, [1, h * w], 1)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        q = self.conv_proj_q(x) if self.conv_proj_q is not None else rearrange(x, 'b c h w -> b (h w) c')
        k = self.conv_proj_k(x) if self.conv_proj_k is not None else rearrange(x, 'b c h w -> b (h w) c')
        v = self.conv_proj_v(x) if self.conv_proj_v is not None else rearrange(x, 'b c h w -> b (h w) c')
        if self.with_cls_token:
            q = torch.cat((cls_token, q), dim=1)
            k = torch.cat((cls_token, k), dim=1)
            v = torch.cat((cls_token, v), dim=1)
        return q, k, v

    def forward(self, x, h, w, attn_score_grad, mask=None):
        batch_size, _, _ = x.size()

        # 原有卷积投影逻辑（不变）
        if (self.conv_proj_q is not None or self.conv_proj_k is not None or self.conv_proj_v is not None):
            q, k, v = self.forward_conv(x, h, w)

        # 原有QKV线性投影和维度重排（不变）
        q = rearrange(self.proj_q(q), 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(self.proj_k(k), 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(self.proj_v(v), 'b t (h d) -> b h t d', h=self.num_heads)
        kv = torch.cat((k, v), dim=-1)
        length = kv.size(2)  # 第2维是序列长度维度
        kv = kv.unsqueeze(3)
        kv_repeated = kv.repeat(1, 1, 1, length, 1)

        # -------------------------- 修改：归一化 attn_score_grad（排除最后一维第一个元素） --------------------------
        # attn_score_grad = torch.abs(attn_score_grad)
        # # 分离最后一维的第一个元素和剩余元素
        # first_element = attn_score_grad[..., 0:1]  # 保留最后一维的第一个元素，形状保持为 (..., 1)
        # remaining_elements = attn_score_grad[..., 1:]  # 最后一维从第二个元素开始的所有元素
        # # 对剩余元素进行层归一化
        # normalized_shape = (remaining_elements.size(-1),)  # 剩余元素最后一维的大小
        # remaining_elements_norm = F.layer_norm(
        #     remaining_elements,
        #     normalized_shape=normalized_shape,
        #     eps=1e-5  # 数值稳定项
        # )
        # attn_score_grad_norm = torch.cat([first_element, remaining_elements_norm], dim=-1)

        first_element = attn_score_grad[..., 0:1]  # 不参与归一化的第一个元素，保持形状 (..., 1)
        remaining_elements = attn_score_grad[..., 1:]  # 需要归一化的剩余元素

        # -------------------------- 修改：归一化 attn_score_grad  -------
        # eps = 1e-8
        # mean = remaining_elements.mean(dim=-1, keepdim=True)
        # std = remaining_elements.std(dim=-1, keepdim=True)
        # remaining_elements = (remaining_elements - mean) / (std + eps)
        # 双向 softmax
        temperature = 0.5  # 可根据需求调整（如0.5/2.0）
        pos_prob = F.softmax(remaining_elements / temperature, dim=-1)  # 正向梯度的关注概率
        neg_prob = F.softmax(-remaining_elements / temperature, dim=-1)  # 负向梯度的抑制概率（反向softmax）
        # 融合概率（α=0.7 表示更侧重正向关注）
        alpha = 0.7
        remaining_elements_norm = alpha * pos_prob + (1 - alpha) * (1 - neg_prob)
        # remaining_elements_norm = pos_prob *  (1 - neg_prob)
        attn_score_grad_norm = torch.cat([first_element, remaining_elements_norm], dim=-1)

        # -------------------------- 原有加权逻辑（数值已稳定） --------------------------
        attn_score_grad_expanded = attn_score_grad_norm.unsqueeze(-1)  # 扩展为 (b, h_heads, t_seq, t_seq, 1)
        weighted_kv = attn_score_grad_expanded * kv_repeated  # 现在范围≈[-3, 3]，避免过小

        batch_size, num_heads, length1, length2, dim = weighted_kv.shape

        # 1. 将倒数第二维reshape为1+8×8（1个cls token + 8×8特征图）
        # 拆分cls token和8×8特征图（从倒数第二维拆分）
        cls_token = weighted_kv[..., 0:1, :]  # 提取cls token，形状: [32, 8, 65, 1, 16]
        features_64 = weighted_kv[..., 1:, :]  # 提取剩余64个元素，形状: [32, 8, 65, 64, 16]

        # 将64个元素reshape为8×8特征图
        features_8x8 = features_64.reshape(
            batch_size, length1, h, w, dim*num_heads  # 形状: [32, 8, 65, 8, 8, 16]
        )

        # 2. 在8×8特征图上执行二维卷积
        # 调整维度顺序以适应卷积输入格式 [batch, channels, height, width]
        conv_input = features_8x8.permute(0, 1, 4, 2, 3)  # 形状: [32, 65, 16*8, 8, 8]
        conv_input = conv_input.reshape(-1, dim*num_heads, h, w)  # 合并batch相关维度，形状: [32×8×65, 16, 8, 8]

        # 执行卷积操作
        conv_output = self.conv_layer(conv_input)  # 形状: [32×8×65, 16, 8, 8]

        # 恢复原始维度结构
        conv_output = conv_output.view(
            batch_size, length1, num_heads, dim, h//2, w//2  # 形状: [32, 8, 65, 16, 8, 8]
        )
        conv_output = conv_output.permute(0, 2, 1, 3, 4, 5)  # 形状: [32, 8, 65, 8, 8, 16]

        # 3. 将卷积结果展平为一维序列（8×8→64）
        conv_flat = conv_output.reshape(
            batch_size, num_heads, length1, h//2*w//2, dim  # 形状: [32, 8, 65, 64, 16]
        )

        # 4. 与cls token拼接（在倒数第二维拼接，恢复为65长度）
        kv_pix_sel = torch.cat(
            [cls_token, conv_flat],  # cls_token(1) + 卷积结果(64)
            dim=-2  # 在倒数第二维拼接
        )  # 最终形状: [32, 8, 65, 65, 16]
        kv_pix_sel = kv_pix_sel.view(-1, *kv_pix_sel.shape[2:]).unsqueeze(dim=-2)



        # # 1. Sigmoid激活：将topk_param强制限制在[0, 1]
        # topk_param_norm = self.topk_param(x).mean(1)
        # topk_param_norm = self.non_negative_act(topk_param_norm)  # 激活后确保非负（≥ 0）
        # # 2. 映射到实际topk范围：[min_topk, max_topk]
        # topk_float = self.min_topk + (self.max_topk - self.min_topk) * topk_param_norm
        # # 3. 整数转换：用floor()得到整数topk_int
        # topk_int = torch.floor(topk_float).long()
        # # 安全兜底：确保topk_int不小于min_topk
        # topk_int = torch.max(topk_int, torch.tensor(self.min_topk, device=topk_int.device))
        #
        # # 直通估计器：前向传播使用topk_int，反向传播梯度流向topk_float
        # # 使用更明确的直通估计器实现
        # topk = topk_float + (topk_int.float() - topk_float).detach()
        # self.topk_num = topk_int  # 保存整数用于监控
        #
        # # 原有KV收集和注意力计算
        # r_weight, r_idx = self.router(attn_score_grad, topk_int)
        #
        # # 确保r_weight依赖于topk，从而保留梯度流
        # # 使用更直接的方式确保梯度流
        # r_weight = r_weight  # 这不会改变数值，但会创建依赖关系
        #
        # kv_pix = rearrange(kv, 'n h t c -> (n h) t c').unsqueeze(2)
        # r_weight, r_idx = rearrange(r_weight, 'n h t c -> (n h) t c'), rearrange(r_idx, 'n h t c -> (n h) t c')
        # kv_pix_sel = self.k_gather(r_idx=r_idx, r_weight=r_weight, k=kv_pix)  #(256, 65, 30 16)

        k_pix_sel, v_pix_sel = kv_pix_sel.split([self.dim // self.num_heads, self.dim // self.num_heads], dim=-1)
        k_pix_sel = rearrange(k_pix_sel.squeeze(), '(n h) t k c -> (n t) h c k', h=self.num_heads)
        v_pix_sel = rearrange(v_pix_sel.squeeze(), '(n h) t k c -> (n t) h k c', h=self.num_heads)
        q2 = rearrange(q, 'n h t c -> (n t) h c').unsqueeze(2)

        attn_weight = (q2 * self.scale) @ k_pix_sel
        attn_weight = self.attn_act(attn_weight)
        v = attn_weight @ v_pix_sel
        v = rearrange(v.squeeze(), '(n t) h k -> n h t k', n=batch_size)
        v = F.gelu(rearrange(v, 'b h t d -> b t (h d)', h=self.num_heads))

        return v


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout, image_size, patch_size, kernel_size, batch_size, in_chans=3,
                 patch_stride=2,
                 patch_padding=1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.layers = nn.ModuleList([])
        # 计算patch总数（用于设定Attention的max_topk）
        patch_num = (image_size // patch_size) ** 2  # 不含cls_token的patch数
        for _ in range(depth):
            # 传入max_topk=patch_num（确保topk不超过patch总数）
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(
                    dim=dim, heads=heads, dropout=dropout,
                    max_topk=patch_num  # 关键：按实际patch数设定topk上限
                ))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, image_size, patch_size, dropout=dropout)))
            ]))
        self.patch_h = image_size // patch_size
        self.patch_w = image_size // patch_size

    def forward(self, x, gen_trans, mask=None):
        h, w = self.patch_h, self.patch_w
        self.first_attn_score_grad = None
        for i, (attn, ff) in enumerate(self.layers):
            if i == 0:
                self.first_attn_score_grad = gen_trans(x[:, 1:, :])
                current_attn_grad = self.first_attn_score_grad
            else:
                current_attn_grad = self.first_attn_score_grad
            x = attn(x, h, w, current_attn_grad, mask=mask)
            x = ff(x)
        return x


class SparseViT(nn.Module):
    def __init__(self, *, image_size, patch_size, kernel_size, batch_size, num_classes, dim, depth, heads, mlp_dim,
                 patch_stride, dropout=0., emb_dropout=0., expansion_factor=1):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        pantchesalow = image_size // patch_size
        num_patches = pantchesalow ** 2
        channels = 3
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # 初始化Transformer（不变）
        self.transformer = Transformer(
            dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout,
            image_size=image_size, patch_size=patch_size, kernel_size=kernel_size, batch_size=batch_size,
            patch_stride=patch_stride, patch_padding=1
        )

        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim * expansion_factor),
            nn.Linear(dim * expansion_factor, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes)
        )
        # 初始化GenTransformer（不变）
        self.gen_trans = GenTransformer(
            dim=dim, depth=depth//2, heads=heads, mlp_dim=dim, dropout=dropout,
            image_size=image_size, patch_size=patch_size
        ).to(device)

    def forward(self, img, aa=None, mask=None):
        p = self.patch_size
        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x, self.gen_trans, mask)

        cls_token_out = x[:, 0, :]
        x = self.to_cls_token(cls_token_out)
        classifier_result = self.mlp_head(x)

        return x, classifier_result

