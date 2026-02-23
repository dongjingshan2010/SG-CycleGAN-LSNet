import torch
import torch.nn as nn
from models.gen_grad import GenTransformer
from einops import rearrange
import torch.nn.functional as F
from collections import OrderedDict

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def Conv_BN_ReLU(inp, oup, kernel, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=kernel, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn  # fn直接是Attention实例（无额外包装）

    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)


class PreNormCross(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn  # fn直接是Attention实例（无额外包装）

    def forward(self, x, x_mr, grad, *args, **kwargs):
        return self.fn(self.norm(x), self.norm(x_mr), grad, *args, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        # self.ffn = nn.Sequential(
        #     nn.Linear(dim, hidden_dim),
        #     nn.SiLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_dim, dim),
        #     nn.Dropout(dropout)
        # )
        self.conv_layer = nn.Sequential(OrderedDict([
            ('depthwise', nn.Conv2d(
                in_channels=dim,
                out_channels=dim,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=dim  # 深度可分离卷积
            )),
            ('pointwise', nn.Conv2d(
                in_channels=dim,
                out_channels=dim,
                kernel_size=1
            )),
            ('bn', nn.BatchNorm2d(dim)),
        ]))

    def forward(self, x):
        b, p, L, d = x.shape
        w = int(L ** 0.5)
        features_8x8 = x.reshape(
            b * p, w, w, d
        )
        conv_input = features_8x8.permute(0, 3, 1, 2)
        x = self.conv_layer(conv_input)
        x = x.reshape(b, p, w*w, d)
        return x
        # return self.ffn(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        # conv_dim = dim_head * 2
        project_out = not (heads == 1 and dim_head == dim)
        self.is_pretraining = False  # 预训练模式标志
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        # #使用深度可分离卷积进一步减少FLOPs
        # self.conv_layer = nn.Sequential(OrderedDict([
        #     ('depthwise', nn.Conv2d(
        #         in_channels=conv_dim,
        #         out_channels=conv_dim,
        #         kernel_size=5,
        #         stride=4,
        #         padding=0,
        #         groups=conv_dim  # 深度可分离卷积
        #     )),
        #     ('pointwise', nn.Conv2d(
        #         in_channels=conv_dim,
        #         out_channels=conv_dim,
        #         kernel_size=1
        #     )),
        #     ('bn', nn.BatchNorm2d(conv_dim)),
        # ]))

        # self.conv_layer = nn.Sequential(OrderedDict([
        #     ('depthwise', nn.Conv2d(
        #         in_channels=conv_dim,
        #         out_channels=conv_dim // 2,  # 输出通道数减半
        #         kernel_size=5,
        #         stride=4,
        #         padding=0,
        #         groups=conv_dim // 2  # 深度卷积的groups需与输出通道数一致
        #     )),
        #     ('pointwise', nn.Conv2d(
        #         in_channels=conv_dim // 2,  # 输入来自depthwise的输出
        #         out_channels=conv_dim // 2,  # 保持减半的通道数
        #         kernel_size=1
        #     )),
        #     ('bn', nn.BatchNorm2d(conv_dim // 2)),  # 批量归一化适配减半的通道数
        #     ('linear_up', nn.Conv2d(  # 1x1卷积作为线性层升维
        #         in_channels=conv_dim // 2,
        #         out_channels=conv_dim,  # 升维回原始conv_dim
        #         kernel_size=1
        #     ))
        # ]))

        self.attn_act = nn.Softmax(dim=-1)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def set_pretraining_mode(self, is_pretraining):
        """设置预训练模式标志（核心方法）"""
        self.is_pretraining = is_pretraining

    # using Top k selection to reduce the token numbers of KV
    def forward(self, x, grad):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)

        kv = torch.cat((k, v), dim=-1)
        b, p, h, length, dim = kv.shape  # 注意第4维也是d，用_忽略
        w = int(length ** 0.5)

        kv = kv.unsqueeze(4)
        kv_repeated = kv.repeat(1, 1, 1, 1, length, 1)
        temperature = 0.5  # 可根据需求调整（如0.5/2.0）
        pos_prob = F.softmax(grad / temperature, dim=-1)  # 正向梯度的关注概率
        neg_prob = F.softmax(-grad / temperature, dim=-1)  # 负向梯度的抑制概率（反向softmax）
        # 融合概率（α=0.7 表示更侧重正向关注）
        alpha = 0.7
        grad_norm = alpha * pos_prob + (1 - alpha) * (1 - neg_prob)

        # 假设grad_norm形状为[3,4,4,400,400,1]，kv_repeated形状为[3,4,4,400,400,16]
        # 目标：对每个400×400矩阵的每行（第4维）选取最大的24个位置，提取对应kv值

        topk = w * w // 25 #int(w * w) 8;
        # 2. 在每行（第4维的每个元素对应的最后一维）选取top24最大值的索引
        # dim=4表示在400列的维度上选取
        topk_values, topk_indices = torch.topk(grad_norm, k=topk, dim=4, largest=True)
        # topk_indices形状: [3,4,4,400,24]

        topk_indices = topk_indices.unsqueeze(-1)  # 形状: [3,4,4,400,24,1]

        kv_pix_sel = torch.gather(kv_repeated, dim=4, index=topk_indices.expand(-1, -1, -1, -1, -1, dim))

        kv_pix_sel = kv_pix_sel.reshape(
            b*h, p, length, topk, dim  # 形状: [32, 8, 65, 64, 16]
        )

        kv_pix_sel = kv_pix_sel.reshape(-1, *kv_pix_sel.shape[2:]).unsqueeze(dim=-2)

        k_pix_sel, v_pix_sel = kv_pix_sel.split([dim // 2, dim // 2], dim=-1)
        k_pix_sel = rearrange(k_pix_sel.squeeze(3), '(n p) t k c -> (n t) p c k', p=p)
        v_pix_sel = rearrange(v_pix_sel.squeeze(3), '(n p) t k c -> (n t) p k c', p=p)

        q = q.reshape(b*h, p, length, dim//2)
        q2 = rearrange(q, 'n h t c -> (n t) h c').unsqueeze(2)

        attn_weight = (q2 * self.scale) @ k_pix_sel
        attn_weight = self.attn_act(attn_weight)

        v = attn_weight @ v_pix_sel
        v = F.gelu(rearrange(v.squeeze(), '(n t) h k -> n h t k', n=b*h))

        out = rearrange(v, '(b h) p n d -> b p n (h d)', b=b, h=h)
        return self.to_out(out)

    # using conv to reduce the token numbers of KV
    # def forward(self, x, grad):
    #     # 计算并分解qkv
    #     qkv = self.to_qkv(x).chunk(3, dim=-1)
    #     q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)
    #
    #     # 合并k和v，获取维度信息
    #     kv = torch.cat((k, v), dim=-1)
    #     b, p, h, length, dim = kv.shape  # 维度: [b, p, h, length, dim]
    #
    #     # 计算梯度概率（修正维度匹配问题）
    #     temperature = 0.5
    #     alpha = 0.7
    #     pos_prob = F.softmax(grad / temperature, dim=-1)  # 形状: [b, p, h, length, length]
    #     neg_prob = F.softmax(-grad / temperature, dim=-1)
    #     grad_norm = alpha * pos_prob + (1 - alpha) * (1 - neg_prob)  # 形状不变
    #
    #     # 关键修正：只增加最后一个维度用于广播，保持第4维为length
    #     # grad_norm形状变为: [b, p, h, length, length, 1]
    #     grad_norm = grad_norm.unsqueeze(-1)
    #
    #     # kv增加第4维（长度维度）用于广播，形状变为: [b, p, h, length, 1, dim]
    #     # 此时grad_norm的length维度(第4维)与kv的length维度(第3维)匹配，可正确广播
    #     weighted_kv = grad_norm * kv.unsqueeze(4)  # 结果形状: [b, p, h, length, length, dim]
    #
    #     # 后续操作保持不变
    #     w = int(length ** 0.5)
    #     w4 = w // 5
    #
    #     features_8x8 = weighted_kv.reshape(
    #         b * h, length, w, w, dim * p
    #     )
    #
    #     conv_input = features_8x8.permute(0, 1, 4, 2, 3)
    #     conv_input = conv_input.reshape(-1, dim, w, w)
    #
    #     conv_output = self.conv_layer(conv_input)
    #
    #     conv_output = conv_output.view(
    #         b * h, length, p, dim, w4, w4
    #     ).permute(0, 2, 1, 3, 4, 5)
    #
    #     kv_pix_sel = conv_output.reshape(
    #         b * h, p, length, w4 * w4, dim
    #     )
    #
    #     kv_pix_sel = kv_pix_sel.reshape(-1, *kv_pix_sel.shape[2:]).unsqueeze(dim=-2)
    #     k_pix_sel, v_pix_sel = kv_pix_sel.split([dim // 2, dim // 2], dim=-1)
    #
    #     k_pix_sel = rearrange(k_pix_sel.squeeze(3), '(n p) t k c -> (n t) p c k', p=p)
    #     v_pix_sel = rearrange(v_pix_sel.squeeze(3), '(n p) t k c -> (n t) p k c', p=p)
    #
    #     q = q.reshape(b * h, p, length, dim // 2)
    #     q2 = rearrange(q, 'n h t c -> (n t) h c').unsqueeze(2)
    #
    #     attn_weight = (q2 * self.scale) @ k_pix_sel
    #     attn_weight = self.attn_act(attn_weight)
    #
    #     v = attn_weight @ v_pix_sel
    #     v = F.gelu(rearrange(v.squeeze(), '(n t) h k -> n h t k', n=b * h))
    #
    #     out = rearrange(v, '(b h) p n d -> b p n (h d)', b=b, h=h)
    #     return self.to_out(out)

    # def forward(self, x, grad=None):
    #     qkv = self.to_qkv(x).chunk(3, dim=-1)
    #     q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)
    #
    #     attn_score = torch.matmul(q, k.transpose(-1, -2)) * self.scale
    #
    #     current_attn_score = attn_score if not hasattr(self, 'attn_score') else self.attn_score
    #
    #     attn = self.attend(current_attn_score)
    #     out = torch.matmul(attn, v)
    #     out = rearrange(out, 'b p h n d -> b p n (h d)')
    #     return self.to_out(out)


# class CrossAttention(nn.Module):
#     def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
#         super().__init__()
#         inner_dim = dim_head * heads
#         project_out = not (heads == 1 and dim_head == dim)
#         self.is_pretraining = False  # 预训练模式标志
#         self.heads = heads
#         self.scale = dim_head ** -0.5
#
#         self.attend = nn.Softmax(dim=-1)
#         # 分别创建query, key, value的线性变换
#         self.to_q = nn.Linear(dim, inner_dim, bias=False)
#         self.to_k = nn.Linear(dim, inner_dim, bias=False)
#         self.to_v = nn.Linear(dim, inner_dim, bias=False)
#
#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, dim),
#             nn.Dropout(dropout)
#         ) if project_out else nn.Identity()
#
#     def set_pretraining_mode(self, is_pretraining):
#         """设置预训练模式标志"""
#         self.is_pretraining = is_pretraining
#
#     def forward(self, x, context, grad):
#         # x: [batch, patches, seq_len, dim]
#         # context: [batch, patches, seq_len, dim] - 交叉模态的上下文
#
#         # 分别计算query, key, value
#         q = self.to_q(x)
#         k = self.to_k(context)
#         v = self.to_v(context)
#
#         # 重排维度
#         q = rearrange(q, 'b p n (h d) -> b p h n d', h=self.heads)
#         k = rearrange(k, 'b p n (h d) -> b p h n d', h=self.heads)
#         v = rearrange(v, 'b p n (h d) -> b p h n d', h=self.heads)
#
#         kv = torch.cat((k, v), dim=-1)
#         b, p, h, length, dim = kv.shape  # 注意第4维也是d，用_忽略
#         w = int(length ** 0.5)
#
#         kv = kv.unsqueeze(4)
#         kv_repeated = kv.repeat(1, 1, 1, 1, length, 1)
#         temperature = 0.5  # 可根据需求调整（如0.5/2.0）
#         pos_prob = F.softmax(grad / temperature, dim=-1)  # 正向梯度的关注概率
#         neg_prob = F.softmax(-grad / temperature, dim=-1)  # 负向梯度的抑制概率（反向softmax）
#         # 融合概率（α=0.7 表示更侧重正向关注）
#         alpha = 0.7
#         grad_norm = alpha * pos_prob + (1 - alpha) * (1 - neg_prob)
#
#         topk = w // 5 * w // 5
#         topk_values, topk_indices = torch.topk(grad_norm, k=topk, dim=4, largest=True)
#
#         topk_indices = topk_indices.unsqueeze(-1)  # 形状: [3,4,4,400,24,1]
#
#         kv_pix_sel = torch.gather(kv_repeated, dim=4, index=topk_indices.expand(-1, -1, -1, -1, -1, dim))
#
#         kv_pix_sel = kv_pix_sel.reshape(
#             b * h, p, length, topk, dim  # 形状: [32, 8, 65, 64, 16]
#         )
#
#         kv_pix_sel = kv_pix_sel.reshape(-1, *kv_pix_sel.shape[2:]).unsqueeze(dim=-2)
#
#         k_pix_sel, v_pix_sel = kv_pix_sel.split([dim // 2, dim // 2], dim=-1)
#         k_pix_sel = rearrange(k_pix_sel.squeeze(3), '(n p) t k c -> (n t) p c k', p=p)
#         v_pix_sel = rearrange(v_pix_sel.squeeze(3), '(n p) t k c -> (n t) p k c', p=p)
#
#         q = q.reshape(b * h, p, length, dim // 2)
#         q2 = rearrange(q, 'n h t c -> (n t) h c').unsqueeze(2)
#
#         attn_weight = (q2 * self.scale) @ k_pix_sel
#         attn_weight = self.attend(attn_weight)
#
#         v = attn_weight @ v_pix_sel
#         v = F.gelu(rearrange(v.squeeze(), '(n t) h k -> n h t k', n=b * h))
#
#         out = rearrange(v, '(b h) p n d -> b p n (h d)', b=b, h=h)
#
#         # out = torch.matmul(attn_weight, v_pix_sel)
#         # out = rearrange(out, 'b p h n d -> b p n (h d)')
#         return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, att_size, dropout=0.):
        super().__init__()

        # 第一个分支：只有自注意力层
        self.branch1_layers = nn.ModuleList([])
        for _ in range(depth):
            self.branch1_layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

        self.first_attn_score_grad = None
        self.gen_grad = GenTransformer(
            dim=dim, depth=depth // 2, heads=heads, mlp_dim=dim, dropout=dropout, att_size=att_size
        ).to(device)

    def forward(self, x):
        self.first_attn_score_grad = self.gen_grad(x)
        current_attn_grad = self.first_attn_score_grad

        # 处理第一个分支（只有自注意力）
        for self_attn, ff in self.branch1_layers:
            x = self_attn(x, current_attn_grad) + x
            x = ff(x) + x

        # # 处理第二个分支（自注意力 + 可选的交叉注意力）
        # for layer_components in self.branch2_layers:
        #     if len(layer_components) == 3:  # 有交叉注意力的层
        #         self_attn, cross_attn, ff = layer_components
        #         x = self_attn(x, current_attn_grad) + x
        #
        #         x_gate = torch.cat([x, x_mr], dim=-1)
        #         x_gate = self.linear(x_gate)
        #
        #         x = x_gate * cross_attn(x, x_mr, current_attn_grad_mr) + x  # 用branch1的信息增强branch2
        #         x = ff(x) + x

        return x  # , x_mr  only one branch was returned.


class MV2Block(nn.Module):
    def __init__(self, inp, oup, stride=1, expand_ratio=4):
        super(MV2Block, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, heads, kernel_size, att_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = Conv_BN_ReLU(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)

        # 每个MobileViTBlock包含一个Transformer实例
        self.transformer = Transformer(dim, depth, heads, dim//heads, mlp_dim, att_size, dropout)

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = Conv_BN_ReLU(2 * channel, channel, kernel_size)

    def forward(self, x):
        y = x.clone()
        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)

        # Global representations（调用内部Transformer）
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)

        x = self.transformer(x)

        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)',
                      h=torch.div(h, self.ph, rounding_mode='trunc'),
                      w=torch.div(w, self.pw, rounding_mode='trunc'),
                      ph=self.ph, pw=self.pw)

        # Fusion
        x = self.conv3(x)

        x1 = torch.cat((x, y), 1)

        x1 = self.conv4(x1)

        return x1


class SparseMobileViT(nn.Module):
    def __init__(self, image_size, dims, channels, heads, num_classes, expansion=4, kernel_size=3, patch_size=(2, 2)):
        super().__init__()
        ih, iw = image_size
        ph, pw = patch_size
        att_size = ih // 16
        assert ih % ph == 0 and iw % pw == 0

        L = [2, 4, 3]

        self.conv1 = Conv_BN_ReLU(3, channels[0], kernel=3, stride=2)

        self.mv2 = nn.ModuleList([])
        self.mv2.append(MV2Block(channels[0], channels[1], 1, expansion))
        self.mv2.append(MV2Block(channels[1], channels[2], 2, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))  # Repeat
        self.mv2.append(MV2Block(channels[3], channels[4], 2, expansion))
        self.mv2.append(MV2Block(channels[5], channels[6], 2, expansion))
        self.mv2.append(MV2Block(channels[7], channels[8], 2, expansion))

        # 所有MobileViTBlock实例存储在self.mvit中
        self.mvit = nn.ModuleList([])
        self.mvit.append(MobileViTBlock(dims[0], L[0], channels[5], heads, kernel_size, att_size, patch_size, int(dims[0] * 2)))
        self.mvit.append(MobileViTBlock(dims[1], L[1], channels[7], heads, kernel_size, att_size//2, patch_size, int(dims[1] * 4)))
        self.mvit.append(MobileViTBlock(dims[2], L[2], channels[9], heads, kernel_size, att_size//4, patch_size, int(dims[2] * 4)))

        self.conv2 = conv_1x1_bn(channels[-2], channels[-1])

        self.pool = nn.AvgPool2d(ih // 32, 1)
        self.fc = nn.Linear(channels[-1], num_classes, bias=False)

    def set_pretraining_mode(self, is_pretraining):
        """
        核心修改：遍历所有MobileViTBlock，为其Transformer内的Attention层设置预训练模式
        """
        # 1. 遍历self.mvit中的每个MobileViTBlock实例
        for mvit_block in self.mvit:
            # 2. 获取当前MobileViTBlock内的Transformer
            transformer = mvit_block.transformer
            # 3. 遍历Transformer的每一层（每层含Attention和FeedForward）
            for layer in transformer.branch1_layers:
                # 4. 提取Attention层（layer[0]是PreNorm，其.fn是Attention实例）
                attn_layer = layer[0].fn
                # 5. 调用Attention的set_pretraining_mode方法
                if hasattr(attn_layer, 'set_pretraining_mode'):
                    attn_layer.set_pretraining_mode(is_pretraining)

    def forward(self, x):
        x = self.conv1(x)
        x = self.mv2[0](x)

        x = self.mv2[1](x)
        x = self.mv2[2](x)
        x = self.mv2[3](x)  # Repeat

        x = self.mv2[4](x)

        x = self.mvit[0](x)  # 第一个MobileViTBlock

        x = self.mv2[5](x)

        x = self.mvit[1](x)  # 第二个MobileViTBlock

        x = self.mv2[6](x)

        x = self.mvit[2](x)  # 第三个MobileViTBlock
        x = self.conv2(x)

        x = self.pool(x).view(-1, x.shape[1])
        classifier_result = self.fc(x)
        return x, classifier_result


def mobilevit_xxs():
    dims = [64, 80, 96]
    channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
    return SparseMobileViT((256, 256), dims, channels, num_classes=1000, expansion=2)


def mobilevit_xs():
    dims = [96, 120, 144]
    channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384]
    return SparseMobileViT((256, 256), dims, channels, num_classes=1000)


def mobilevit_s():
    dims = [144, 192, 240]
    channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
    return SparseMobileViT((256, 256), dims, channels, num_classes=1000)