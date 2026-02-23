import torch.nn as nn
import torch.nn.functional as F
from DRconv import DRconv2d
import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from collections import OrderedDict


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features),
                      nn.ReLU(inplace=True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, 64, 7),
                 nn.InstanceNorm2d(64),
                 nn.ReLU(inplace=True)]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(64, output_nc, 7),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

import torch.nn.functional as F


class Domain_Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Domain_Discriminator, self).__init__()

        # 保持前面的卷积特征提取部分不变
        model = [nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=False)]

        model += [nn.Conv2d(64, 128, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(128),
                  nn.LeakyReLU(0.2, inplace=False)]

        model += [nn.Conv2d(128, 256, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(256),
                  nn.LeakyReLU(0.2, inplace=False)]

        model += [nn.Conv2d(256, 512, 4, padding=1),
                  nn.InstanceNorm2d(512),
                  nn.LeakyReLU(0.2, inplace=False)]

        model += [nn.Conv2d(512, 256, 4, padding=1),
                  nn.InstanceNorm2d(256),
                  nn.LeakyReLU(0.2, inplace=False)]

        # 二分类修改：最后一层卷积输出通道数改为2（对应两个类别）
        model += [nn.Conv2d(256, 2, 4, padding=1)]  # 原输出通道数为1，现在改为2

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)  # 输出形状：[batch_size, 2, H, W]（H和W为最后一层卷积的空间尺寸）
        # 对空间维度做平均池化，压缩为[batch_size, 2, 1, 1]，再挤压掉冗余维度得到[batch_size, 2]
        x = F.avg_pool2d(x, x.size()[2:])  # 对H和W维度平均池化
        return x.view(x.size(0), -1)  # 最终形状：[batch_size, 2]（二分类logits）

class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(64, 128, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(128),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(128, 256, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(256),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(256, 512, 4, padding=1),
                  nn.InstanceNorm2d(512),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(512, 256, 4, padding=1),
                  nn.InstanceNorm2d(256),
                  nn.LeakyReLU(0.2, inplace=True)]

        # FCN classification layer
        model += [nn.Conv2d(256, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


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
    def __init__(self, dim, hidden_dim, image_size, patch_size, dropout=0.):
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

        # self.proj_q = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        # self.proj_k = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        # self.proj_v = nn.Linear(dim_in, dim_out, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_out, dim_out)
        # self.proj1 = nn.Linear(dim_out, 64)
        # self.proj2 = nn.Linear(64, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

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
            cls_token, x = torch.split(x, [1, h * w], 1)

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
        # head_num = self.heads

        if (
                self.conv_proj_q is not None
                or self.conv_proj_k is not None
                or self.conv_proj_v is not None
        ):
            q, k, v = self.forward_conv(x, h, w)

        # q = rearrange(self.proj_q(q), 'b t (h d) -> b h t d', h=self.num_heads)
        # k = rearrange(self.proj_k(k), 'b t (h d) -> b h t d', h=self.num_heads)
        # v = rearrange(self.proj_v(v), 'b t (h d) -> b h t d', h=self.num_heads)
        #
        q = rearrange((q), 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange((k), 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange((v), 'b t (h d) -> b h t d', h=self.num_heads)

        attn_score = torch.einsum('bhlk,bhtk->bhlt', [q, k]) * self.scale
        attn = F.softmax(attn_score, dim=-1)
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
    def __init__(self, dim, depth, heads, mlp_dim, dropout, image_size, patch_size):
        super().__init__()

        # self.patch_embed = ConvEmbed(
        #     image_size=image_size,
        #     patch_size=patch_size,
        #     kernel_size=kernel_size,
        #     batch_size=batch_size,
        #     in_chans=in_chans,
        #     stride=patch_stride,
        #     padding=patch_padding,
        #     embed_dim=dim,
        #     norm_layer=norm_layer
        # )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, image_size, patch_size, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        # x = self.patch_embed(x)
        # B, C, H, W = x.size()
        # x = rearrange(x, 'b c h w -> b (h w) c')

        B, hw, C = x.size()
        H = int(hw ** 0.5)
        W = H
        for attn, ff in self.layers:
            x = attn(x, H, W, mask=mask)
            x = ff(x)

        # x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)     # used when two transformers are used
        return x


class ViTToImageDecoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, out_channels=3):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.num_patches = (img_size // patch_size) ** 2

        # 从Transformer特征维度恢复到patch维度
        self.proj = nn.Linear(embed_dim, patch_size * patch_size * out_channels)

        # 可选的上采样层，如果需要更高分辨率
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        )

    def forward(self, x):
        # 假设输入x形状为 [batch_size, num_patches + 1, embed_dim]
        # 移除class token
        x = x[:, 1:]  # [batch_size, num_patches, embed_dim]

        # 投影到patch特征
        x = self.proj(x)  # [batch_size, num_patches, patch_size*patch_size*out_channels]

        # 重塑为patch网格
        h_w = int(self.num_patches ** 0.5)
        x = x.reshape(x.shape[0], h_w, h_w, self.patch_size, self.patch_size, self.out_channels)

        # 重新排列为图像格式
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.reshape(x.shape[0], self.out_channels, h_w * self.patch_size, h_w * self.patch_size)

        # 可选的上采样
        if h_w * self.patch_size < self.img_size:
            x = self.upsample(x)

        return x


class SharedGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, image_size, depth4vit, n_residual_blocks=7, shared_layers=3):
        super(SharedGenerator, self).__init__()

        # 共享层
        shared_model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            # DRconv2d(in_channels=input_nc, out_channels=64, kernel_size=7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]
        # 下采样层
        in_features = 64
        out_features = in_features * 2
        for i in range(shared_layers):
            shared_model += [
                nn.Conv2d(in_features, out_features, 5, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2

        out_features = in_features // 2
        for _ in range(shared_layers):
            shared_model += [
                nn.ConvTranspose2d(in_features, out_features, 5, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2

        shared_model += [
            nn.ReflectionPad2d(0),
            nn.Conv2d(out_features * 2, output_nc, 7),
            nn.Tanh()
        ]

        self.shared_layers = nn.Sequential(*shared_model)


        # self.shared_layers = nn.Sequential(*shared_model)

        # ###########  Up -- public layers ############
        # A2B私有层
        a2b_private_model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            # DRconv2d(in_channels=input_nc, out_channels=64, kernel_size=7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        # 下采样层
        in_features = 64
        out_features = in_features * 2
        for i in range(shared_layers):
            a2b_private_model += [
                nn.Conv2d(in_features, out_features, 5, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2

        a2b_in_features = in_features

        a2b_private_model += [DRconv2d(in_channels=a2b_in_features, out_channels=a2b_in_features, kernel_size=1), ]

        # 完成下采样（如果有未共享的下采样层）
        if shared_layers <= 2:
            for i in range(2 - (shared_layers - 1)):
                a2b_private_model += [
                    nn.Conv2d(a2b_in_features, a2b_in_features * 2, 3, stride=1, padding=1),
                    nn.InstanceNorm2d(a2b_in_features * 2),
                    nn.ReLU(inplace=True)
                ]
                a2b_in_features = a2b_in_features * 2

        # A2B残差块
        for _ in range(n_residual_blocks):
            a2b_private_model += [ResidualBlock(a2b_in_features)]

        # A2B上采样
        a2b_out_features = a2b_in_features // 2
        for _ in range(shared_layers):
            a2b_private_model += [
                nn.ConvTranspose2d(a2b_in_features, a2b_out_features, 5, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(a2b_out_features),
                nn.ReLU(inplace=True)
            ]
            a2b_in_features = a2b_out_features
            a2b_out_features = a2b_in_features // 2

        # A2B输出层
        a2b_private_model += [
            # nn.ReflectionPad2d(3),
            nn.Conv2d(a2b_out_features * 2, output_nc, 7),
            nn.Tanh()
        ]

        self.a2b_private_layers = nn.Sequential(*a2b_private_model)

        # B2A私有层
        b2a_private_model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            # DRconv2d(in_channels=input_nc, out_channels=64, kernel_size=7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        # 下采样层
        in_features = 64
        out_features = in_features * 2
        for i in range(shared_layers):
            b2a_private_model += [
                nn.Conv2d(in_features, out_features, 5, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2

        b2a_in_features = in_features

        # 完成下采样（如果有未共享的下采样层）
        if shared_layers <= 2:
            for i in range(2 - (shared_layers - 1)):
                b2a_private_model += [
                    nn.Conv2d(b2a_in_features, b2a_in_features * 2, 3, stride=2, padding=1),
                    nn.InstanceNorm2d(b2a_in_features * 2),
                    nn.ReLU(inplace=True)
                ]
                b2a_in_features = b2a_in_features * 2

        # B2A残差块
        for _ in range(n_residual_blocks):
            b2a_private_model += [ResidualBlock(b2a_in_features)]

        # B2A上采样
        b2a_out_features = b2a_in_features // 2
        for _ in range(shared_layers):
            b2a_private_model += [
                nn.ConvTranspose2d(b2a_in_features, b2a_out_features, 5, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(b2a_out_features),
                nn.ReLU(inplace=True)
            ]
            b2a_in_features = b2a_out_features
            b2a_out_features = b2a_in_features // 2

        # B2A输出层
        b2a_private_model += [
            # nn.ReflectionPad2d(3),
            nn.Conv2d(b2a_out_features *2, input_nc, 7),
            nn.Tanh()
        ]

        self.b2a_private_layers = nn.Sequential(*b2a_private_model)

    def forward_a2b(self, x):
        x = self.shared_layers(x)
        private = self.a2b_private_layers(x)

        return private, x

    def forward_b2a(self, x):
        x = self.shared_layers(x)
        private = self.b2a_private_layers(x)
        return private, x

