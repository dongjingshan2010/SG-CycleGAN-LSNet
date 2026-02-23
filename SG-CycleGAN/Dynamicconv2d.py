import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.autograd import Function



class DynamicWeightGenerator(nn.Module):
    def __init__(self, in_ch, K=4, reduction=16):
        super().__init__()
        self.K = K
        self.gating = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局池化降维
            nn.Conv2d(in_ch, in_ch//reduction, 1),  # 通道压缩
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch//reduction, K, 1),  # 生成K维权重
            nn.Rearrange('b k 1 1 -> b k')  # 形状调整为[B, K]
        )
    
    def forward(self, x):
        weights = self.gating(x)
        return F.softmax(weights, dim=1)  # 归一化为概率分布

class DynamicConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, K=4, bias=True):
        super().__init__()
        self.K = K
        self.convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch, kernel_size, 
                      padding='same', bias=bias) for _ in range(K)
        ])
        self.generator = DynamicWeightGenerator(in_ch, K)
    
    def forward(self, x):
        B, C, H, W = x.shape
        weights = self.generator(x).view(B, self.K, 1, 1, 1)  # [B,K,1,1,1]
        # 动态权重与各分支卷积结果加权求和
        return sum(weight * conv(x) for weight, conv in zip(weights.unbind(1), self.convs))
