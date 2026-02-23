import math
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
# from models import resnet4GCN
from collections import OrderedDict
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class GraphConvolution_gru(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, head_num, out_features, image_size, patch_size, stride=1, padding=1, kernel_size=3, bias=True):
        super(GraphConvolution_gru, self).__init__()
        in_features = int(in_features/head_num)
        out_features = int(out_features/head_num)
        self.image_size = image_size
        self.patch_size = patch_size
        self.len = image_size//patch_size

        # 选择在图卷积中使用全连接层
        # (1)
        self.weight = Parameter(torch.FloatTensor(head_num, in_features, out_features)).to(device)
        self.weight4coeff = Parameter(torch.FloatTensor(head_num, in_features, 1)).to(device)
        self.hop_num = 5   # 邻接跳跃的层数， 邻接矩阵最大几次方
        self.bilinear = nn.Bilinear(in_features, in_features, self.hop_num)
        self.pool = nn.AdaptiveAvgPool2d((self.len//4, self.len//4))

        # (2)
        # self.head_num = head_num
        # self.weight = Parameter(torch.FloatTensor(int(in_features*head_num), int(out_features*head_num)))

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features)).to(device)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.weight4coeff.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def norm(self, adjcency):
        degree = torch.pow(torch.einsum('ihjk->ihj', [adjcency]), -0.5)

        degree_diag = torch.diag_embed(degree.squeeze())

        norm_adj = degree_diag.matmul(adjcency).matmul(degree_diag).to(device)
        return norm_adj  # norm_adj

    def forward(self, input, adj):

        b, h, len2, d = input.size()
        len = self.hop_num
        input4bilinear = rearrange(input, 'b head (h w) d->(b head) d h w', h=self.len)
        input4bilinear = self.pool(input4bilinear)
        input4bilinear = rearrange(input4bilinear, '(b head) d h w-> (h w) b head d', b=b)


        # coeff = F.softmax(self.bilinear(input4bilinear, input4bilinear), dim=-1).unsqueeze(dim=-1)
        # coeff = torch.mean(coeff, dim=0)

        coeff = torch.mean(self.bilinear(input4bilinear, input4bilinear), dim=0)
        coeff = F.softmax(coeff, dim=-1).unsqueeze(dim=-1)


        coeff = rearrange(coeff, 'b h n m->(b h) m n')
        support = torch.matmul(input, self.weight)

        # (1)
        norm_adjs = []
        norm_adj = self.norm(adj)
        norm_adjs.append(norm_adj)
        norm_adj_temp = norm_adj
        for _ in range(len-1):
            norm_adj_temp = torch.matmul(norm_adj_temp, norm_adj)
            norm_adjs.append(self.norm(norm_adj_temp))

        norm_adjs = torch.stack(norm_adjs, dim=0)
        norm_adjs = norm_adjs.permute(1, 2, 0, 3, 4)
        norm_adjs = rearrange(norm_adjs, 'b h n h1 w1 -> (b h) n (h1 w1)')

        norm_adj = torch.einsum('bmn,bnk->bmk', [coeff, norm_adjs])
        norm_adj = rearrange(norm_adj, '(b h)m (h1 w1) -> b h m h1 w1', b=b, h1=len2).squeeze()

        output = torch.matmul(norm_adj, support)  # spmm
        if self.bias is not None:
            output = output + self.bias

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(in_features) + ' -> ' \
               + str(out_features) + ')'
