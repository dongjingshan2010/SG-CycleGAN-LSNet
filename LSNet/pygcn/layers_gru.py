import math
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.init as init
# from models import resnet4GCN
from collections import OrderedDict
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        # self.weight4coeff = Parameter(torch.FloatTensor(head_num, in_features, 1)).to(device)
        self.hop_num = 5   # 邻接跳跃的层数， 邻接矩阵最大几次方
        # self.bilinear = nn.Bilinear(in_features, in_features, self.hop_num)
        # self.pool = nn.AdaptiveMaxPool2d((self.len//2, self.len//2))
        self.gru_seq = nn.GRU(int(in_features*head_num), int(in_features*head_num), 1)  #, nonlinearity='relu'

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
        # self.weight4coeff.data.uniform_(-stdv, stdv)
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

        support = torch.matmul(input, self.weight)

        norm_adjs = []
        norm_adj = self.norm(adj)
        norm_adjs.append(norm_adj)
        norm_adj_temp = norm_adj
        for _ in range(len-1):
            norm_adj_temp = torch.matmul(norm_adj_temp, norm_adj)
            norm_adjs.append(self.norm(norm_adj_temp))

        output = []
        for i in range(len):
            temp = torch.matmul(norm_adjs[i], support)  # spmm
            temp = rearrange(temp, 'b h l d->(b l) (h d)')
            output.append(temp)
        output = torch.stack(output, dim=0)

        # output = torch.flip(output, dims=[0])

        output, hw = self.gru_seq(output)

        final = rearrange(hw.squeeze(), '(b l) (h d)->b h l d', b=b, h=h)

        if self.bias is not None:
            final = final + self.bias

        return final

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(in_features) + ' -> ' \
               + str(out_features) + ')'


class ChebConv(Module):  #
    """
    The ChebNet convolution operation.

    :param in_c: int, number of input channels.
    :param out_c: int, number of output channels.
    :param K: int, the order of Chebyshev Polynomial.
    """
    #self, in_features, head_num, out_features, image_size, patch_size, stride=1, padding=1, kernel_size=3, bias=True
    def __init__(self, in_c, head_num, out_c, K=1, bias=True, normalize=False):
        super(ChebConv, self).__init__()
        self.normalize = normalize
        self.head_num = head_num

        self.weight = nn.Parameter(torch.Tensor(K + 1, 1, in_c//head_num, out_c//head_num))  # [K+1, 1, in_c, out_c]
        init.xavier_normal_(self.weight)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_c//head_num))
            init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)

        self.K = K + 1

    def forward(self, inputs, graph):
        """
        :param inputs: the input data, [B, N, C]
        :param graph: the graph structure, [N, N]
        :return: convolution result, [B, N, D]
        """
        L = ChebConv.get_laplacian(graph, self.normalize)  # [N, N]
        mul_L = self.cheb_polynomial(L).unsqueeze(2)   # [K, 1, N, N]

        # inputs = rearrange(inputs, 'b h n1 n2 -> (b h) n1 n2').unsqueeze(1)
        # results = []
        # for i in range(inputs.size(0)):
        #     result = torch.matmul(mul_L[i], inputs[i])  # [K, B, N, C]
        #     results.append(result)
        # results = torch.stack(results, dim=0).squeeze().permute(1, 0, 2, 3)

        inputs = rearrange(inputs, 'b h n1 n2 -> (b h) n1 n2').unsqueeze(1).unsqueeze(1)
        results = torch.matmul(mul_L, inputs).squeeze().permute(1, 0, 2, 3)  # [K, B, N, C]

        results = torch.matmul(results, self.weight)  # [K, B, N, D]
        results = torch.sum(results, dim=0) + self.bias  # [B, N, D]
        results = F.gelu(results)
        results = rearrange(results, '(b h) n1 n2 -> b h n1 n2', h=self.head_num)
        return results

    def cheb_polynomial(self, laplacian):
        """
        Compute the Chebyshev Polynomial, according to the graph laplacian.

        :param laplacian: the graph laplacian, [N, N].
        :return: the multi order Chebyshev laplacian, [K, N, N].
        """
        N = laplacian.size(1)  # [N, N]
        bsz = laplacian.size(0)
        multi_order_laplacian = torch.zeros([bsz, self.K, N, N], device=laplacian.device, dtype=torch.float, requires_grad=False)  # [K, N, N]
        multi_order_laplacian[:, 0] = torch.eye(N, device=laplacian.device, dtype=torch.float).unsqueeze(0).expand(bsz, -1, -1)

        if self.K == 1:
            return multi_order_laplacian
        else:
            multi_order_laplacian[:, 1] = laplacian
            if self.K == 2:
                return multi_order_laplacian
            else:
                for k in range(2, self.K):
                    multi_order_laplacian[:, 2, :, :] = 2 * torch.bmm(laplacian, multi_order_laplacian[:, k-1, :, :].clone()) - \
                                                        multi_order_laplacian[:, k-2, :, :].clone()

        return multi_order_laplacian

    @staticmethod
    def get_laplacian(graphp, normalize):
        """
        return the laplacian of the graph.

        :param graph: the graph structure without self loop, [N, N].
        :param normalize: whether to used the normalized laplacian.
        :return: graph laplacian.
        """
        graph = rearrange(graphp, 'b h n1 n2 -> (b h) n1 n2')
        if normalize:
            D = torch.diag_embed(torch.sum(graph, dim=-1) ** (-1 / 2))
            temp = torch.matmul(torch.matmul(D, graph), D)
            L = torch.eye(graph.size(1), device=graph.device, dtype=graph.dtype) - temp
        else:
            D = torch.diag_embed(torch.sum(graph, dim=-1))
            L = D - graph
        return L
