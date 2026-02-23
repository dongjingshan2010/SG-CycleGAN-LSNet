import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers_gru import GraphConvolution_gru, ChebConv


class GCN_gru(nn.Module):
    def __init__(self, nfeat, head_num, nhid, image_size, patch_size, stride, padding, kernel_size, nclass, dropout):
        super(GCN_gru, self).__init__()

        # self.gc1 = GraphConvolution_gru(nfeat, head_num, nhid, image_size, patch_size, stride, padding, kernel_size)
        self.gc1 = ChebConv(nfeat, head_num, nhid, K=5, bias=True, normalize=True)

        # self.gc2 = GraphConvolution(nfeat, head_num, nhid, image_size, patch_size, stride, padding, kernel_size)
        # self.gc3 = GraphConvolution(nfeat, nhid, image_size, patch_size, stride, padding, kernel_size)
        self.dropout = dropout
        # self.linear = nn.Linear(int(nfeat/head_num), int(nhid/head_num))

    def forward(self, x, adj):
        x = self.gc1(x, adj)  # Having GELU
        x = F.dropout(x, self.dropout, training=self.training)
        # x = F.gelu(self.gc2(x, adj))

        # x = self.linear(x)
        return x
