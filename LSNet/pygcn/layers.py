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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, head_num, out_features, image_size, patch_size, stride=1, padding=1, kernel_size=3, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = int(in_features/head_num)
        self.out_features = int(out_features/head_num)
        self.image_size = image_size
        self.patch_size = patch_size

        # 选择在图卷积中使用全连接层
        # (1)
        self.weight = Parameter(torch.FloatTensor(head_num, self.in_features, self.out_features).to(device))
        self.weight2 = Parameter(torch.FloatTensor(head_num, self.in_features, self.out_features).to(device))
        # randomatrix = torch.randn((head_num, self.in_features, self.out_features), requires_grad=True).to(device)
        # self.weight = torch.nn.Parameter(randomatrix)
        self.register_parameter('weight', self.weight)
        self.register_parameter('weight2', self.weight2)
        # (2)
        # self.head_num = head_num
        # Parameter = Parameter(torch.FloatTensor(int(in_features*head_num), int(out_features*head_num)))

        if bias:
            self.bias = Parameter(torch.FloatTensor(self.out_features).to(device))
            self.register_parameter('bias_gcn', self.bias)
        else:
            self.bias = None
            self.register_parameter('bias_gcn', None)
        self.reset_parameters()

    def reset_parameters(self):
        # stdv = 1. / math.sqrt(self.weight.size(1))
        # self.weight.data.uniform_(-stdv, stdv)
        # stdv = 1. / math.sqrt(self.weight2.size(1))
        # self.weight2.data.uniform_(-stdv, stdv)

        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.weight2)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def norm(self, adj):
        head_num = adj.size(1)
        adj = rearrange(adj, 'b h n1 n2 -> (b h) n1 n2')
        adj = adj + torch.eye(adj.size(1), device=adj.device, dtype=adj.dtype)   # 为每个结点增加自环
        adj = (adj + adj.transpose(2, 1)) / 2
        D = torch.diag_embed(torch.sum(adj, dim=-1) ** (-1 / 2))
        D[D == float('inf')] = 0
        norm_adj = torch.matmul(torch.matmul(D, adj), D)
        norm_adj = rearrange(norm_adj, '(b h) n1 n2 -> b h n1 n2', h=head_num)

        # head_num = adj.size(1)
        # adj = rearrange(adj, 'b h n1 n2 -> (b h) n1 n2')
        # adj = (adj + adj.transpose(2, 1)) / 2
        # norm_adj = rearrange(adj, '(b h) n1 n2 -> b h n1 n2', h=head_num)
        return norm_adj  # norm_adj

    def forward(self, input, adj):

        # 选择在图卷积中全连接层替换为卷积操作
        # (1)
        support = torch.matmul(input, self.weight)
        norm_adj = self.norm(adj)
        output = torch.matmul(norm_adj, support)  # spmm
        # if self.bias is not None:
        #     output = output + self.bias
        support2 = torch.matmul(input, self.weight2)
        output = output + support2

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


from pprint import pprint
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
import torch_geometric.nn as tgnn
from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data, Batch


class GINAttnConv(tgnn.MessagePassing):
    def __init__(self, indim, outdim, head_num):
        super().__init__(aggr="add")

        self.in_features = int(indim / head_num)
        self.out_features = int(outdim / head_num)

        # self.l1 = nn.Linear(self.in_features, self.out_features)
        # self.l2 = nn.Linear(self.out_features, self.out_features)
        # self.l3 = nn.Linear(self.out_features, self.out_features)
        init_eps = 1e-2
        self.eps = torch.nn.Parameter(torch.FloatTensor([init_eps])) #.to(device)

        # self.leaky_relu = torch.nn.LeakyReLU(0.2)
        # self.proj_back = torch.nn.Linear(2 * outdim// head_num, outdim// head_num)

        # self.a = torch.nn.Parameter(torch.zeros(2 * outdim// head_num, 1))
        # torch.nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.mlp = nn.Sequential(
            nn.Linear(self.in_features, 2 * self.out_features),
            nn.ReLU(),
            nn.Linear(2 * self.out_features, self.out_features)
        )


    def forward(self, x, edge_index, edge_values):
        # x = self.l1(x)
        proj = (1 + self.eps) * x

        data_list = []

        # for i in range(x.size(0)):
        #     data_list.append(Data(x=proj[i], edge_index=edge_index[i], edge_attr=edge_values[i] ))

        for i in range(x.size(0)):
            edge_index_temp = add_self_loops(edge_index[i], num_nodes=x.size(1))[0]
            data_list.append(Data(x=proj[i], edge_index=edge_index_temp, edge_attr=edge_values[i] ))


        data_batch = Batch.from_data_list(data_list)
        temp = self.propagate(edge_index=data_batch.edge_index, size=(data_batch.num_nodes, data_batch.num_nodes),
                              x=data_batch.x, edge_attr=data_batch.edge_attr)  #
        return rearrange(temp, '(b h) c -> b h c', b = x.size(0))

        # temp = []
        # for i in range(x.size(0)):
        #     temp.append(self.propagate(edge_index[i], x=proj[i]))
        # temp = torch.stack(temp, dim=0)
        # return temp

    # def message(self, x_j, edge_attr):
    #     return x_j * edge_attr.unsqueeze(1)

    def message(self, x_j, edge_attr):
        edge_attr_selfloop = torch.mean(edge_attr)*torch.ones(x_j.size(0), 1, device=x_j.device, dtype=x_j.dtype)
        edge_attr_selfloop[:len(edge_attr)] = edge_attr.unsqueeze(1)
        return x_j * edge_attr_selfloop

    def update(self, aggr_out):
        # Use the aggregated messages to update the node embeddings.
        return self.mlp(aggr_out)

