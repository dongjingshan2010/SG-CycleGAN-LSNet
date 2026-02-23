import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution, GINAttnConv
import torch
from einops import rearrange
import math


class GCN(nn.Module):
    def __init__(self, nfeat, head_num, nhid, image_size, patch_size, stride, padding, kernel_size, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, head_num, nhid, image_size, patch_size, stride, padding, kernel_size)
        # self.gc2 = GraphConvolution(nfeat, head_num, nhid, image_size, patch_size, stride, padding, kernel_size)
        # self.gc3 = GraphConvolution(nfeat, nhid, image_size, patch_size, stride, padding, kernel_size)
        self.dropout = dropout
        # self.linear = nn.Linear(int(nfeat/head_num), int(nhid/head_num))

    def norm(self, adj, head_num):

        # adj = rearrange(adj, 'b h n1 n2 -> (b h) n1 n2')
        # mean_adj = torch.diag_embed(adj.mean(dim=-1))
        # adj = adj + mean_adj #torch.eye(adj.size(1), device=adj.device, dtype=adj.dtype)   # 为每个结点增加自环

        adj = (adj + adj.transpose(2, 1)) / 2

        D = torch.diag_embed(torch.sum(adj, dim=-1) ** (-1 / 2))
        D[D == float('inf')] = 0
        norm_adj = torch.matmul(torch.matmul(D, adj), D)
        norm_adj = rearrange(norm_adj, '(b h) n1 n2 -> b h n1 n2', h=head_num)

        return norm_adj


    def sparse_adj(self, adj_matrix):
        head_num = adj_matrix.size(1)
        adj_matrix = rearrange(adj_matrix, 'b h t d -> (b h) t d')
        adj_matrix_copy = torch.zeros_like(adj_matrix)

        num_rows = adj_matrix.size(1)
        k = int(math.sqrt(num_rows))


        for i in range(adj_matrix.size(0)):

            top_values, topk_indices = torch.topk(adj_matrix[i, :, :], k, dim=-1)

            topk_indices = topk_indices.view(-1)
            col_indices = topk_indices.unsqueeze(0)


            row_indices = torch.arange(num_rows).repeat(1, k).cuda()
            indices = torch.stack([row_indices, col_indices], dim=0).view(2, -1)
            values = top_values.view(-1)
            sparse_matrix = torch.sparse_coo_tensor(indices, values, (num_rows, num_rows))





            dense_matrix = sparse_matrix.to_dense()

            # topk_indices = topk_indices % (adj_matrix[i].size(0) * adj_matrix[i].size(1))  # 转换为线性索引
            # row_indices = torch.div(topk_indices, adj_matrix[i].size(1), rounding_mode='trunc').long()
            # col_indices = topk_indices % adj_matrix[i].size(1)  # 计算列索引

            adj_matrix_copy[i, :, :] = dense_matrix

        return adj_matrix_copy, head_num

    def forward(self, x, adj):
        head_num = adj.size(1)
        # adj, _ = self.sparse_adj(adj)

        # adj = rearrange(adj, 'b h t d -> (b h) t d')
        # adj = rearrange(adj, '(b h) t d -> b h t d', h=head_num)
        # adj = self.norm(adj, head_num)

        adj_copy = rearrange(adj, 'b h t d -> (b h) t d', h=head_num)

        x = F.gelu(self.gc1(x, adj))  # Having GELU
        x = F.dropout(x, self.dropout, training=self.training)
        return x, adj_copy


class GIN(nn.Module):
    def __init__(self, nfeat, head_num, nhid, image_size, patch_size, stride, padding, kernel_size, nclass, dropout):
        super(GIN, self).__init__()

        self.gc1 = GINAttnConv(nfeat, nhid, head_num)
        self.dropout = dropout
        self.head_num = head_num

    def norm(self, adj, head_num):

        # adj = rearrange(adj, 'b h n1 n2 -> (b h) n1 n2')
        # mean_adj = torch.diag_embed(adj.mean(dim=-1))
        # adj = adj + mean_adj #torch.eye(adj.size(1), device=adj.device, dtype=adj.dtype)   # 为每个结点增加自环

        adj = (adj + adj.transpose(2, 1)) / 2

        D = torch.diag_embed(torch.sum(adj, dim=-1) ** (-1 / 2))
        D[D == float('inf')] = 0
        norm_adj = torch.matmul(torch.matmul(D, adj), D)
        norm_adj = rearrange(norm_adj, '(b h) n1 n2 -> b h n1 n2', h=head_num)

        return norm_adj


    # def to_edgeindex(self, adj):
    #     edges = []
    #
    #     # 遍历邻接矩阵（仅上三角或下三角即可，因为是无向图）
    #     for i in range(adj.size(0)):
    #         for j in range(adj.size(1)):  # 只遍历上三角
    #             if adj[i, j] != 0:  # 如果存在边
    #                 edges.append((i, j))  # 添加到边列表中
    #
    #     # 将边列表转换为edge_index张量
    #     edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    #     return edge_index


    def sparse_adj(self, adj_matrix):
        head_num = adj_matrix.size(1)
        adj_matrix = rearrange(adj_matrix, 'b h t d -> (b h) t d')
        adj_matrix_copy = torch.zeros_like(adj_matrix)

        k = adj_matrix.size(1)*adj_matrix.size(2) // int(math.sqrt(adj_matrix.size(1)))
        for i in range(adj_matrix.size(0)):

            top_values, topk_indices = torch.topk(adj_matrix[i, :, :].view(-1), k)
            topk_indices = topk_indices % (adj_matrix[i].size(0) * adj_matrix[i].size(1))  # 转换为线性索引
            row_indices = torch.div(topk_indices, adj_matrix[i].size(1), rounding_mode='trunc').long()
            col_indices = topk_indices % adj_matrix[i].size(1)  # 计算列索引

            adj_matrix_copy[i][row_indices, col_indices] = adj_matrix[i][row_indices, col_indices]

        return adj_matrix_copy, head_num


    def adjacency_matrix_to_edge_index(self, adj_matrix):

        adj_matrix = rearrange(adj_matrix, 'b h t d -> (b h) t d')
        edge = []
        edge_values = []

        for i in range(adj_matrix.size(0)):
            edges = adj_matrix[i].nonzero(as_tuple=False)  # as_tuple=False是为了确保输出是二维张量而非元组的元组

            row_indices = edges[:, 0]
            col_indices = edges[:, 1]

            edge_index = torch.stack([row_indices, col_indices], dim=0)
            edge.append(edge_index)

            values = adj_matrix[i][row_indices, col_indices]  # 这一步是可选的，如果你需要边的权重
            # values = torch.softmax(values, dim=-1)

            edge_values.append(values)

        # edge = torch.stack(edge, dim=0)  #考虑不堆叠
        # edge_values = torch.stack(edge_values, dim=0)
        return edge, edge_values


    def forward(self, x, adj):
        adj, head_num = self.sparse_adj(adj)

        adj = self.norm(adj, head_num)
        adj_copy = rearrange(adj, 'b h t d -> (b h) t d')

        edge_index, edge_values = self.adjacency_matrix_to_edge_index(adj)

        x = rearrange(x, 'b c h w -> (b c) h w')
        x = F.gelu(self.gc1(x, edge_index, edge_values))  # Having GELU
        x = rearrange(x, '(b c) h w -> b c h w', c = self.head_num)
        x = F.dropout(x, self.dropout, training=self.training)
        return x, adj_copy

