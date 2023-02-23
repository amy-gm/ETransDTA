import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# sag_global_pooling.py
#
# This is the implementation for the SAG-DTA of global architecture, i.e., SAG-DTA(Glob_Pool).
#
# Author: Shugang Zhang
# Created: Wednesday, Aug 4th, 2021
# Last update: Thursday, Aug 5th, 2021

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGPooling, SGConv, GCNConv, SAGEConv, GatedGraphConv, EdgePooling, global_mean_pool as gap, global_max_pool as gmp
# from models.layers import SAGPool


# 3DGCN model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGPooling, GCNConv, SAGEConv, PANConv, TAGConv, PANPooling, global_mean_pool as gap, global_max_pool as gmp
# from models.layers import SAGPool


# 3DGCN model
class edge_global_tag_transformer_121_k4(torch.nn.Module):
    def __init__(self, n_output=1, n_filters=32, embed_dim=128, num_features_xd=78, num_features_xt=25, output_dim=128, dropout=0.1):

        super(edge_global_tag_transformer_121_k4, self).__init__()
        self.pooling_ratio = 1.0

        # SMILES graph branch
        self.n_output = n_output

        self.gc1 = TAGConv(num_features_xd, num_features_xd, K=4)
        self.gc_bn1 = nn.BatchNorm1d(num_features_xd)

        self.gc2 = TAGConv(num_features_xd, num_features_xd, K=4)
        self.gc_bn2 = nn.BatchNorm1d(num_features_xd)

        self.gc3 = TAGConv(num_features_xd, num_features_xd, K=4)
        self.gc_bn3 = nn.BatchNorm1d(num_features_xd)

        # self.pool1 = SAGPooling(3 * num_features_xd, ratio=self.pooling_ratio, GNN=GCNConv)
        self.pool1 = EdgePooling(in_channels=3 * num_features_xd, dropout=0.2, add_to_edge_score=0)

        self.fc_g1 = torch.nn.Linear(3 * num_features_xd, output_dim)  # 1024
        self.gc_bn4 = nn.BatchNorm1d(output_dim)  # 1024
        self.fc_g2 = torch.nn.Linear(1024, output_dim)
        self.bn5 = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # protein sequence branch (1d conv)
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv1 = nn.Conv1d(1000, 1000, kernel_size=3, padding=1, stride=1)
        self.cv_bn1 = nn.BatchNorm1d(1000)
        self.conv2 = nn.Conv1d(1000, 1000, kernel_size=3, padding=1, stride=1)
        self.cv_bn2 = nn.BatchNorm1d(1000)
        self.conv3 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        self.cv_bn3 = nn.BatchNorm1d(32)

        self.cnn_attn = TransformerEncoder(d_model=128, n_head=8, nlayers=3)

        self.fc1_xt = nn.Linear(121, output_dim)

        self.fc2_xt = nn.Linear(128*32, output_dim)
        self.cv_bn5 = nn.BatchNorm1d(output_dim)

        # combined layers.py
        self.fc1 = nn.Linear(2*output_dim, 1024)
        # self.fc1 = nn.Linear(output_dim, 1024)  # protein only
        self.bn7 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn8 = nn.BatchNorm1d(512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, data):
        # get graph input
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # get protein input
        target = data.target  # 512x1000


        x = self.gc1(x, edge_index)
        x = self.gc_bn1(x)
        x = self.relu(x)
        x1 = x

        x = self.gc2(x, edge_index)
        x = self.gc_bn2(x)
        x = self.relu(x)
        x2 = x

        x = self.gc3(x, edge_index)
        x = self.gc_bn3(x)
        x = self.relu(x)
        x3 = x

        x = torch.cat([x1, x2, x3], dim=1)  # 16571*234
        x, edge_index, batch, _ = self.pool1(x, edge_index, batch)
        # x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = gmp(x, batch)

        x = self.fc_g1(x)
        x = self.gc_bn4(x)
        x = self.relu(x)
        x = self.dropout(x)

        embedded_xt = self.embedding_xt(target)  # 512*1000*121

        conv_xt = self.conv1(embedded_xt)  # 512*1000*121
        conv_xt = self.cv_bn1(conv_xt)
        conv_xt = self.relu(conv_xt)

        conv_xt = self.conv2(conv_xt)  # 512*1000*121
        conv_xt = self.cv_bn2(conv_xt)
        conv_xt = self.relu(conv_xt)

        conv_xt = self.conv3(conv_xt)  # 512*32*121
        conv_xt = self.cv_bn3(conv_xt)
        conv_xt = self.relu(conv_xt)

        conv_xt = self.fc1_xt(conv_xt)
        conv_xt = self.relu(conv_xt)
        # conv_xt = self.cv_bn4(conv_xt)
        conv_xt = self.dropout(conv_xt)

        conv_xt = self.cnn_attn(conv_xt)
        # 512,32,128

        # flatten
        xt = conv_xt.view(-1, 32 * 128)
        xt = self.fc2_xt(xt)
        xt = self.relu(xt)
        xt = self.cv_bn5(xt)
        xt = self.dropout(xt)

        # concat
        xc = torch.cat((x, xt), 1)
        # xc = x  # protein only
        # add some dense layers.py
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.bn7(xc)
        xc = self.dropout(xc)

        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.bn8(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out







# by xmm
class ScaledDotProductAttention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention'
    Attention(Q,K,V) = softmax(Q*Kt/sqrt(dk)) *V
    """
    """ for test 
            q = torch.randn(4, 8, 10, 64)  # (batch, n_head, seqLen, dim)
            k = torch.randn(4, 8, 10, 64)
            v = torch.randn(4, 8, 10, 64)
            mask = torch.ones(4, 8, 10, 10)
            model = ScaledDotProductAttention()
            res = model(q, k, v, mask)
            print(res[0].shape)  # torch.Size([4, 8, 10, 64])
    """

    def forward(self, query, key, value, attn_mask=None, dropout=None):
        """
        当QKV来自同一个向量的矩阵变换时称作self-attention;
        当Q和KV来自不同的向量的矩阵变换时叫soft-attention;
        url:https://www.e-learn.cn/topic/3764324
        url:https://my.oschina.net/u/4228078/blog/4497939
          :param query: (batch, n_head, seqLen, dim)  其中n_head表示multi-head的个数，且n_head*dim = embedSize
          :param key: (batch, n_head, seqLen, dim)
          :param value: (batch, n_head, seqLen, dim)
          :param mask:  (batch, n_head, seqLen,seqLen) 这里的mask应该是attn_mask；原来attention的位置为0，no attention部分为1
          :param dropout:
          """
        # (batch, n_head, seqLen,seqLen) attention weights的形状是L*L，因为每个单词两两之间都有一个weight
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)  # 保留位置为0的值，其他位置填充极小的数

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn  # (batch, n_head, seqLen, dim)


# by xmm
class MultiHeadAttention(nn.Module):
    """
    for test :
                q = torch.randn(4, 10, 8 * 64)  # (batch, n_head, seqLen, dim)
                k = torch.randn(4, 10, 8 * 64)
                v = torch.randn(4, 10, 8 * 64)
                mask = torch.ones(4, 8, 10, 10)
                model = MultiHeadAttention(h=8, d_model=8 * 64)
                res = model(q, k, v, mask)
                print(res.shape)  # torch.Size([4, 10, 512])
    """

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, attn_mask=None):
        """
        :param query: (batch,seqLen, d_model)
        :param key: (batch,seqLen, d_model)
        :param value: (batch,seqLen, d_model)
        :param mask: (batch, seqLen,seqLen)
        :return: (batch,seqLen, d_model)
        """
        batch_size = query.size(0)

        # 1, Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2,Apply attention on all the projected vectors in batch.
        if attn_mask:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.h, 1, 1)  # (batch, n_head,seqLen,seqLen)
        x, atten = self.attention(query, key, value, attn_mask=attn_mask, dropout=self.dropout)

        # 3, "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.output_linear(x)


# by xmm
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, dim_feedforward, dropout, activation):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, dim_feedforward)
        self.w_2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x):
        return self.dropout(self.w_2(self.activation(self.w_1(x))))


# by xmm
class TransformerEncoderLayer(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    Example:
    """

    def __init__(self, d_model, n_head, dim_feedforward, dropout=0.1, activation="relu"):
        """
        :param d_model:
        :param n_head:
        :param dim_feedforward:
        :param dropout:
        :param activation: default :relu
        """

        super().__init__()
        self.self_attn = MultiHeadAttention(h=n_head, d_model=d_model, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        if activation == "relu":
            self.activation = F.relu
        if activation == "gelu":
            self.activation = F.gelu

        self.PositionwiseFeedForward = PositionwiseFeedForward(d_model=d_model, dim_feedforward=dim_feedforward,
                                                               dropout=dropout, activation=self.activation)

    def forward(self, x, atten_mask):
        """
        :param x: (batch, seqLen, em_dim)
        :param mask: attn_mask
        :return:
        """
        # add & norm 1
        attn = self.dropout(self.self_attn(x, x, x, attn_mask=atten_mask))
        x = self.norm1((x + attn))

        # # add & norm 2
        x = self.norm2(x + self.PositionwiseFeedForward(x))
        return x


class TransformerEncoder(nn.Module):
    """
    Example:
           x = torch.randn(4, 10, 128)  # (batch, seqLen, em_dim)
        model = TransformerEncoder(d_model=128, n_head=8, nlayers=3)
        res = model.forward(x)
        print(res.shape)  # torch.Size([4, 10, 128])
    """

    def __init__(self, d_model, n_head, nlayers, dim_feedforward=1024, dropout=0.1, activation="relu"):
        super(TransformerEncoder, self).__init__()
        self.encoder = nn.ModuleList([TransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout, activation)
                                      for _ in range(nlayers)])

    def forward(self, x, atten_mask=None):
        """
        :param x: input dim == out dim
        :param atten_mask: 对应源码的src_mask，没有实现src_key_padding_mask
        :return:
        """
        for layer in self.encoder:
            x = layer.forward(x, atten_mask)
        return x


# if __name__ == '__main__':
#     x = torch.randn(512, 32, 128)  # (batch, seqLen, em_dim)
#     model = TransformerEncoder(d_model=128, n_head=8, nlayers=3)
#     res = model.forward(x)
#     print(res.shape)  # torch.Size([4, 10, 128]