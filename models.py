# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return nn.Parameter(initial, requires_grad=True)


class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, adj, activation=F.leaky_relu, **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)
        self.weight = glorot_init(input_dim, output_dim)
        self.adj = adj
        self.activation = activation

    def forward(self, x):
        x = torch.mm(x, self.weight)
        x = torch.mm(self.adj, x)
        outputs = self.activation(x)
        return outputs


class HyDiff_DCD(nn.Module):
    def __init__(self, adj, feature_dim, args):
        super(HyDiff_DCD, self).__init__()
        self.start_mf = args.start_mf
        self.base_gcn = GraphConvSparse(feature_dim, args.encoded_space_dim, adj)
        self.cluster_centroid = glorot_init(args.n_cluster, args.encoded_space_dim)
        self.log_kappa = nn.Parameter(torch.tensor(10.0))

    def kappa(self):
        return torch.nn.functional.softplus(self.log_kappa)

    def restart_clusters(self):
        torch.nn.init.xavier_normal_(self.cluster_centroid.data)

    def encode(self, _X):
        hidden_z = self.base_gcn(_X)
        return hidden_z

    @staticmethod
    def normalize(X):
        X_std = (X - X.min(dim=1).values[:, None]) / (
            X.max(dim=1).values - X.min(dim=1).values
        )[:, None]
        return X_std / torch.sum(X_std, dim=1)[:, None]

    def forward(self, _input, flag):
        z = self.encode(_input)
        A_pred = torch.sigmoid(torch.matmul(z, z.t()))
        if type(flag) != bool or flag is True:
            pinv_weight = torch.linalg.pinv(self.cluster_centroid)
            indicator = self.normalize(torch.mm(z, pinv_weight))
            return A_pred, z, indicator
        else:
            return A_pred, z, None
