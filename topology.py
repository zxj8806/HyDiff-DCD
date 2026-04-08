# -*- coding: utf-8 -*-
import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch
import gudhi as gd
from gudhi.wasserstein import wasserstein_distance


def build_community_graph(Q: torch.Tensor, W: sp.coo_matrix):
    indicator = torch.argmax(Q, dim=1)
    indicator_hat = torch.stack(
        [torch.where(indicator == k, 1.0, 0.0) for k in range(Q.size(1))]
    ).T
    Q_hat = indicator_hat * Q
    if not torch.is_tensor(W):
        W = torch.tensor(W.todense(), dtype=torch.float, device="cuda:0")
    result = torch.mm(torch.mm(Q_hat.T, W), Q_hat).fill_diagonal_(0)
    result = result / W.sum()
    return result


def subgraphCosine(G: nx.Graph):
    st = gd.SimplexTree()
    for v in G.nodes():
        st.insert([v], filtration=0)
    distinct_weights = np.unique([i[2] for i in G.edges.data("weight")])[::-1]
    for t, w in enumerate(distinct_weights):
        subg = G.edge_subgraph(
            [(u, v) for u, v, _w in G.edges.data("weight") if _w >= w]
        )
        for clique in nx.find_cliques(subg):
            st.insert(clique, filtration=1 / w)
    return st


def Cosine_Index(G, dim, c):
    st = subgraphCosine(nx.from_numpy_array(G))
    _ = st.persistence()
    pairs = st.persistence_pairs()

    indices, pers = [], []
    for s1, s2 in pairs:
        if len(s1) == dim + 1 and len(s2) > 0:
            l1, l2 = np.array(s1), np.array(s2)
            i1 = [
                s1[v]
                for v in np.unravel_index(
                    np.argmax(G[l1, :][:, l1]), [len(s1), len(s1)]
                )
            ]
            i2 = [
                s2[v]
                for v in np.unravel_index(
                    np.argmax(G[l2, :][:, l2]), [len(s2), len(s2)]
                )
            ]
            indices += i1
            indices += i2
            pers.append(st.filtration(s2) - st.filtration(s1))

    perm = np.argsort(pers)
    indices = list(np.reshape(indices, [-1, 4])[perm][::-1, :].flatten())

    indices = indices[: 4 * c] + [
        0 for _ in range(0, max(0, 4 * c - len(indices)))
    ]
    return np.array(indices, dtype=np.int32)


class CosineLayer(torch.nn.Module):
    def __init__(self, dim=1, c=50):
        super(CosineLayer, self).__init__()
        self.dim = dim
        self.c = c

    def forward(self, G: torch.Tensor):
        d, c = self.dim, self.c
        G = G.cpu()
        with torch.no_grad():
            ids = torch.from_numpy(Cosine_Index(G.numpy(), d, c))
        if d > 0:
            indices = ids.view([2 * c, 2]).long()
            dgm = G[indices[:, 0], indices[:, 1]].view(c, 2)
        else:
            indices = ids.view([2 * c, 2])[1::2, :].long()
            dgm = torch.cat(
                [
                    torch.zeros(c, 1),
                    G[indices[:, 0], indices[:, 1]].view(c, 1).float(),
                ],
                dim=1,
            )
        return dgm.cuda()


class TempLoss(torch.nn.Module):
    def __init__(self, nearby_temps, args) -> None:
        super().__init__()
        self.layer0 = CosineLayer(dim=0, c=args.c)
        self.layer1 = CosineLayer(dim=1, c=args.c)
        self.temp_gt = nearby_temps
        self.LAMBDA = args.LAMBDA

    def forward(self, adj, soft_label):
        C = build_community_graph(soft_label, adj)
        temp0 = self.layer0(C)
        temp1 = self.layer1(C)

        if self.temp_gt[0]:
            topo_dim0_before = wasserstein_distance(
                temp0,
                self.temp_gt[0][0],
                order=1,
                enable_autodiff=True,
                keep_essential_parts=False,
            )
            topo_dim1_before = wasserstein_distance(
                temp1,
                self.temp_gt[0][1],
                order=1,
                enable_autodiff=True,
                keep_essential_parts=False,
            )
        if self.temp_gt[1]:
            topo_dim0_next = wasserstein_distance(
                temp0,
                self.temp_gt[1][0],
                order=1,
                enable_autodiff=True,
                keep_essential_parts=False,
            )
            topo_dim1_next = wasserstein_distance(
                temp1,
                self.temp_gt[1][1],
                order=1,
                enable_autodiff=True,
                keep_essential_parts=False,
            )

        if not self.temp_gt[0]:
            loss_topo = topo_dim0_next + topo_dim1_next
        elif not self.temp_gt[1]:
            loss_topo = topo_dim0_before + topo_dim1_before
        else:
            loss_topo = (
                topo_dim0_before
                + topo_dim1_before
                + topo_dim0_next
                + topo_dim1_next
            )
        return self.LAMBDA * loss_topo
