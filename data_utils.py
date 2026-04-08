# -*- coding: utf-8 -*-
import pickle
import numpy as np
import scipy.sparse as sp
import networkx as nx
import torch

graph_pkl = ["enron", "highschool", "DBLP", "Cora", "DBLPdyn"]
label_num_dic = {"Cora": 10, "enron": 7, "highschool": 9, "DBLP": 15, "DBLPdyn": 14}
COMPLETE_GRAPH = False


def load_graphs(file_name, network_type):
    if file_name in graph_pkl:
        return load_graphs_pkl("Data/" + file_name, network_type)
    else:
        raise NameError(f"Unknown dataset {file_name}")


def load_graphs_pkl(file_name, network_type, complete_graph=COMPLETE_GRAPH):
    with open(file_name + ".pkl", "rb") as handle:
        try:
            graph_snapshots = pickle.load(handle, encoding="bytes", fix_imports=True)
        except ValueError:
            handle.seek(0)
            graph_snapshots = pickle.load(
                handle, encoding="bytes", fix_imports=True, protocol=2
            )
    with open(file_name + "_label.pkl", "rb") as handle:
        try:
            labels = pickle.load(handle, encoding="bytes", fix_imports=True)
        except ValueError:
            handle.seek(0)
            labels = pickle.load(
                handle, encoding="bytes", fix_imports=True, protocol=2
            )

    print("Lengths of snapshots:", len(graph_snapshots))
    print("Types of labels:", label_num_dic[file_name.split("/")[-1]])
    # keep the original condition literally
    if file_name == "DBLP":
        graph_snapshots = graph_snapshots[:8]
    if complete_graph:
        graph_snapshots = get_complete_graphs(graph_snapshots)
    return graphSnapshots(
        graph_snapshots, labels, network_type, file_name
    ), label_num_dic[file_name.split("/")[-1]]


def graphSnapshots(graph_snapshots, labels, network_type, file_name):
    snapshots = []
    if file_name != "Data/DBLPdyn":
        label_set = set(labels.values())
        label_map = {l: i for i, l in enumerate(label_set)}
    else:
        label_map = {str(i): i - 1 for i in range(1, 15)}
        assert len(label_map) == 14
    for i, g in enumerate(graph_snapshots):
        adj = sp.coo_matrix(nx.adjacency_matrix(g))
        features = sp.coo_matrix(np.eye(adj.shape[0]), dtype=np.int64)
        coords, values, shape = sparse_to_tuple(features.tocoo())
        indices = torch.LongTensor(coords.T).to("cuda:0")
        values_t = torch.ones(
            indices.size(1), device="cuda:0", dtype=torch.float32
        )
        features = torch.sparse_coo_tensor(
            indices,
            values_t,
            torch.Size(shape),
            device="cuda:0",
            dtype=torch.float32,
        ).coalesce()
        if file_name == "Data/DBLPdyn":
            label_snap = [
                label_map[labels[i][n]] for idx, n in enumerate(g.nodes())
            ]
        else:
            label_snap = [label_map[labels[n]] for idx, n in enumerate(g.nodes())]
        snapshots.append([adj, features, label_snap])
    return snapshots


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def graph_normalization(adj):
    if isinstance(adj, sp.coo_matrix):
        adj_ = adj + sp.eye(adj.shape[0])
        rowsum = np.array(adj_.sum(1))
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = (
            adj_.dot(degree_mat_inv_sqrt)
            .transpose()
            .dot(degree_mat_inv_sqrt)
            .tocoo()
        )
        return sparse_to_tuple(adj_normalized)
    elif isinstance(adj, torch.Tensor):
        device = torch.device("cuda" if adj.is_cuda else "cpu")
        mx = adj + torch.eye(adj.shape[0]).to(device)
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1 / 2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.0
        r_mat_inv = torch.diag(r_inv)
        mx = torch.mm(r_mat_inv, mx)
        mx = torch.mm(mx, r_mat_inv)
        return mx
    else:
        raise TypeError


def get_complete_graphs(dynamic_graph):
    all_node_ids = set()
    for graph in dynamic_graph:
        all_node_ids.update(graph.nodes)
    complete_graphs = []
    for graph in dynamic_graph:
        complete_graph = nx.Graph()
        complete_graph.add_nodes_from(all_node_ids)
        complete_graphs.append(complete_graph)
    return complete_graphs


def get_acc(adj_rec, adj_label):
    labels_all = adj_label.long()
    preds_all = torch.gt(adj_rec, 0.5).long()
    return torch.eq(labels_all, preds_all).float().mean()
