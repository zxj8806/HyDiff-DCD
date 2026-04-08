#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import pickle
import torch
from tensorflow.keras.utils import to_categorical  # unused, kept for compatibility

from args import Args
from data_utils import load_graphs, graph_normalization
from models import HyDiff_DCD
from topology import CosineLayer, build_community_graph
from train_utils import base_train, retrain_with_Consistency


def main(file_name, network_type):
    device = "cuda"
    snapshot_list, n_cluster = load_graphs(file_name=file_name, network_type=network_type)
    print(len(snapshot_list))
    args = Args(n_cluster, file_name, network_type)

    layer0 = CosineLayer(dim=0, c=args.c)
    layer1 = CosineLayer(dim=1, c=args.c)

    results_raw = []
    results_topo = []
    model_list = {}

    for idx, (adj, features, labels) in enumerate(snapshot_list):
        adj_norm = graph_normalization(adj)
        indices = torch.LongTensor(adj_norm[0].T).to(device)
        values_t = torch.FloatTensor(adj_norm[1]).to(device)
        adj_norm_t = torch.sparse_coo_tensor(
            indices,
            values_t,
            torch.Size(adj_norm[2]),
            device=device,
            dtype=torch.float32,
        ).coalesce()

        model = HyDiff_DCD(adj_norm_t, features.size(1), args).to(device)
        model_list[idx] = model
        base_train(network_type, model, features, adj, args, str(idx))

        with torch.no_grad():
            _, Z, Q = model(features, True)
            results_raw.append(
                [
                    Z.cpu().detach().numpy(),
                    Q.cpu().detach().numpy(),
                    adj,
                    labels,
                ]
            )
            community_graph = build_community_graph(Q, adj)
            temp0 = layer0(community_graph)
            temp1 = layer1(community_graph)
            model_list[idx]._cached_temp = [temp0, temp1]
            results_topo.append(None)

    for t in range(len(snapshot_list)):
        m = model_list[t]
        adj, features, labels = snapshot_list[t]
        if t == 0:
            _temp = [None, model_list[t + 1]._cached_temp]
        elif t == len(snapshot_list) - 1:
            _temp = [model_list[t - 1]._cached_temp, None]
        else:
            _temp = [model_list[t - 1]._cached_temp, model_list[t + 1]._cached_temp]

        retrain_with_Consistency(network_type, m, _temp, adj, features, args, str(t))

        with torch.no_grad():
            _, Z, Q = m(features, True)
            community_graph = build_community_graph(Q, adj)
            temp0 = layer0(community_graph)
            temp1 = layer1(community_graph)
            model_list[t]._cached_temp = [temp0, temp1]
            results_topo[t] = [
                Z.cpu().detach().numpy(),
                Q.cpu().detach().numpy(),
                adj,
                labels,
            ]

    out_dir = os.path.join("Data", file_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results_raw.pkl"), "wb") as handle:
        pickle.dump(results_raw, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(out_dir, "results_topo.pkl"), "wb") as handle:
        pickle.dump(results_topo, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    torch.manual_seed(42)
    network = "HyDiff-DCD"
    dataset = "DBLPdyn"
    print(dataset)
    main(dataset, network_type=network)
