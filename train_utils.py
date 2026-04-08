# -*- coding: utf-8 -*-
import time
import torch
import torch.nn.functional as F

from bessel_vmf import vmf_kl_to_uniform
from data_utils import sparse_to_tuple, get_acc
from diffusion_utils import decode_diffusion_graph, linear_ramp
from topology import TempLoss


def trainer(model, features, adj, args, topo, idx):
    device = "cuda"
    coords, values, shape = sparse_to_tuple(adj)
    indices = torch.LongTensor(coords.T).to(device)
    values_t = torch.FloatTensor(values).to(device)
    adj_label = torch.sparse_coo_tensor(
        indices,
        values_t,
        torch.Size(shape),
        device=device,
        dtype=torch.float32,
    ).coalesce()

    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float(
        (adj.shape[0] * adj.shape[0] - adj.sum()) * 2
    )
    weight_mask = adj_label.to_dense().view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0), device=device)
    weight_tensor[weight_mask] = pos_weight

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    model.train()
    for epoch in range(args.num_epoch):
        t = time.time()
        mf_flag = epoch > args.start_mf
        if topo:
            mf_flag = True
        A_pred, z, q = model(features, mf_flag)

        A_diff = decode_diffusion_graph(z, model.base_gcn.adj, T=args.diffuse_T)

        re_base = norm * F.binary_cross_entropy(
            A_pred.view(-1),
            adj_label.to_dense().view(-1),
            weight=weight_tensor,
        )
        re_diff = norm * F.binary_cross_entropy(
            A_diff.view(-1),
            adj_label.to_dense().view(-1),
            weight=weight_tensor,
        )
        w_diff = linear_ramp(
            epoch,
            start=args.start_mf,
            end=args.start_mf + 100,
            maxv=args.diff_rec_weight,
        )
        re_loss = re_base + w_diff * re_diff

        z_u = F.normalize(z, p=2, dim=1)
        kappa = model.kappa()
        kl = vmf_kl_to_uniform(kappa, p=z_u.size(1))
        w_kl = linear_ramp(
            epoch, start=args.start_mf, end=args.start_mf + 100, maxv=0.01
        )
        loss_kl = w_kl * kl

        w_clu = linear_ramp(
            epoch,
            start=args.start_mf,
            end=args.start_mf + 100,
            maxv=args.cluster_reg_weight,
        )
        if (q is not None) and (w_clu > 0.0):
            centers = F.normalize(model.cluster_centroid, p=2, dim=1)
            p_soft = q
            intra = (
                (
                    p_soft[:, :, None]
                    * (z_u[:, None, :] - centers[None, :, :]).pow(2)
                ).sum()
                / z_u.size(0)
            )
            inter = torch.pdist(centers, p=2).mean()
            loss_clu_raw = intra / (inter + 1e-9)
            loss_clu = w_clu * loss_clu_raw
        else:
            loss_clu_raw = torch.tensor(0.0, device=device)
            loss_clu = loss_clu_raw

        if topo:
            q_safe = (
                q
                if q is not None
                else model.normalize(
                    torch.mm(z, torch.linalg.pinv(model.cluster_centroid))
                )
            )
            loss = topo(adj, q_safe) + re_loss + loss_kl + loss_clu
        elif epoch > args.start_mf:
            loss_kmeans = F.mse_loss(z, torch.mm(q, model.cluster_centroid))
            loss = 1 * loss_kmeans + re_loss + loss_kl + loss_clu
        else:
            loss = re_loss + loss_kl + loss_clu

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            acc_base = get_acc(A_pred, adj_label.to_dense())
            acc_diff = get_acc(A_diff, adj_label.to_dense())

        if epoch % 100 == 0:
            if int(idx) >= 9:
                train_acc = 0
            else:
                train_acc = acc_base
            print(
                "Epoch:",
                "%04d" % (epoch + 1),
                "re_base=",
                "{:.5f}".format(re_base.item()),
                "re_diff=",
                "{:.5f}".format(re_diff.item()),
                "kl=",
                "{:.5f}".format(
                    kl.item() if torch.is_tensor(kl) else float(kl)
                ),
                "clu=",
                "{:.5f}".format(
                    loss_clu_raw.item()
                    if torch.is_tensor(loss_clu_raw)
                    else float(loss_clu_raw)
                ),
                "Acc(diff)=",
                "{:.5f}".format(acc_diff),
                "time=",
                "{:.5f}".format(time.time() - t),
            )


def base_train(network_type, model, features, adj, args, idx):
    if network_type == "HyDiff-DCD":
        trainer(model, features, adj, args, None, idx)
    else:
        raise ValueError(f"Unknown network type: {network_type}")


def retrain_with_Consistency(network_type, _model, temp_gt, adj, features, args, idx):
    topo_loss = TempLoss(nearby_temps=temp_gt, args=args)
    if network_type == "HyDiff-DCD":
        trainer(_model, features, adj, args, topo_loss, idx)
    else:
        raise ValueError(f"Unknown network type: {network_type}")
