# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F


def linear_ramp(epoch: int, start: int, end: int, maxv: float) -> float:
    if epoch <= start:
        return 0.0
    if epoch >= end:
        return maxv
    return (epoch - start) / float(end - start) * maxv


def linear_beta_schedule(timesteps: int, beta_start=1e-4, beta_end=2e-2, device=None):
    return torch.linspace(beta_start, beta_end, timesteps, device=device)


def compute_diffusion_params(timesteps: int, device):
    betas = linear_beta_schedule(timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_1m = torch.sqrt(1.0 - alphas_cumprod)
    rev = torch.flip(sqrt_1m, dims=[0]).cumsum(0)
    norm = torch.flip(rev, dims=[0])
    return sqrt_1m, norm


def decode_diffusion_graph(z, adj_norm, T=8, start=1):
    device = z.device
    sqrt_1m, cum_norm = compute_diffusion_params(T, device=device)
    denom = cum_norm[start - 1]
    acc = None
    z_t = F.normalize(z, p=2, dim=1)
    for t in range(start, T + 1):
        if t > start:
            if adj_norm.is_sparse:
                z_t = torch.sparse.mm(adj_norm, z_t)
            else:
                z_t = adj_norm @ z_t
            z_t = F.normalize(z_t, p=2, dim=1)
        sim = z_t @ z_t.t()
        w = sqrt_1m[t - 1]
        acc = sim * w if acc is None else acc + sim * w
    return torch.clamp(acc / denom, 0, 1)
