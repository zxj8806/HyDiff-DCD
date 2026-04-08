# -*- coding: utf-8 -*-
import math
import numpy as np
import torch

_TORCH_IVE = False
try:
    import torch.special as _ts

    if hasattr(_ts, "ive"):
        _TORCH_IVE = True
        _torch_ive = _ts.ive
except Exception:
    _TORCH_IVE = False

_warned_ive_once = False


def _ive_fn(v_tensor: torch.Tensor, x_tensor: torch.Tensor) -> torch.Tensor:
    global _warned_ive_once
    if _TORCH_IVE:
        return _torch_ive(v_tensor, x_tensor)
    from scipy.special import ive as _scipy_ive

    if not _warned_ive_once:
        print(
            "torch.special.ive not found; falling back to scipy.special.ive (no gradient wrt kappa)."
        )
        _warned_ive_once = True
    v_val = (
        float(v_tensor.detach().cpu().item())
        if torch.is_tensor(v_tensor)
        else float(v_tensor)
    )
    x_np = x_tensor.detach().cpu().numpy()
    y_np = _scipy_ive(v_val, x_np)
    y_np = np.asarray(y_np, dtype=np.float64)
    y = torch.tensor(y_np, device=x_tensor.device, dtype=x_tensor.dtype)
    return y


def _log_Iv(v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    x_safe = torch.clamp(x, min=1e-8)
    return torch.log(_ive_fn(v, x_safe)) + x_safe


def _A_p(kappa: torch.Tensor, p: int) -> torch.Tensor:
    v = torch.tensor(p / 2.0 - 1.0, dtype=kappa.dtype, device=kappa.device)
    num = _ive_fn(v + 1.0, torch.clamp(kappa, min=1e-8))
    den = _ive_fn(v, torch.clamp(kappa, min=1e-8))
    return num / (den + 1e-20)


def _log_C_p(kappa: torch.Tensor, p: int) -> torch.Tensor:
    v = torch.tensor(p / 2.0 - 1.0, dtype=kappa.dtype, device=kappa.device)
    return (
        v * torch.log(torch.clamp(kappa, min=1e-8))
        - (p / 2.0) * math.log(2.0 * math.pi)
        - _log_Iv(v, kappa)
    )


def _log_surface_area_S(p: int) -> float:
    return (
        math.log(2.0)
        + (p / 2.0) * math.log(math.pi)
        - torch.lgamma(torch.tensor(p / 2.0)).item()
    )


def vmf_kl_to_uniform(kappa: torch.Tensor, p: int) -> torch.Tensor:
    logC = _log_C_p(kappa, p)
    A = _A_p(kappa, p)
    logS = _log_surface_area_S(p)
    return logC + kappa * A + logS
