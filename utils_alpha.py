import os
import csv
import math
import random
from typing import Optional, Tuple, Dict, Iterable, Any

import torch
import torch.nn as nn
from loguru import logger


USE_FARMS: bool = True
_env_mode = os.getenv("ALPHA_MODE", "").strip().upper()
if _env_mode in {"FARMS", "BASELINE"}:
    USE_FARMS = (_env_mode == "FARMS")

FARMS_M_SUB: int = int(os.getenv("FARMS_M_SUB", "128"))
FARMS_N_SUB: int = int(os.getenv("FARMS_N_SUB", "128"))
FARMS_STRIDE_M: int = int(os.getenv("FARMS_STRIDE_M", str(FARMS_M_SUB)))
FARMS_STRIDE_N: int = int(os.getenv("FARMS_STRIDE_N", str(FARMS_N_SUB)))
FARMS_MAX_BLOCKS: int = int(os.getenv("FARMS_MAX_BLOCKS", "256"))
FARMS_RANDOM_SEED: Optional[int] = (
    int(os.getenv("FARMS_SEED", "0")) if os.getenv("FARMS_SEED") else None
)


def _ensure_2d_dense_weight(W: torch.Tensor) -> torch.Tensor:
    if W.is_sparse:
        W = W.to_dense()
    if W.ndim > 2:
        W = W.reshape(W.shape[0], -1)
    return W


@torch.no_grad()
def _svd_eigs_baseline(W: torch.Tensor) -> torch.Tensor:
    W = _ensure_2d_dense_weight(W)
    m, n = W.shape
    if min(m, n) < 2:
        return torch.tensor([], dtype=torch.float32)
    W_ = W.to(dtype=torch.float32, device="cpu")
    s = torch.linalg.svdvals(W_)
    lam = (s ** 2)
    lam, _ = torch.sort(lam)
    return lam


def _iter_farms_blocks_indices(
    m: int,
    n: int,
    m_sub: int,
    n_sub: int,
    stride_m: int,
    stride_n: int,
) -> Iterable[Tuple[int, int]]:
    if m_sub > m or n_sub > n:
        return []
    for i in range(0, m - m_sub + 1, max(1, stride_m)):
        for j in range(0, n - n_sub + 1, max(1, stride_n)):
            yield (i, j)


@torch.no_grad()
def _svd_eigs_farms(
    W: torch.Tensor,
    m_sub: int = FARMS_M_SUB,
    n_sub: int = FARMS_N_SUB,
    stride_m: int = FARMS_STRIDE_M,
    stride_n: int = FARMS_STRIDE_N,
    max_blocks: int = FARMS_MAX_BLOCKS,
    seed: Optional[int] = FARMS_RANDOM_SEED,
) -> torch.Tensor:
    W = _ensure_2d_dense_weight(W)
    m, n = W.shape
    if min(m, n) < 2:
        return torch.tensor([], dtype=torch.float32)

    if m_sub > m or n_sub > n:
        return _svd_eigs_baseline(W)

    idx = list(
        _iter_farms_blocks_indices(m, n, m_sub, n_sub, stride_m, stride_n)
    )
    if len(idx) == 0:
        return _svd_eigs_baseline(W)

    if seed is not None:
        random.seed(seed)
    if len(idx) > max_blocks:
        idx = random.sample(idx, max_blocks)

    W_cpu = W.to(dtype=torch.float32, device="cpu")

    eig_list = []
    for (i, j) in idx:
        sub = W_cpu[i : i + m_sub, j : j + n_sub]
        s = torch.linalg.svdvals(sub)
        lam = (s ** 2)
        eig_list.append(lam)

    if not eig_list:
        return torch.tensor([], dtype=torch.float32)

    lam_cat = torch.cat(eig_list, dim=0)
    lam_cat, _ = torch.sort(lam_cat)
    return lam_cat


@torch.no_grad()
def _hill_alpha_from_sorted_eigs(
    lam_sorted: torch.Tensor,
    k: Optional[int] = None,
    k_frac: float = 0.1,
    eps: float = 1e-12,
) -> Tuple[float, int, int]:
    n_eigs = lam_sorted.numel()
    if n_eigs < 2:
        return float("nan"), 1, n_eigs

    k_used = max(10, int(n_eigs * k_frac)) if k is None else int(k)
    k_used = max(1, min(k_used, n_eigs - 1))

    eps_t = torch.tensor(eps, dtype=lam_sorted.dtype, device=lam_sorted.device)
    lam_ref = torch.clamp(lam_sorted[-k_used - 1], min=eps_t)
    top = lam_sorted[-k_used:]
    denom = torch.log(top / lam_ref).sum().clamp_min(eps_t)
    alpha = float(1.0 + (k_used / float(denom)))
    return alpha, k_used, n_eigs


@torch.no_grad()
def alpha_hill_from_weight(
    W: torch.Tensor,
    k: Optional[int] = None,
    k_frac: float = 0.1,
    eps: float = 1e-12,
    *,
    use_farms: Optional[bool] = None,
    farms_m_sub: int = FARMS_M_SUB,
    farms_n_sub: int = FARMS_N_SUB,
    farms_stride_m: int = FARMS_STRIDE_M,
    farms_stride_n: int = FARMS_STRIDE_N,
    farms_max_blocks: int = FARMS_MAX_BLOCKS,
    farms_seed: Optional[int] = FARMS_RANDOM_SEED,
) -> Tuple[float, int, int]:
    mode_farms = USE_FARMS if use_farms is None else bool(use_farms)

    if mode_farms:
        lam_sorted = _svd_eigs_farms(
            W,
            m_sub=farms_m_sub,
            n_sub=farms_n_sub,
            stride_m=farms_stride_m,
            stride_n=farms_stride_n,
            max_blocks=farms_max_blocks,
            seed=farms_seed,
        )
    else:
        lam_sorted = _svd_eigs_baseline(W)

    if lam_sorted.numel() < 2:
        min_dim = (
            min(W.shape[0], W.reshape(W.shape[0], -1).shape[1])
            if W.ndim > 1
            else 1
        )
        return float("nan"), 1, int(min_dim)

    return _hill_alpha_from_sorted_eigs(lam_sorted, k=k, k_frac=k_frac, eps=eps)


def compute_alpha_values(
    model: nn.Module,
    cache_dir: Optional[str] = None,
    *,
    use_farms: Optional[bool] = None,
) -> Dict[str, Dict[str, float]]:
    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        mode_tag = "farms" if (USE_FARMS if use_farms is None else use_farms) else "baseline"
        cache_path = os.path.join(cache_dir, f"alpha_values_{mode_tag}.csv")
        if os.path.exists(cache_path):
            logger.info(f"Loading alpha values from cache: {cache_path}")
            cached_results = load_alpha_from_csv(cache_path)
            if cached_results and all(
                isinstance(stats, dict)
                and "alpha" in stats
                and "variance" in stats
                and stats["alpha"] == stats["alpha"]
                and stats["variance"] == stats["variance"]
            for stats in cached_results.values()):
                return cached_results
            logger.info("Cached alpha values missing variance information. Recomputing.")

    logger.info("Computing alpha values for all linear layers...")
    results: Dict[str, Dict[str, float]] = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight = getattr(module, "weight", None)
            if weight is None:
                continue
            try:
                detached_weight = weight.detach()
                alpha, k_used, n_eigs = alpha_hill_from_weight(
                    weight.detach(),
                    use_farms=use_farms,
                )
                weight_cpu = detached_weight.to(dtype=torch.float32, device="cpu")
                variance = float(torch.var(weight_cpu, unbiased=False).item())
                results[name] = {
                    "alpha": alpha,
                    "variance": variance,
                }
            except Exception as e:
                logger.warning(f"Failed to compute alpha for {name}: {e}")
                results[name] = {
                    "alpha": float("nan"),
                    "variance": float("nan"),
                }

    if cache_path:
        logger.info(f"Saving alpha values to: {cache_path}")
        save_alpha_to_csv(results, cache_path)

    return results


def save_alpha_to_csv(alpha_results: Dict[str, Dict[str, float]], filename: str) -> None:
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer_name", "alpha", "variance"])
        for name, stats in alpha_results.items():
            if isinstance(stats, dict):
                alpha_val = stats.get("alpha", float("nan"))
                variance_val = stats.get("variance", float("nan"))
            else:
                alpha_val = float(stats)
                variance_val = float("nan")
            writer.writerow([name, alpha_val, variance_val])


def load_alpha_from_csv(filename: str) -> Dict[str, Dict[str, float]]:
    alpha_results: Dict[str, Dict[str, float]] = {}
    with open(filename, "r") as f:
        reader = csv.DictReader(f)
        has_variance = "variance" in (reader.fieldnames or [])
        for row in reader:
            try:
                alpha_val = float(row["alpha"])
            except (ValueError, KeyError):
                alpha_val = float("nan")
            variance_val = float("nan")
            if has_variance:
                try:
                    variance_val = float(row.get("variance", float("nan")))
                except (ValueError, TypeError):
                    variance_val = float("nan")
            alpha_results[row.get("layer_name", "")] = {
                "alpha": alpha_val,
                "variance": variance_val,
            }

    # Remove potential empty keys from malformed rows
    alpha_results = {
        name: stats for name, stats in alpha_results.items() if name
    }
    return alpha_results


__all__ = [
    "USE_FARMS",
    "FARMS_M_SUB",
    "FARMS_N_SUB",
    "FARMS_STRIDE_M",
    "FARMS_STRIDE_N",
    "FARMS_MAX_BLOCKS",
    "FARMS_RANDOM_SEED",
    "alpha_hill_from_weight",
    "compute_alpha_values",
    "save_alpha_to_csv",
    "load_alpha_from_csv",
]

