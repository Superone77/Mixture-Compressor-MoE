import os
import pickle
import time
import torch
import logging
import torch.nn as nn
from transformers.models.mixtral.modeling_mixtral import MixtralForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import csv
import re
from pathlib import Path
from typing import Optional, Tuple, Dict, Iterable
import math
import random
import pandas as pd

from gptq import GPTQ
from modelutils import find_layers
from datautils import get_loaders
from quant.QLinear import *
from loguru import logger

try:
    import pulp  # MILP 建模与求解器接口
except ImportError as e:
    pulp = None
    logger.warning(f"PuLP not installed. no_calib_auto_programming mode will not work. Install with: pip install pulp")

atten_modules = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
]
expert_modules = [
    "block_sparse_moe.experts.0.w1",
    "block_sparse_moe.experts.1.w1",
    "block_sparse_moe.experts.2.w1",
    "block_sparse_moe.experts.3.w1",
    "block_sparse_moe.experts.4.w1",
    "block_sparse_moe.experts.5.w1",
    "block_sparse_moe.experts.6.w1",
    "block_sparse_moe.experts.7.w1",
    "block_sparse_moe.experts.0.w3",
    "block_sparse_moe.experts.1.w3",
    "block_sparse_moe.experts.2.w3",
    "block_sparse_moe.experts.3.w3",
    "block_sparse_moe.experts.4.w3",
    "block_sparse_moe.experts.5.w3",
    "block_sparse_moe.experts.6.w3",
    "block_sparse_moe.experts.7.w3",
    "block_sparse_moe.experts.0.w2",
    "block_sparse_moe.experts.1.w2",
    "block_sparse_moe.experts.2.w2",
    "block_sparse_moe.experts.3.w2",
    "block_sparse_moe.experts.4.w2",
    "block_sparse_moe.experts.5.w2",
    "block_sparse_moe.experts.6.w2",
    "block_sparse_moe.experts.7.w2",
]


# ========================
# ======== MACROS ========
# ========================
# 方式一：直接改布尔开关
USE_FARMS: bool = False

# 方式二：用环境变量（优先级高于 USE_FARMS）
#   export ALPHA_MODE=FARMS  或  BASELINE
_env_mode = os.getenv("ALPHA_MODE", "").strip().upper()
if _env_mode in {"FARMS", "BASELINE"}:
    USE_FARMS = (_env_mode == "FARMS")

# FARMS 的宏（也可用环境变量覆盖）
FARMS_M_SUB: int   = int(os.getenv("FARMS_M_SUB", "128"))   # 子块行数 m'
FARMS_N_SUB: int   = int(os.getenv("FARMS_N_SUB", "128"))   # 子块列数 n'
FARMS_STRIDE_M: int = int(os.getenv("FARMS_STRIDE_M", str(FARMS_M_SUB)))  # 行步长
FARMS_STRIDE_N: int = int(os.getenv("FARMS_STRIDE_N", str(FARMS_N_SUB)))  # 列步长
FARMS_MAX_BLOCKS: int = int(os.getenv("FARMS_MAX_BLOCKS", "256"))         # 最多抽样子块数
FARMS_RANDOM_SEED: Optional[int] = int(os.getenv("FARMS_SEED", "0")) if os.getenv("FARMS_SEED") else None

# ================================
# ======== Core Utilities ========
# ================================

def _ensure_2d_dense_weight(W: torch.Tensor) -> torch.Tensor:
    if W.is_sparse:
        W = W.to_dense()
    if W.ndim > 2:
        W = W.reshape(W.shape[0], -1)
    return W

@torch.no_grad()
def _svd_eigs_baseline(W: torch.Tensor) -> torch.Tensor:
    """
    Baseline: 对整矩阵做 SVD，返回特征值（奇异值平方）升序张量 lam
    """
    W = _ensure_2d_dense_weight(W)
    m, n = W.shape
    if min(m, n) < 2:
        return torch.tensor([], dtype=torch.float32)
    W_ = W.to(dtype=torch.float32, device="cpu")
    s = torch.linalg.svdvals(W_)
    lam = (s ** 2)
    lam, _ = torch.sort(lam)
    return lam

def _iter_farms_blocks_indices(m: int, n: int,
                               m_sub: int, n_sub: int,
                               stride_m: int, stride_n: int) -> Iterable[Tuple[int, int]]:
    """
    生成 FARMS 子块左上角坐标 (i, j)
    """
    if m_sub > m or n_sub > n:
        return []
    for i in range(0, m - m_sub + 1, max(1, stride_m)):
        for j in range(0, n - n_sub + 1, max(1, stride_n)):
            yield (i, j)

@torch.no_grad()
def _svd_eigs_farms(W: torch.Tensor,
                    m_sub: int = FARMS_M_SUB,
                    n_sub: int = FARMS_N_SUB,
                    stride_m: int = FARMS_STRIDE_M,
                    stride_n: int = FARMS_STRIDE_N,
                    max_blocks: int = FARMS_MAX_BLOCKS,
                    seed: Optional[int] = FARMS_RANDOM_SEED) -> torch.Tensor:
    """
    FARMS: 固定宽高比的子矩阵抽样 + 谱拼接。
    返回拼接后（所有子块 SVD 的奇异值平方）的升序张量 lam_cat
    """
    W = _ensure_2d_dense_weight(W)
    m, n = W.shape
    if min(m, n) < 2:
        return torch.tensor([], dtype=torch.float32)

    # 如果矩阵本身比子块小，回退到 baseline（与 FARMS 目标等价）
    if m_sub > m or n_sub > n:
        return _svd_eigs_baseline(W)

    # 生成所有候选子块索引
    idx = list(_iter_farms_blocks_indices(m, n, m_sub, n_sub, stride_m, stride_n))
    if len(idx) == 0:
        return _svd_eigs_baseline(W)

    # 控制计算量：必要时随机抽样若干子块
    if seed is not None:
        random.seed(seed)
    if len(idx) > max_blocks:
        idx = random.sample(idx, max_blocks)

    W_cpu = W.to(dtype=torch.float32, device="cpu")

    eig_list = []
    for (i, j) in idx:
        sub = W_cpu[i:i+m_sub, j:j+n_sub]
        # 对子块做 SVD
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
    """
    对“升序特征值序列 lam_sorted”计算 Hill α。
    返回: (alpha, k_used, n_eigs)
    """
    n_eigs = lam_sorted.numel()
    if n_eigs < 2:
        return float("nan"), 1, n_eigs

    # 选择尾部样本数 k
    k_used = max(10, int(n_eigs * k_frac)) if k is None else int(k)
    k_used = max(1, min(k_used, n_eigs - 1))

    # Hill 估计
    eps_t = torch.tensor(eps, dtype=lam_sorted.dtype, device=lam_sorted.device)
    lam_ref = torch.clamp(lam_sorted[-k_used-1], min=eps_t)
    top = lam_sorted[-k_used:]
    denom = torch.log(top / lam_ref).sum().clamp_min(eps_t)
    alpha = float(1.0 + (k_used / float(denom)))
    return alpha, k_used, n_eigs

# =======================================
# ======== Public API (macro-aware) =====
# =======================================

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
    """
    计算 PL_Alpha_Hill（支持 BASELINE 与 FARMS）。
    - use_farms=None: 采用全局宏/环境变量；True/False: 强制指定。
    返回: (alpha, k_used, n_eigs)，其中 n_eigs 为用于估计的特征值数量（整矩阵或拼接后）
    """
    mode_farms = USE_FARMS if use_farms is None else bool(use_farms)

    if mode_farms:
        lam_sorted = _svd_eigs_farms(
            W, m_sub=farms_m_sub, n_sub=farms_n_sub,
            stride_m=farms_stride_m, stride_n=farms_stride_n,
            max_blocks=farms_max_blocks, seed=farms_seed
        )
    else:
        lam_sorted = _svd_eigs_baseline(W)

    if lam_sorted.numel() < 2:
        # 回退值：最小信息
        min_dim = min(W.shape[0], W.reshape(W.shape[0], -1).shape[1]) if W.ndim > 1 else 1
        return float("nan"), 1, int(min_dim)

    return _hill_alpha_from_sorted_eigs(lam_sorted, k=k, k_frac=k_frac, eps=eps)

def compute_alpha_values(model: nn.Module, cache_dir: Optional[str] = None,
                         *, use_farms: Optional[bool] = None) -> Dict[str, float]:
    """为模型中所有线性层计算 α 值；支持宏/参数切换 FARMS 或 BASELINE。"""
    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        # 把模式写入文件名，避免不同模式命中同一缓存
        mode_tag = "farms" if (USE_FARMS if use_farms is None else use_farms) else "baseline"
        cache_path = os.path.join(cache_dir, f"alpha_values_{mode_tag}.csv")
        if os.path.exists(cache_path):
            logger.info(f"Loading alpha values from cache: {cache_path}")
            return load_alpha_from_csv(cache_path)

    logger.info("Computing alpha values for all linear layers...")
    results: Dict[str, float] = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight = getattr(module, "weight", None)
            if weight is None:
                continue
            try:
                alpha, k_used, n_eigs = alpha_hill_from_weight(
                    weight.detach(),
                    use_farms=use_farms  # None=遵循宏；True/False=强制
                )
                results[name] = alpha
            except Exception as e:
                logger.warning(f"Failed to compute alpha for {name}: {e}")
                results[name] = float("nan")

    if cache_path:
        logger.info(f"Saving alpha values to: {cache_path}")
        save_alpha_to_csv(results, cache_path)

    return results

def save_alpha_to_csv(alpha_results: Dict[str, float], filename: str):
    """保存 α 结果到 CSV。"""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['layer_name', 'alpha'])
        for name, alpha in alpha_results.items():
            writer.writerow([name, alpha])

def load_alpha_from_csv(filename: str) -> Dict[str, float]:
    """从 CSV 读入 α 结果。"""
    alpha_results: Dict[str, float] = {}
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                alpha_results[row['layer_name']] = float(row['alpha'])
            except (ValueError, KeyError):
                continue
    return alpha_results

def build_and_solve_milp_for_layer(
    expert_alphas: Dict[int, float],
    candidate_bits: list,
    bpp_budget: float,
    gamma: float = 1.0,
) -> Dict[int, int]:
    """
    为核心层专家计算MILP最优位宽分配。
    
    参数:
        expert_alphas: {expert_idx: alpha_value} 每专家的alpha值
        candidate_bits: 候选位宽列表 [2,3,4,8]
        bpp_budget: 平均位宽预算
        gamma: 形状先验指数，默认1.0
    
    返回:
        assignment: {expert_idx: chosen_bit} 每专家分配到的位宽
    """
    if pulp is None:
        raise ImportError("PuLP is required for MILP optimization. Install with: pip install pulp")
    
    E = len(expert_alphas)
    if E == 0:
        return {}
    
    # 将expert_alphas转换为列表，保证顺序
    expert_indices = sorted(expert_alphas.keys())
    alphas = [expert_alphas[idx] for idx in expert_indices]
    
    # 候选位宽检查
    candidate_bits = [int(b) for b in candidate_bits]
    if any(b <= 0 for b in candidate_bits):
        raise ValueError("候选位宽必须为正整数。")
    min_bit = min(candidate_bits)
    if bpp_budget < min_bit:
        raise ValueError(f"预算不可行：bpp_budget={bpp_budget} 小于最小位宽 {min_bit}。")
    
    # 计算形状先验敏感度
    alpha0 = float(pd.Series(alphas).median())
    eps = 1e-8
    sensitivities = [((alpha0 / max(a, eps)) ** gamma) for a in alphas]
    
    # 构造代价表（无校准解析近似）
    q_b_scalar = {b: 2.0 ** (-2 * b) for b in candidate_bits}
    
    # 建立 MILP 模型
    prob = pulp.LpProblem("NoCalib_LayerExperts", pulp.LpMinimize)
    
    # 二元决策变量 x_{e,b}
    x = {}
    for i in range(E):
        for b in candidate_bits:
            x[(i, b)] = pulp.LpVariable(f"x_{i}_{b}", lowBound=0, upBound=1, cat=pulp.LpBinary)
    
    # 目标函数：sum_e sum_b x_{e,b} * s_e * q_{e,b}
    obj_terms = []
    for i in range(E):
        s_e = sensitivities[i]
        for b in candidate_bits:
            obj_terms.append(x[(i, b)] * (s_e * q_b_scalar[b]))
    prob += pulp.lpSum(obj_terms), "Total_Cost"
    
    # 约束1：每层恰好选择一个位宽
    for i in range(E):
        prob += pulp.lpSum([x[(i, b)] for b in candidate_bits]) == 1, f"one_bit_{i}"
    
    # 约束2：平均位宽预算
    prob += pulp.lpSum([x[(i, b)] * b for i in range(E) for b in candidate_bits]) <= bpp_budget * E, "bit_budget"
    
    # 求解
    solver = pulp.PULP_CBC_CMD(msg=False)
    status = prob.solve(solver)
    
    # 检查求解状态
    lp_status = pulp.LpStatus[status]
    if lp_status != "Optimal":
        raise RuntimeError(
            f"求解失败：LpStatus={lp_status}。"
            f"建议：提高 bpp_budget 或调整候选位宽范围。"
        )
    
    # 解析结果
    assignment = {}
    for i, expert_idx in enumerate(expert_indices):
        chosen_b = None
        for b in candidate_bits:
            if pulp.value(x[(i, b)]) >= 0.5:
                chosen_b = b
                break
        if chosen_b is None:
            raise RuntimeError(f"Expert {expert_idx} 未选定位宽。")
        assignment[expert_idx] = chosen_b
    
    return assignment



def get_model():
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    config = AutoConfig.from_pretrained(
        args.model, attn_implementation=args.attn_implementation
    )
    model = AutoModelForCausalLM.from_pretrained(args.model, config=config, device_map='cpu',torch_dtype=torch.float16)

    assert isinstance(
        model, MixtralForCausalLM), 'Successfully loaded `Mixtral` model!'
    model.seqlen = 2048
    return model


@torch.no_grad()
def mixtral_sequential(model, dataloader, dev, bit_config=None):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    
    # Compute alpha values if needed
    alpha_values = None
    if args.mixed_type == "mixed_with_alpha" or args.mixed_type == "no_calib_auto_programming":
        alpha_values = compute_alpha_values(model, cache_dir=args.cache_dir)
        print(f"Computed alpha values for {len(alpha_values)} layers")

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):

        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    print('Ready.')
    quantizers = {}
    # Dictionary to store bit assignments for each layer and expert
    # Format: {layer_idx: {expert_idx: bit_value}}
    layer_expert_bits = {}
    for i in range(len(layers)):

        print(f'Quantizing layer {i+1}/{len(layers)}..')
        print('+--------------------------------+------------+------------+------------+---------+')
        print('|              name              |weight_error| fp_inp_SNR | q_inp_SNR  |  time   |')
        print('+================================+============+============+============+=========+')

        layer = layers[i].to(dev)
        full = find_layers(layer)

        sequential = [list(full.keys())]
        
        # Initialize variables
        low_bit_experts = []
        high_bit_experts = []
        expert_bit_assignments = {}  # For no_calib_auto_programming mode
        
        # random generation
        if args.mixed_type == "random":
            import random
            numbers = list(range(8))
            low_bit_config = random.sample(numbers, 2)
            for num in low_bit_config:
                numbers.remove(num)
            high_bit_config = random.sample(numbers, 2)
            low_bit_experts = ["block_sparse_moe.experts."+str(j) for j in low_bit_config]   
            high_bit_experts = ["block_sparse_moe.experts."+str(j) for j in high_bit_config]
        elif args.mixed_type == "manual":
            if bit_config is not None:
                _, indices_max = torch.topk(bit_config[i], args.h_experts)
                _, indices_min = torch.topk(bit_config[i], args.l_experts, largest=False)
                low_bit_experts = ["block_sparse_moe.experts."+str(j.item()) for j in indices_min]   
                high_bit_experts = ["block_sparse_moe.experts."+str(j.item()) for j in indices_max]
            else:
                print("Please generate the high_experts.pkl and low_experts.pkl first!")
                exit()
        elif args.mixed_type == "mixed":
             if bit_config is not None:
                low_bit_experts = []
                high_bit_experts = []
                for expert_index in bit_config[i].keys():
                    if bit_config[i][expert_index] == 1:
                        low_bit_experts.append("block_sparse_moe.experts."+str(expert_index))
                    elif bit_config[i][expert_index] == 3:
                        high_bit_experts.append("block_sparse_moe.experts."+str(expert_index))
             else:
                print("Please generate the high_experts.pkl and low_experts.pkl first!")
                exit()
        elif args.mixed_type == "mixed_with_alpha":
            # Compute alpha values for expert modules in this layer
            expert_alpha_values = {}
            layer_name_prefix = f'model.layers.{i}.block_sparse_moe.experts.'
            
            # Get alpha values for all experts in this layer
            for expert_idx in range(8):
                expert_prefix = f'{layer_name_prefix}{expert_idx}'
                # Get alpha values for all three weight matrices (w1, w2, w3)
                alpha_sum = 0.0
                count = 0
                for weight_name in ['w1', 'w2', 'w3']:
                    weight_full_name = f'{expert_prefix}.{weight_name}'
                    if weight_full_name in alpha_values:
                        alpha_val = alpha_values[weight_full_name]
                        # Check if alpha is valid (not NaN and not infinite)
                        if isinstance(alpha_val, (int, float)) and alpha_val == alpha_val and abs(alpha_val) != float('inf'):
                            alpha_sum += alpha_val
                            count += 1
                
                if count > 0:
                    expert_alpha_values[expert_idx] = alpha_sum / count
                else:
                    # If no valid alpha found, use a default value
                    expert_alpha_values[expert_idx] = 0.0
            
            # Sort experts by alpha value (ascending: smaller alpha = more sensitive = higher precision)
            sorted_experts = sorted(expert_alpha_values.items(), key=lambda x: x[1])
            
            # Calculate number of experts for each precision
            total_experts = 8
            n_high_bit = int(total_experts * args.high_bit_experts_ratio)
            n_low_bit = int(total_experts * args.low_bit_experts_ratio)
            
            # Smaller alpha gets higher precision (wbits+1)
            high_bit_experts = ["block_sparse_moe.experts."+str(sorted_experts[j][0]) for j in range(n_high_bit)]
            # Larger alpha gets lower precision (wbits-1)
            low_bit_experts = ["block_sparse_moe.experts."+str(sorted_experts[total_experts - j - 1][0]) for j in range(n_low_bit)]
            
            print(f"Layer {i}: High bit experts (alpha_sorted): {[expert_alpha_values[int(e.split('.')[-1])] for e in high_bit_experts]}")
            print(f"Layer {i}: Low bit experts (alpha_sorted): {[expert_alpha_values[int(e.split('.')[-1])] for e in low_bit_experts]}")
        elif args.mixed_type == "no_calib_auto_programming":
            # Use MILP to automatically assign bit-widths based on alpha values
            try:
                # Parse candidate bits
                candidate_bits = [int(s.strip()) for s in args.milp_candidate_bits.split(",") if s.strip()]
                if not candidate_bits:
                    raise ValueError("Empty candidate bits list")
                
                # Compute alpha values for expert modules in this layer
                expert_alpha_values = {}
                layer_name_prefix = f'model.layers.{i}.block_sparse_moe.experts.'
                
                # Get alpha values for all experts in this layer
                for expert_idx in range(8):
                    expert_prefix = f'{layer_name_prefix}{expert_idx}'
                    # Get alpha values for all three weight matrices (w1, w2, w3)
                    alpha_sum = 0.0
                    count = 0
                    for weight_name in ['w1', 'w2', 'w3']:
                        weight_full_name = f'{expert_prefix}.{weight_name}'
                        if weight_full_name in alpha_values:
                            alpha_val = alpha_values[weight_full_name]
                            # Check if alpha is valid
                            if isinstance(alpha_val, (int, float)) and alpha_val == alpha_val and abs(alpha_val) != float('inf'):
                                alpha_sum += alpha_val
                                count += 1
                    
                    if count > 0:
                        expert_alpha_values[expert_idx] = alpha_sum / count
                    else:
                        # If no valid alpha found, use a default value
                        expert_alpha_values[expert_idx] = 0.0
                
                # Solve MILP for this layer
                expert_bit_assignments = build_and_solve_milp_for_layer(
                    expert_alphas=expert_alpha_values,
                    candidate_bits=candidate_bits,
                    bpp_budget=args.milp_bpp_budget,
                    gamma=args.milp_gamma
                )
                
                # Log the results
                print(f"Layer {i}: MILP bit assignment:")
                for expert_idx in sorted(expert_bit_assignments.keys()):
                    bit = expert_bit_assignments[expert_idx]
                    alpha_val = expert_alpha_values.get(expert_idx, 0.0)
                    print(f"  Expert {expert_idx}: {bit} bits (alpha={alpha_val:.4f})")
                
            except Exception as e:
                logger.error(f"Failed to solve MILP for layer {i}: {e}")
                # Fallback: use uniform quantization
                expert_bit_assignments = {idx: args.wbits for idx in range(8)}
                print(f"Layer {i}: MILP failed, falling back to uniform {args.wbits}-bit quantization")


        for names in sequential:
            subset = {n: full[n] for n in names}
            gptq = {}
            for name in subset:

                gptq[name] = GPTQ(subset[name], logger, name, args.wbits)

                if args.mixed_type == "uniform":
                    gptq[name].quantizer.configure(args.wbits, perchannel=True, sym=args.sym, mse=False, pack=args.pack) 
                    gptq[name].wbits = args.wbits
                elif args.mixed_type == "no_calib_auto_programming":
                    # Use MILP-assigned bit-widths
                    if name not in expert_modules:
                        gptq[name].quantizer.configure(args.attn_bits, perchannel=True, sym=args.sym, mse=False, pack=args.pack)
                        gptq[name].wbits = args.attn_bits
                    else:
                        # Extract expert index from name (e.g., "block_sparse_moe.experts.0.w1" -> 0)
                        try:
                            expert_name = name[:-3]  # Remove ".w1", ".w2", or ".w3"
                            expert_idx = int(expert_name.split('.')[-1])
                            assigned_bit = expert_bit_assignments.get(expert_idx, args.wbits)
                            gptq[name].quantizer.configure(assigned_bit, perchannel=True, sym=args.sym, mse=False, pack=args.pack)
                            gptq[name].wbits = assigned_bit
                        except (ValueError, KeyError):
                            # Fallback to default if parsing fails
                            gptq[name].quantizer.configure(args.wbits, perchannel=True, sym=args.sym, mse=False, pack=args.pack)
                            gptq[name].wbits = args.wbits
                else:
                    if name not in expert_modules:
                        gptq[name].quantizer.configure(args.attn_bits, perchannel=True, sym=args.sym, mse=False, pack=args.pack)
                        gptq[name].wbits = args.attn_bits
                    else:
                        if name[:-3] in high_bit_experts:
                            gptq[name].quantizer.configure(args.wbits+1, perchannel=True, sym=args.sym, mse=False, pack=args.pack)
                            gptq[name].wbits = args.wbits+1
                        elif name[:-3] in low_bit_experts:
                            gptq[name].quantizer.configure(args.wbits-1, perchannel=True, sym=args.sym, mse=False, pack=args.pack)
                            gptq[name].wbits = args.wbits-1
                        else:
                            gptq[name].quantizer.configure(args.wbits, perchannel=True, sym=args.sym, mse=False, pack=args.pack)
                            gptq[name].wbits = args.wbits
            # print(layer)
            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            for h in handles:
                h.remove()

            for name in subset:
                scale, zero, g_idx, error = gptq[name].fasterquant(percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, name=name)
                # quantizers['model.layers.%d.%s' % (i, name)] = (gptq[name].quantizer.cpu(), scale.cpu(), zero.cpu(), g_idx.cpu(), args.wbits, args.groupsize)
                quantizers['model.layers.%d.%s' % (i, name)] = None
                
                # Collect bit assignment information for experts
                if name in expert_modules:
                    try:
                        expert_name = name[:-3]  # Remove ".w1", ".w2", or ".w3"
                        expert_idx = int(expert_name.split('.')[-1])
                        if i not in layer_expert_bits:
                            layer_expert_bits[i] = {}
                        # Store the bit value (use wbits from gptq)
                        # Note: w1, w2, w3 should have the same bit, so we can overwrite
                        layer_expert_bits[i][expert_idx] = gptq[name].wbits
                    except (ValueError, KeyError):
                        pass  # Skip if parsing fails
                if args.pack:
                    # real quant for compact memory
                    quant_config = BaseQuantizeConfig(nbits=gptq[name].wbits, group_size=args.groupsize)
                    name_parts = name.split('.')
                    if len(name_parts) == 2: # atten layer
                        _module = getattr(layer, name_parts[-2])
                        linear_layer = getattr(_module, name_parts[-1])
                    else: 
                        experts = getattr(layer.block_sparse_moe, "experts")
                        _module = experts[int(name_parts[-2])]
                        linear_layer = getattr(_module, name_parts[-1])
                    quant_layer = QLinear(quant_config=quant_config, device=linear_layer.weight.device)
                    quant_layer.replace_quantized_weight(linear_layer.weight, scale, zero)
                    setattr(_module, name_parts[-1], quant_layer)
                    print(getattr(_module, name_parts[-1]).W_q.dtype)
                gptq[name].free()
            
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()
        inps, outs = outs, inps
        print('+--------------------------------+------------+------------+------------+---------+')
        print('\n')

    model.config.use_cache = use_cache
    
    # Save layer-expert bit assignments to file
    if args.save_bit_assignments and layer_expert_bits:
        bit_assignments_path = args.save_bit_assignments
        logger.info(f"Saving bit assignments to: {bit_assignments_path}")
        with open(bit_assignments_path, 'wb') as f:
            pickle.dump(layer_expert_bits, f)
        logger.info(f"Saved bit assignments for {len(layer_expert_bits)} layers")

    return quantizers


if __name__ == "__main__":
    import argparse

    def list_of_ints(arg):
        return list(map(int, arg.split(',')))
    
    def list_of_floats(arg):
        return list(map(float, arg.split(',')))

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model", type=str, help="model to load; for example `huggyllama/llama-7b`."
    )
    parser.add_argument(
        "--wbits",
        type=str,
        choices=["1bit", "2bit", "3bit", "4bit", "5bit", "6bit", "7bit", "8bit"],
        help="weight bit-width",
    )
    parser.add_argument(
        "--attn_bits",
        type=str,
        choices=["1bit", "2bit", "3bit", "4bit", "5bit", "6bit", "7bit", "8bit"],
        help="attention weight bit-width",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["wikitext2", "ptb", "c4", "gsm8k", "mix"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument(
        "--gsm8k_field",
        type=str,
        choices=["question", "answer"],
        default="question",
        help="For gsm8k, choose whether to use only the question or only the answer as calibration text.",
    )
    parser.add_argument("--load_quantized", action="store_true")
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration data samples."
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument(
        "--groupsize",
        type=int,
        default=128,
        help="Group size",
    )
    parser.add_argument(
        "--num_fewshot", 
        type=int, 
        default=0
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=1, 
        help="batch size."
    )
    parser.add_argument(
        "--attn_implementation",
        type=str, required=False, default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="attention implementation that the model works with",
    )
    parser.add_argument(
        '--sym', 
        action='store_true', 
        help='Whether to perform symmetric quantization.'
    )
    parser.add_argument(
        '--act-order', 
        action='store_true', 
        help='Whether to apply the activation order GPTQ heuristic'
    )
    parser.add_argument(
        "--multigpu",
        action="store_true",
    )
    parser.add_argument(
        "--eval_ppl", action="store_true", help="Evaluate perplexity."
    )
    parser.add_argument(
        "--tasks", 
        type=str,
        default="",
        help="Test datasets",
    )
    parser.add_argument(
        "--save",
        action="store_true",
    )
    parser.add_argument(
        "--pack", action="store_true", help="Whether to save the packed model."
    )
    parser.add_argument(
        "--use_flash_attention_2", action="store_true", help="Whether to use flash_attention2 for inference."
    )
    parser.add_argument(
        '--r', type=int, default=7, help='Number of experts to preserve'
    )
    parser.add_argument(
        "--mixed_type",
        type=str,
        choices=["uniform", "mixed", "random", "manual", "mixed_with_alpha", "no_calib_auto_programming"],
        help='Whether to use mixed-precision',
    )
    parser.add_argument(
        "--h_experts", 
        type=int, 
        default=2, 
        help="batch size." 
    )
    parser.add_argument(
        "--l_experts", 
        type=int, 
        default=2, 
        help="batch size." 
    )
    parser.add_argument(
        "--precisions", type=str, help="the file path of experts precision"
    )
    parser.add_argument(
        "--saving_path", type=str, help="the saving path of quantized model"
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None, help="Directory to cache alpha values"
    )
    parser.add_argument(
        "--high_bit_experts_ratio", type=float, default=0.25, help="Ratio of high bit experts (for mixed_with_alpha)"
    )
    parser.add_argument(
        "--low_bit_experts_ratio", type=float, default=0.25, help="Ratio of low bit experts (for mixed_with_alpha)"
    )
    parser.add_argument(
        "--milp_candidate_bits", type=str, default="2,3,4,8", 
        help="Candidate bit-widths for MILP (comma-separated, e.g., '2,3,4,8')"
    )
    parser.add_argument(
        "--milp_bpp_budget", type=float, default=3.5, 
        help="Average bit-width budget for MILP solver"
    )
    parser.add_argument(
        "--milp_gamma", type=float, default=1.0, 
        help="Gamma parameter for sensitivity weighting in MILP"
    )
    parser.add_argument(
        "--save_bit_assignments", type=str, default=None,
        help="Path to save layer-expert bit assignments (pickle format)"
    )

    args = parser.parse_args()
    print(f'Arguments: {args}')

    # Propagate gsm8k field selection via environment for datautils
    if args.dataset == "gsm8k":
        os.environ["GSM8K_FIELD"] = args.gsm8k_field

    groupsize = args.groupsize
    args.wbits = int(args.wbits[0])
    args.attn_bits = int(args.attn_bits[0])

    model = get_model()
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    bit_config = None
    if args.mixed_type == "manual" or args.mixed_type == "mixed":
        high_bit = args.precisions
        if os.path.exists(high_bit):
            with open(high_bit, 'rb') as file:
                bit_config = pickle.load(file)
        else:
            print("Please generate the high_experts.pkl and low_experts.pkl first!")
            exit()

    dataloader, testloader = get_loaders(
        args.dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        model=args.model,
        seqlen=model.seqlen,
    )
    device = "cuda:0"
    tick = time.time()
    quantizers = mixtral_sequential(model, dataloader, device, bit_config)
    print("quantization time:", time.time() - tick, "s")
    print(model)

    if args.eval_ppl:
        for dataset in ["wikitext2"]:#, "c4", "ptb"]:
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, seqlen=2048, model=args.model
            )
            print(dataset)
            from eval_ppl_utils import llama_eval
            t1 = time.time()
            llama_eval(model, testloader, device, dataset)
            print("Time: ", time.time() - t1)
    if args.save:
        # Calculate average bits for different mixed_type modes
        if args.mixed_type in ["manual", "mixed"] and args.precisions:
            average_bits = int(args.precisions[-9:-7])/8
            saving_path = args.saving_path + f"-atten_{args.attn_bits}-e_{average_bits}"
        elif args.mixed_type == "mixed_with_alpha":
            # For mixed_with_alpha, calculate average bits based on ratios
            avg_bit_ratio = (
                args.high_bit_experts_ratio * (args.wbits + 1) +
                args.low_bit_experts_ratio * (args.wbits - 1) +
                (1 - args.high_bit_experts_ratio - args.low_bit_experts_ratio) * args.wbits
            )
            saving_path = args.saving_path + f"-atten_{args.attn_bits}-e_{avg_bit_ratio:.2f}"
        elif args.mixed_type == "no_calib_auto_programming":
            # For MILP, use the budget as the average bits
            saving_path = args.saving_path + f"-atten_{args.attn_bits}-e_{args.milp_bpp_budget:.2f}"
        else:
            # For uniform or other modes, use wbits directly
            saving_path = args.saving_path + f"-atten_{args.attn_bits}-e_{args.wbits}"
        
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        tokenizer.save_pretrained(saving_path)
        from utils.pack import save_quantized
        save_quantized(model, saving_path)