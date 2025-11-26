#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
No-Calib Mixed-Precision Bit Allocation (MILP, α-guided)
=======================================================
功能：
  - 无校准场景（No-Calib）：只用每层的 tail index α（已预先计算并存入 CSV）
  - 各层参数量一致：平均位宽预算 bpp 直接约束 sum(bits)/E <= bpp
  - 按“α 形状先验”分配离散位宽，使敏感度加权的量化代价最小（ILP 精确求解）

输入 CSV（至少两列）：
  layer, alpha [, ce]
  - layer: 层/专家标识（字符串）
  - alpha: 该层的幂律尾指数 α（浮点数；越小越 heavy-tailed）
  - ce (可选): 解析代价中的缩放系数 c_e；若缺省则取 1.0
    （c_e 典型取值：Var(W_e) 或 ||W_e||_F^2 / N_e，用于层间归一化）

目标函数：
  min_{x_{e,b}} sum_e sum_b x_{e,b} * s_e * q_{e,b}
  其中：
    - s_e = (alpha0 / alpha_e)^gamma   # 形状先验敏感度（alpha0 为 α 的全局中位数）
    - q_{e,b} = c_e * 2^{-2b}          # 均匀量化的解析误差近似（无校准）
    - x_{e,b} ∈ {0,1} 代表层 e 是否选择位宽 b

约束：
  1) 每层恰好选择一个位宽： sum_b x_{e,b} = 1
  2) 平均位宽预算：          (1/E) * sum_e sum_b x_{e,b} * b <= bpp  ⟹  sum_e sum_b ... <= bpp * E

依赖：
  pip install pandas pulp
  （PuLP 默认带 CBC 求解器；若环境无 CBC，可安装 coin-or-cbc，或切换到已安装的商业求解器）

用法示例：
  python no_calib_milp.py \
    --csv alphas.csv \
    --bits 2,3,4,8 \
    --bpp 3.5 \
    --gamma 1.0 \
    --out bit_assignments.csv
"""

import argparse
import pandas as pd
from pathlib import Path

try:
    import pulp  # MILP 建模与求解器接口（默认 CBC）
except ImportError as e:
    raise SystemExit("请先安装依赖：pip install pulp pandas") from e


def build_and_solve_milp(
    df: pd.DataFrame,
    candidate_bits,
    bpp_budget: float,
    gamma: float = 1.0,
    alpha_col: str = "alpha",
    layer_col: str = "layer",
    ce_col: str = "ce",
    solver_time_limit: int = None,
):
    """
    核心求解函数：构建并求解无校准场景的混精 MILP。

    参数：
      df            : 读入 CSV 的 DataFrame（至少包含 layer_col、alpha_col）
      candidate_bits: 候选位宽列表（如 [2,3,4,8]），必须是正整数
      bpp_budget    : 平均位宽预算（bits-per-parameter，因各层等参，等价为平均每层位宽）
      gamma         : 形状先验的指数 γ，s_e = (alpha0 / alpha_e)^gamma
      alpha_col     : α 的列名（默认 "alpha"）
      layer_col     : 层名列名（默认 "layer"）
      ce_col        : c_e 的列名（可选；缺省或列不存在则 c_e=1.0）
      solver_time_limit: CBC 求解的时间上限（秒），可选

    返回：
      assignment      : dict[layer_name -> chosen_bit]  每层分配到的位宽
      objective_value : 目标函数值（敏感度加权代价之和）
      actual_avg_bits : 实际得到的平均位宽（应 ≤ bpp_budget）
    """
    # ----------- 基本列检查 -----------
    for col in [layer_col, alpha_col]:
        if col not in df.columns:
            raise ValueError(f"CSV 缺少必要列 `{col}`。")

    # ----------- 读取基础数据 -----------
    layers = df[layer_col].astype(str).tolist()
    alphas = df[alpha_col].astype(float).tolist()
    E = len(layers)
    if E == 0:
        raise ValueError("CSV 为空或未解析到任何层。")

    # 若提供 ce 列则使用，否则统一取 1.0（无校准解析近似的缩放因子）
    if ce_col in df.columns:
        ces = df[ce_col].astype(float).tolist()
    else:
        ces = [1.0] * E

    # ----------- 候选位宽检查（基础可行性） -----------
    # 预算只提供“上界”：avg_bits <= bpp_budget
    # 因每层必须选择一个位宽 -> 最小可达的平均位宽为 min(candidate_bits)
    # 若 bpp_budget < min_bits，则问题一定不可行（提前报错）
    candidate_bits = [int(b) for b in candidate_bits]
    if any(b <= 0 for b in candidate_bits):
        raise ValueError("候选位宽必须为正整数。")
    min_bit = min(candidate_bits)
    max_bit = max(candidate_bits)
    if bpp_budget < min_bit:
        raise ValueError(
            f"预算不可行：bpp_budget={bpp_budget} 小于可用的最小位宽 {min_bit}。"
            f"请提高预算或提供更小的候选位宽。"
        )

    # ----------- 计算形状先验敏感度 s_e -----------
    # 采用 α 的全局中位数做归一：alpha0 = median({alpha_e})
    alpha0 = float(pd.Series(alphas).median())

    # α 理论上 > 0；数值上保底到一个很小正数，防止极端数据造成除零
    eps = 1e-8
    sensitivities = [((alpha0 / max(a, eps)) ** gamma) for a in alphas]

    # ----------- 构造代价表 q_{e,b}（解析近似，无校准） -----------
    # 均匀量化：MSE ~ Δ^2，Δ ∝ 2^{-b}，故 q_{e,b} = c_e * 2^{-2b}
    # 说明：这里不区分层的离散化误差模型差异，统一用解析式；若将来扩展到 Calib，可替换为 GPTQ 扫描表。
    q_b_scalar = {b: 2.0 ** (-2 * b) for b in candidate_bits}

    # ----------- 建立 MILP 模型 -----------
    prob = pulp.LpProblem("NoCalib_MixedPrecision", pulp.LpMinimize)

    # 二元决策变量 x_{e,b} ∈ {0,1}：层 e 是否选择位宽 b
    x = {}
    for i, layer in enumerate(layers):
        for b in candidate_bits:
            x[(i, b)] = pulp.LpVariable(
                f"x_{i}_{b}", lowBound=0, upBound=1, cat=pulp.LpBinary
            )

    # 目标函数：sum_e sum_b x_{e,b} * s_e * q_{e,b}
    # 其中 q_{e,b} = c_e * 2^{-2b}
    obj_terms = []
    for i in range(E):
        s_e = sensitivities[i]
        c_e = ces[i]
        for b in candidate_bits:
            obj_terms.append(x[(i, b)] * (s_e * (c_e * q_b_scalar[b])))
    prob += pulp.lpSum(obj_terms), "Total_Sensitivity_Weighted_Cost"

    # 约束 1：每层恰好选择一个位宽  sum_b x_{e,b} = 1
    for i in range(E):
        prob += pulp.lpSum([x[(i, b)] for b in candidate_bits]) == 1, f"one_bit_per_layer_{i}"

    # 约束 2：平均位宽预算  sum_e sum_b x_{e,b}*b <= bpp_budget * E
    prob += pulp.lpSum([x[(i, b)] * b for i in range(E) for b in candidate_bits]) <= bpp_budget * E, "bit_budget"

    # 可选：设置求解时间上限（适合特别大的 E 或位宽集合较多时）
    solver = pulp.PULP_CBC_CMD(msg=True)
    if solver_time_limit is not None:
        try:
            solver.timeLimit = solver_time_limit
        except Exception:
            pass  # 不同版本 PuLP/求解器可能无此属性，忽略即可

    # ----------- 求解 -----------
    status = prob.solve(solver)

    # 状态检查：Optimal 正常；Infeasible/Undefined/Unbounded 等需要调参数
    lp_status = pulp.LpStatus[status]
    if lp_status != "Optimal":
        raise RuntimeError(
            f"求解失败：LpStatus={lp_status}。"
            f"建议：提高 bpp_budget、减少层数、或扩大候选位宽范围。"
        )

    # ----------- 解析解并返回 -----------
    assignment = {}
    total_bits = 0.0
    for i, layer in enumerate(layers):
        chosen_b = None
        for b in candidate_bits:
            # PuLP 变量值可能出现 0/1 的浮点近似（如 0.999999），用阈值判断
            if pulp.value(x[(i, b)]) >= 0.5:
                chosen_b = b
                break
        if chosen_b is None:
            # 理论上不应发生（因为有“恰好一个位宽”约束）
            raise RuntimeError(f"层 {layer} 未选定位宽，可能是数值问题。")
        assignment[layer] = chosen_b
        total_bits += chosen_b

    avg_bits = total_bits / E
    objective_value = pulp.value(prob.objective)

    return assignment, objective_value, avg_bits


def main():
    # ---------- 命令行参数 ----------
    parser = argparse.ArgumentParser(
        description="No-Calib MoE Mixed-Precision (MILP) via α-guided sensitivity"
    )
    parser.add_argument("--csv", type=str, required=True,
                        help="包含列: layer, alpha [, ce] 的 CSV 路径。")
    parser.add_argument("--bits", type=str, required=True,
                        help="候选位宽，逗号分隔，如 '2,3,4,8'。")
    parser.add_argument("--bpp", type=float, required=True,
                        help="平均位宽预算（bits-per-parameter, 各层等参时即平均每层位宽）。")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="形状先验指数 γ，s_e=(alpha0/alpha_e)^γ；默认 1.0。")
    parser.add_argument("--alpha-col", type=str, default="alpha",
                        help="α 列名，默认 'alpha'。")
    parser.add_argument("--layer-col", type=str, default="layer",
                        help="层名列名，默认 'layer'。")
    parser.add_argument("--ce-col", type=str, default="ce",
                        help="c_e 列名，默认 'ce'；若缺失则使用 1.0。")
    parser.add_argument("--out", type=str, default="bit_assignments.csv",
                        help="输出指派结果的 CSV 文件名；默认 'bit_assignments.csv'。")
    parser.add_argument("--time-limit", type=int, default=None,
                        help="CBC 求解时间上限（秒，可选）。")
    args = parser.parse_args()

    # ---------- 解析候选位宽 ----------
    try:
        candidate_bits = [int(s) for s in args.bits.split(",") if s.strip() != ""]
    except Exception:
        raise ValueError("无法解析 --bits，请使用逗号分隔的正整数，如 --bits 2,3,4,8")
    if not candidate_bits:
        raise ValueError("候选位宽为空，请提供 --bits 2,3,4,8 之类的参数。")

    # ---------- 读取 CSV ----------
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到输入 CSV：{csv_path}")
    df = pd.read_csv(csv_path)

    # ---------- 求解 ----------
    assignment, obj, avg_bits = build_and_solve_milp(
        df=df,
        candidate_bits=candidate_bits,
        bpp_budget=args.bpp,
        gamma=args.gamma,
        alpha_col=args.alpha_col,
        layer_col=args.layer_col,
        ce_col=args.ce_col,
        solver_time_limit=args.time_limit,
    )

    # ---------- 写出结果 ----------
    out_df = pd.DataFrame({
        "layer": list(assignment.keys()),
        "assigned_bit": list(assignment.values())
    }).sort_values("layer")
    out_df.to_csv(args.out, index=False)

    # ---------- 控制台摘要 ----------
    print("\n=== Solution Summary ===")
    print(f"Objective value : {obj:.6g}")
    print(f"Average bits    : {avg_bits:.4f} (budget={args.bpp})")
    print(f"Layers          : {len(assignment)}")
    print(f"Output saved to : {args.out}")


if __name__ == "__main__":
    main()
