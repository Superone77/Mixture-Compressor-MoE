#!/bin/bash
###############################################################################
# Mixtral Quantization Script Examples
###############################################################################
# This script demonstrates various quantization modes for Mixtral-8x7B:
# 1. uniform: All experts at same precision
# 2. mixed_with_alpha: Based on alpha-Hill estimation (ratio-based)
#    - BASELINE: Full SVD on entire weight matrices
#    - FARMS: Fixed Aspect Ratio Matrix Sampling (sub-matrix approach)
# 3. manual: Pre-defined precision from file
# 4. random: Random expert selection
# 5. no_calib_auto_programming: MILP-based automatic bit-width assignment
#    - Uses ILP solver to optimize bit-width allocation per expert
#    - Minimizes sensitivity-weighted quantization cost under bit budget
# 
# Alpha-based quantization uses Hill estimator to compute alpha values for
# each expert's weight matrices, then allocates higher precision to experts
# with lower alpha values (more sensitive to quantization).
###############################################################################

export CUDA_VISIBLE_DEVICES=1

Model_Path="/local/mnt2/workspace/wanqi/tmp/AI-ModelScope/Mixtral-8x7B-v0.1"
Saving_Path="/local/mnt2/workspace/wanqi/tmp/AI-ModelScope/Mixtral-8x7B-v0.1-2.5b"
Precision_Path="./experts_mixture_bit_selection/experts_mixture_bitwidth_combination_20bit.pkl"
Cache_Dir="./alpha_cache"  # Directory to cache alpha values

# ============================================
# Example 1: Uniform quantization (baseline)
# ============================================
echo "Example 1: Uniform quantization with bf16 for attention layers"
# python main.py ${Model_Path} --wbits bf16 --attn_bits bf16 --dataset wikitext2 --groupsize 128 --eval_ppl --mixed_type uniform --precisions ${Precision_Path} --pack --save --saving_path ${Saving_Path}

# ============================================
# Example 2: Mixed precision with alpha (BASELINE mode)
# ============================================
echo "Example 2: Mixed precision with alpha using BASELINE SVD mode"
# python main.py ${Model_Path} --wbits 2bit --attn_bits 4bit --dataset wikitext2 --groupsize 128 --eval_ppl --mixed_type mixed_with_alpha --cache_dir ${Cache_Dir} --high_bit_experts_ratio 0.25 --low_bit_experts_ratio 0.25 --pack --save --saving_path ${Saving_Path}

# ============================================
# Example 3: Mixed precision with alpha (FARMS mode)
# ============================================
echo "Example 3: Mixed precision with alpha using FARMS sub-matrix sampling"
# export ALPHA_MODE=FARMS
# export FARMS_M_SUB=128
# export FARMS_N_SUB=128
# export FARMS_MAX_BLOCKS=256
# python main.py ${Model_Path} --wbits 2bit --attn_bits 4bit --dataset wikitext2 --groupsize 128 --eval_ppl --mixed_type mixed_with_alpha --cache_dir ${Cache_Dir} --high_bit_experts_ratio 0.25 --low_bit_experts_ratio 0.25 --pack --save --saving_path ${Saving_Path}

# ============================================
# Example 4: Custom FARMS configuration
# ============================================
echo "Example 4: Custom FARMS configuration with different sub-matrix sizes"
# export ALPHA_MODE=FARMS
# export FARMS_M_SUB=256
# export FARMS_N_SUB=256
# export FARMS_STRIDE_M=128
# export FARMS_STRIDE_N=128
# export FARMS_MAX_BLOCKS=512
# python main.py ${Model_Path} --wbits 2bit --attn_bits 4bit --dataset wikitext2 --groupsize 128 --eval_ppl --mixed_type mixed_with_alpha --cache_dir ${Cache_Dir} --high_bit_experts_ratio 0.3 --low_bit_experts_ratio 0.2 --pack --save --saving_path ${Saving_Path}

# ============================================
# Example 5: Manual/Pre-defined precision
# ============================================
echo "Example 5: Manual precision from pre-computed expert precision file"
# python main.py ${Model_Path} --wbits 2bit --attn_bits 4bit --dataset wikitext2 --groupsize 128 --eval_ppl --mixed_type manual --precisions ${Precision_Path} --h_experts 2 --l_experts 2 --pack --save --saving_path ${Saving_Path}

# ============================================
# Example 6: Random expert selection
# ============================================
echo "Example 6: Random expert selection for mixed precision"
# python main.py ${Model_Path} --wbits 2bit --attn_bits 4bit --dataset wikitext2 --groupsize 128 --eval_ppl --mixed_type random --pack --save --saving_path ${Saving_Path}

# ============================================
# Example 7: No-calibration MILP auto-programming
# ============================================
echo "Example 7: MILP-based automatic bit-width assignment without calibration"
# python main.py ${Model_Path} --wbits 2bit --attn_bits 4bit --dataset wikitext2 --groupsize 128 --eval_ppl --mixed_type no_calib_auto_programming --cache_dir ${Cache_Dir} --milp_candidate_bits 2,3,4,8 --milp_bpp_budget 3.5 --milp_gamma 1.0 --pack --save --saving_path ${Saving_Path}

# ============================================
# Notes:
# ============================================
# - ALPHA_MODE: Controls SVD computation mode (BASELINE or FARMS)
# - FARMS_M_SUB/FARMS_N_SUB: Sub-matrix dimensions for FARMS
# - FARMS_STRIDE_M/FARMS_STRIDE_N: Stride for sub-matrix sampling
# - FARMS_MAX_BLOCKS: Maximum number of sub-matrices to sample
# - cache_dir: Directory to cache alpha values (reused across runs)
# - high_bit_experts_ratio: Ratio of experts to quantize at higher precision
# - low_bit_experts_ratio: Ratio of experts to quantize at lower precision
# - milp_candidate_bits: Candidate bit-widths for MILP solver (comma-separated)
# - milp_bpp_budget: Average bit-width budget for MILP solver
# - milp_gamma: Gamma parameter for sensitivity weighting in MILP (default: 1.0)
