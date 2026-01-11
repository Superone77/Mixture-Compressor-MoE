#!/usr/bin/env bash
set -euo pipefail

# 修改为你的模型路径
MODEL_PATH="/mnt/models/mistralai/Mixtral-8x7B-v0.1"

# 校准数据集：c4 或 math
CALIBRATION="c4"

# 保存精度分配结果和 CSV 的目录
SAVE_DIR="experts_mixture_bit_selection"
CSV_DIR="experts_mixture_bit_selection/csv"

# 1) 校准并生成专家激活频率/权重/量化损失
python awareness.py "${MODEL_PATH}" \
  --calibration "${CALIBRATION}" \
  --nsamples 128 \
  --batch_size 1

# 2) 精度分配并导出 CSV
python precision_solver.py \
  --actnum_path experts_act_frequency.pkl \
  --quant_loss_path experts_quant_loss.pkl \
  --weight_path experts_act_weight.pkl \
  --save_path "${SAVE_DIR}" \
  --csv_save_path "${CSV_DIR}" \
  --start_bitwidth 12 \
  --end_bitwidth 21
