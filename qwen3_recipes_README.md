# Qwen3-Coder-Next AlphaQ 混精 + GPTQ

本目录提供基于 AlphaQ bit recipe 对 Qwen3-Coder-Next 做 GPTQ 的脚本与推理脚本，支持多 GPU（`device_map="auto"`）。

## 1. Bit Recipe（CSV）

`qwen3_bit_recipes/` 下为 **gamma=10.0** 时平均位宽 **3 / 3.5 / 4** 的 bit 分配结果（由 AlphaQ MILP 生成）：

- `qwen3_coder_next_gamma10.0_bpp3.0.csv`
- `qwen3_coder_next_gamma10.0_bpp3.5.csv`
- `qwen3_coder_next_gamma10.0_bpp4.0.csv`

格式：`name,bit_width`，其中 `name` 为完整模块路径（如 `model.model.layers.0.mlp.experts.down_proj.expert_0`）。

## 2. GPTQ（按 recipe 量化）

在**有 GPU 的机器**上运行：

```bash
# 安装依赖：transformers, torch, datasets, 以及本仓库 gptq 依赖（utils.quantizer_moe 等）
cd Mixture-Compressor-MoE

# 示例：4 bit 预算，128 条校准样本，多 GPU
python qwen3_gptq_from_recipe.py \
  --model Qwen/Qwen3-Coder-Next \
  --recipe_csv qwen3_bit_recipes/qwen3_coder_next_gamma10.0_bpp4.0.csv \
  --output_dir ./out_qwen3_bpp4 \
  --nsamples 128 \
  --seqlen 2048 \
  --device_map auto
```

输出目录 `output_dir` 内含：`config.json`、`model.safetensors`（或 `pytorch_model.bin`）、tokenizer 相关文件，可直接用于推理。

## 3. 推理

使用保存后的量化模型做生成（支持多 GPU）：

```bash
python inference_qwen3_quantized.py \
  --model_path ./out_qwen3_bpp4 \
  --device_map auto \
  --prompt "Your prompt here" \
  --max_new_tokens 64
```

或直接用 HuggingFace 加载：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("./out_qwen3_bpp4", device_map="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("./out_qwen3_bpp4", trust_remote_code=True)
```

## 4. 说明

- 当前脚本仅对 **MoE 专家**（`gate_up_proj` / `down_proj` 的每个 expert 切片）按 recipe 做 GPTQ；其他线性层可后续按同一 CSV 扩展。
- 校准数据默认使用 **WikiText-2** train；校准与推理均支持 `device_map="auto"` 多 GPU。
- Recipe 在无 GPU 机器上由 `run_bit_allocation_qwen3.py`（见上层 qwen3-coder-next 仓库）生成后，将 CSV 放入 `qwen3_bit_recipes/` 即可在本仓库运行 GPTQ 与推理。
