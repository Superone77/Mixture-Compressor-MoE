# Qwen3-Coder-Next AlphaQ 混精 + GPTQ

本目录提供基于 AlphaQ bit recipe 对 Qwen3-Coder-Next 做 GPTQ 的脚本与推理脚本，支持多 GPU（`device_map="auto"`）。

## 1. Bit Recipe（CSV）

`qwen3_bit_recipes/` 下为 **gamma=10.0** 时平均位宽 **3 / 3.5 / 4** 的 bit 分配结果（由 AlphaQ MILP 生成）：

- `qwen3_coder_next_gamma10.0_bpp3.0.csv`
- `qwen3_coder_next_gamma10.0_bpp3.5.csv`
- `qwen3_coder_next_gamma10.0_bpp4.0.csv`

格式：`name,bit_width`。支持 **gate/up 分 bit**：`...gate_up_proj.expert_{e}.gate`、`...gate_up_proj.expert_{e}.up`；也支持整块 `...gate_up_proj.expert_{e}`。down 仍为 `...down_proj.expert_{e}`。

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

输出目录为 **量化再反量化** 的完整模型：结构与原模型一致，权重为 float（反量化后的值），可直接 `from_pretrained` 加载，无需专用量化推理逻辑。

## 3. 推理

推理脚本**同时支持**原模型（HuggingFace id）与 GPTQ 输出目录，加载方式相同：

```bash
# 使用 GPTQ 输出目录
python inference_qwen3_quantized.py --model_path ./out_qwen3_bpp4 --prompt "Your prompt" --max_new_tokens 64

# 使用原模型
python inference_qwen3_quantized.py --model_path Qwen/Qwen3-Coder-Next --prompt "Your prompt" --max_new_tokens 64
```

或直接用 HuggingFace 加载：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("./out_qwen3_bpp4", device_map="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("./out_qwen3_bpp4", trust_remote_code=True)
```

## 4. 说明

- 当前脚本对 **MoE 专家** 按 recipe 做 GPTQ：支持 gate/up 分块（`.gate`、`.up` 两行）或整块 `gate_up_proj`，以及 `down_proj`；其他线性层可后续按同一 CSV 扩展。
- 保存的是 **量化再反量化** 的模型，除权重大小外与原模型完全一致，推理用 `from_pretrained` 即可。
- 校准数据默认 **WikiText-2** train；校准与推理均支持 `device_map="auto"` 多 GPU。
- Recipe 由 qwen3-coder-next 仓库的 `run_alpha_qwen3_full.py`（gate/up 分条）+ `run_bit_allocation_qwen3.py` 生成后，将 CSV 放入 `qwen3_bit_recipes/` 即可运行 GPTQ。
