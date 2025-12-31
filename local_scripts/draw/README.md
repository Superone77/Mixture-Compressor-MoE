# Experiment Scripts for MoE Calibration Analysis

This directory contains scripts for running experiments described in `experiment_0.md`.

## Experiments

### Experiment 1.1: Expert Activation Heatmaps

**Goal**: Prove that calibration data fails to activate deep-layer experts.

**Script**: `experiment_1_1_expert_activation.py`

**Usage**:
```bash
python local_scripts/draw/experiment_1_1_expert_activation.py \
    --model deepseek-ai/DeepSeek-V2-Lite \
    --output_csv experiment_1_1_expert_activation.csv \
    --nsamples 128 \
    --seed 0 \
    --seqlen 2048 \
    --device cuda
```

**Output**: CSV file with columns:
- `dataset`: Dataset name (wikitext2, gsm8k)
- `layer`: Layer index
- `expert_id`: Expert ID (0-63)
- `utilization_rate`: Expert utilization rate (%)

### Experiment 1.2: Cross-Domain Performance Drop

**Goal**: Prove that data-driven calibration overfits to the calibration domain.

**Script**: `experiment_1_2_cross_domain.py`

**Usage**:
```bash
# First run: Quantize and evaluate (this will take a while)
python local_scripts/draw/experiment_1_2_cross_domain.py \
    --model deepseek-ai/DeepSeek-V2-Lite \
    --output_csv experiment_1_2_cross_domain.csv \
    --model_cache_dir ./quantized_models \
    --nsamples 128 \
    --seed 0 \
    --seqlen 2048 \
    --wbits 4bit \
    --attn_bits 4bit \
    --mixed_type uniform \
    --device cuda

# If models are already quantized, skip quantization:
python local_scripts/draw/experiment_1_2_cross_domain.py \
    --model deepseek-ai/DeepSeek-V2-Lite \
    --output_csv experiment_1_2_cross_domain.csv \
    --model_cache_dir ./quantized_models \
    --skip_quantization \
    --device cuda
```

**Output**: CSV file with columns:
- `model`: Model name (Model_Wiki or Model_GSM)
- `calibration_dataset`: Dataset used for calibration (wikitext2 or gsm8k)
- `test_dataset`: Dataset used for evaluation (wikitext2 or gsm8k)
- `perplexity`: Perplexity score

## Visualization

**Script**: `visualize_experiments.py`

**Usage**:
```bash
# Visualize both experiments
python local_scripts/draw/visualize_experiments.py \
    --experiment both \
    --csv_1_1 experiment_1_1_expert_activation.csv \
    --csv_1_2 experiment_1_2_cross_domain.csv \
    --output_dir ./figures

# Visualize only Experiment 1.1
python local_scripts/draw/visualize_experiments.py \
    --experiment 1.1 \
    --csv_1_1 experiment_1_1_expert_activation.csv \
    --output_dir ./figures

# Visualize only Experiment 1.2
python local_scripts/draw/visualize_experiments.py \
    --experiment 1.2 \
    --csv_1_2 experiment_1_2_cross_domain.csv \
    --output_dir ./figures
```

**Output**:
- `experiment_1_1_heatmap.png`: Heatmap showing expert utilization rates by layer
- `experiment_1_1_bar_chart.png`: Bar chart showing average utilization and zero-activation experts per layer
- `experiment_1_2_bar_chart.png`: Grouped bar chart comparing model performance across test datasets
- `experiment_1_2_heatmap.png`: Heatmap showing performance matrix

## Dependencies

Make sure you have the following packages installed:
- torch
- transformers
- pandas
- matplotlib
- seaborn
- tqdm
- datasets (for data loading)

## Notes

1. **Experiment 1.1** requires running inference on the model, which may take some time depending on the number of samples and model size.

2. **Experiment 1.2** requires quantizing the model twice (once with WikiText2, once with GSM8K), which can take a significant amount of time. The quantized models are cached in `--model_cache_dir` so you can skip quantization on subsequent runs using `--skip_quantization`.

3. Both experiments output CSV files that can be analyzed separately or visualized using the visualization script.

4. The visualization script creates publication-ready figures with proper styling and labels.

