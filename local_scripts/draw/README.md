# Experiment Scripts for MoE Quantization Analysis

This directory contains scripts for running experiments described in `experiment_0.md`.

## Experiments

### Experiment 1.1: Expert Activation Heatmaps

**Goal**: Prove that calibration data fails to activate deep-layer experts.

**Script**: `experiment_1_1_expert_activation.py`

**Usage**:
```bash
python local_scripts/draw/experiment_1_1_expert_activation.py \
    --model mistralai/Mixtral-8x7B-v0.1 \
    --output experiment_1_1_activation_rates.csv \
    --device cuda:0 \
    --nsamples 128 \
    --seed 0
```

**Output**: CSV file with columns:
- `layer`: Layer index
- `expert_id`: Expert ID (0-7)
- `dataset`: Dataset name (wikitext2 or gsm8k)
- `activation_rate`: Activation rate (0.0-1.0)
- `activation_count`: Number of activations
- `total_tokens`: Total tokens processed

### Experiment 1.2: Cross-Domain Performance Drop

**Goal**: Prove that data-driven calibration overfits to the calibration domain.

**Script**: `experiment_1_2_cross_domain.py`

**Usage**:
```bash
python local_scripts/draw/experiment_1_2_cross_domain.py \
    --model mistralai/Mixtral-8x7B-v0.1 \
    --output experiment_1_2_perplexity.csv \
    --device cuda:0 \
    --nsamples 128 \
    --wbits 4bit \
    --attn_bits 4bit \
    --mixed_type uniform \
    --sym \
    --pack
```

**Output**: CSV file with columns:
- `model`: Model name (Model_Wiki or Model_GSM)
- `calibration_dataset`: Dataset used for calibration
- `test_dataset`: Dataset used for testing
- `perplexity`: Perplexity score

**Note**: This experiment requires quantizing the model twice (once with WikiText2, once with GSM8K), which can be time-consuming. You can use `--skip_quantization` with `--model_wiki_path` and `--model_gsm_path` to load pre-quantized models.

## Visualization

**Script**: `visualize_experiments.py`

**Usage**:
```bash
# Visualize both experiments
python local_scripts/draw/visualize_experiments.py \
    --experiment both \
    --csv_1_1 experiment_1_1_activation_rates.csv \
    --csv_1_2 experiment_1_2_perplexity.csv

# Visualize only Experiment 1.1
python local_scripts/draw/visualize_experiments.py \
    --experiment 1.1 \
    --csv_1_1 experiment_1_1_activation_rates.csv

# Visualize only Experiment 1.2
python local_scripts/draw/visualize_experiments.py \
    --experiment 1.2 \
    --csv_1_2 experiment_1_2_perplexity.csv
```

**Output**: 
- `experiment_1_1_heatmap.png`: Heatmap showing expert activation rates per layer
- `experiment_1_1_bar.png`: Bar chart showing average activation rate per layer
- `experiment_1_2_performance.png`: Grouped bar chart showing perplexity comparison

## Dependencies

- torch
- transformers
- pandas
- matplotlib
- seaborn
- tqdm
- datasets

## Notes

1. **Memory Requirements**: These experiments require significant GPU memory. For Mixtral-8x7B, you'll need at least 24GB GPU memory.

2. **Time Requirements**: 
   - Experiment 1.1: ~30-60 minutes depending on number of samples
   - Experiment 1.2: ~2-4 hours (requires quantizing model twice)

3. **Data**: The scripts use the `datautils.py` module to load datasets. Make sure you have internet connection for first-time dataset downloads.

4. **Model Loading**: For Experiment 1.2, if you want to skip quantization and use pre-quantized models, you need to save them first using `main.py` with `--save` flag.

