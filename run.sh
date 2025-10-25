export CUDA_VISIBLE_DEVICES=1

Model_Path="/local/mnt2/workspace/wanqi/tmp/AI-ModelScope/Mixtral-8x7B-v0.1"
Saving_Path="/local/mnt2/workspace/wanqi/tmp/AI-ModelScope/Mixtral-8x7B-v0.1-2.5b"
Precision_Path="./experts_mixture_bit_selection/experts_mixture_bitwidth_combination_20bit.pkl"

# Test full precision model first
echo "Testing full precision Mixtral-8x7B on wikitext2..."

# Example 1: Use bf16 for attention layers (no quantization)
python main.py ${Model_Path} --wbits bf16 --attn_bits bf16 --dataset wikitext2 --groupsize 128 --eval_ppl --mixed_type uniform --precisions ${Precision_Path} --pack --save --saving_path ${Saving_Path}

# Example 2: Use 4bit quantization for attention layers
# python main.py ${Model_Path} --wbits 2bit --attn_bits 4bit --dataset wikitext2 --groupsize 128 --eval_ppl --mixed_type uniform --precisions ${Precision_Path} --pack --save --saving_path ${Saving_Path}
