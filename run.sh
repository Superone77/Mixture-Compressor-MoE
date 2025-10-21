export CUDA_VISIBLE_DEVICES=1

Model_Path="/local/mnt2/workspace/wanqi/tmp/AI-ModelScope/Mixtral-8x7B-v0.1"
Saving_Path="/local/mnt2/workspace/wanqi/tmp/AI-ModelScope/Mixtral-8x7B-v0.1-2.5b"
Precision_Path="./experts_mixture_bit_selection/experts_mixture_bitwidth_combination_20bit.pkl"
python main.py ${Model_Path} --wbits 2bit --attn_bits 4bit --dataset wikitext2 --groupsize 128 --eval_ppl --mixed_type uniform --precisions ${Precision_Path} --pack --save --saving_path ${Saving_Path}
