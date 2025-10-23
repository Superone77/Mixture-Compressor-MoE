#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

Model_Path="/local/mnt2/workspace/wanqi/tmp/AI-ModelScope/Mixtral-8x7B-v0.1"

echo "Testing full precision Mixtral-8x7B on wikitext2..."
python test_full_precision.py ${Model_Path} --dataset wikitext2 --batch_size 1
