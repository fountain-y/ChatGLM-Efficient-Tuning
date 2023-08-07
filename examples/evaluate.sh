#!/bin/bash

CUDA_VISIBLE_DEVICES=5 python ../src/train_bash.py \
    --stage sft \
    --do_eval \
    --dataset alpaca_gpt4_zh \
    --dataset_dir ../data \
    --checkpoint_dir ../work_dirs/model/chatglm2-6b \
    --output_dir ../work_dirs/alpaca_gpt4_zh/predict_raw \
    --overwrite_cache \
    --per_device_eval_batch_size 8 \
    --max_samples 50 \
    --predict_with_generate
