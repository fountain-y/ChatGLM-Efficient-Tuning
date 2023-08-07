#!/bin/bash

# accelerate launch --config_file accelerate_config.yaml ../src/train_sft.py \
#     --do_train \
#     --dataset alpaca_gpt4_zh \
#     --dataset_dir ../data \
#     --finetuning_type full \
#     --output_dir path_to_sft_checkpoint \
#     --overwrite_cache \
#     --per_device_train_batch_size 4 \
#     --gradient_accumulation_steps 4 \
#     --lr_scheduler_type cosine \
#     --logging_steps 10 \
#     --save_steps 1000 \
#     --learning_rate 5e-5 \
#     --num_train_epochs 3.0 \
#     --plot_loss \
#     --fp16

CUDA_VISIBLE_DEVICES=4 accelerate launch --config_file ./examples/accelerate_config_new.yaml src/train_sft.py \
    --model_name_or_path /data/yuanrz/model/chatglm2-6b \
    --use_v2 True \
    --do_train \
    --dataset covid_train,covid_dev \
    --dataset_dir work_dirs/data/covid \
    --finetuning_type full \
    --output_dir work_dirs/covid_lora_sft_full \
    --overwrite_cache \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --fp16
