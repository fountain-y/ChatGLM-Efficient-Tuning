# CUDA_VISIBLE_DEVICES=4,6 \
# accelerate launch src/train_sft.py \
#     --model_name_or_path /DATA/yuanrz/model/chatglm2-6b \
#     --use_v2 True \
#     --ddp_find_unused_parameters False \
#     --do_train \
#     --dataset covid_train,covid_dev \
#     --dataset_dir data/covid \
#     --finetuning_type lora \
#     --output_dir work_dirs/covid_lora_sft \
#     --overwrite_cache \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 2 \
#     --gradient_accumulation_steps 16 \
#     --max_source_length 1024 \
#     --max_target_length 128 \
#     --lr_scheduler_type cosine \
#     --logging_steps 10 \
#     --save_steps 100 \
#     --eval_steps 100 \
#     --learning_rate 5e-5 \
#     --num_train_epochs 10.0 \
#     --dev_ratio 0.05 \
#     --evaluation_strategy steps \
#     --load_best_model_at_end \
#     --plot_loss \
#     --fp16

CUDA_VISIBLE_DEVICES=7 python src/train_sft.py \
    --model_name_or_path /data/yuanrz/model/chatglm2-6b \
    --use_v2 True \
    --do_train \
    --dataset covid_train,covid_dev \
    --dataset_dir data/covid \
    --finetuning_type lora \
    --output_dir work_dirs/covid_lora_sft \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --max_source_length 1024 \
    --max_target_length 128 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 100 \
    --eval_steps 100 \
    --learning_rate 5e-5 \
    --num_train_epochs 10.0 \
    --dev_ratio 0.05 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --plot_loss \
    --fp16