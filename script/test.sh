# CUDA_VISIBLE_DEVICES=3 python src/train_sft.py \
#     --model_name_or_path /data/yuanrz/model/chatglm2-6b \
#     --use_v2 True \
#     --do_predict \
#     --dataset covid_test \
#     --dataset_dir data/covid \
#     --output_dir work_dirs/covid_pred_origin \
#     --overwrite_cache \
#     --per_device_eval_batch_size 4 \
#     --max_source_length 1024 \
#     --max_target_length 128 \
#     --predict_with_generate

CUDA_VISIBLE_DEVICES=3 python src/train_sft.py \
    --model_name_or_path /data/yuanrz/model/chatglm2-6b \
    --use_v2 True \
    --do_predict \
    --dataset covid_test \
    --dataset_dir data/covid \
    --checkpoint_dir work_dirs/covid_lora_sft/ \
    --output_dir work_dirs/covid_lora_sft/predict/covid_pred_best \
    --overwrite_cache \
    --per_device_eval_batch_size 4 \
    --max_source_length 1024 \
    --max_target_length 128 \
    --predict_with_generate