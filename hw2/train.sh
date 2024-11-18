export WANDB_API_KEY="7e2dcc0c310ebcb7cdcafd5e9320d6be55cf1a33"

MODELS=("allenai/scibert_scivocab_uncased")
DATASETS=("restaurant_sup" "acl_sup" "agnews_sup")

for round in {1..5}; do
    for model in ${MODELS[@]}; do
        for dataset in ${DATASETS[@]}; do
            python train.py \
                --model_name_or_path $model \
                --dataset_name $dataset \
                --output_dir ./outputs/${model}/${dataset}/round${round} \
                --num_train_epochs 5 \
                --per_device_train_batch_size 8 \
                --per_device_eval_batch_size 8 \
                --learning_rate 2e-5 \
                --warmup_ratio 0.05 \
                --weight_decay 0.01 \
                --logging_strategy steps \
                --logging_steps 10 \
                --evaluation_strategy epoch \
                --save_strategy epoch \
                --load_best_model_at_end True \
                --metric_for_best_model accuracy \
                --report_to wandb \
                --save_safetensors False
        done
    done
done