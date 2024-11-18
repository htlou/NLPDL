export WANDB_API_KEY="7e2dcc0c310ebcb7cdcafd5e9320d6be55cf1a33"
export WANDB_MODE="offline"

DATASETS=("restaurant_sup" "acl_sup" "agnews_sup")

for round in {1..5}; do
    for dataset in ${DATASETS[@]}; do
        python adapter_train.py \
            --model_name_or_path roberta-base \
            --dataset_name $dataset \
            --output_dir ./adapter_results/roberta-base/round${round}/${dataset} \
            --num_train_epochs 5 \
            --per_device_train_batch_size 16 \
            --per_device_eval_batch_size 16 \
            --learning_rate 5e-4 \
            --warmup_ratio 0.05 \
            --weight_decay 0.01 \
            --logging_steps 10 \
            --evaluation_strategy epoch \
            --save_strategy epoch \
            --load_best_model_at_end true \
            --metric_for_best_model accuracy
    done
done