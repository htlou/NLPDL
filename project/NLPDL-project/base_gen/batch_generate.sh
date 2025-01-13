MODELS=(
    # "/data/align-anything/hantao/NLPDL/project/NLPDL-project/train/output/dpo/preference_1"
    # "/data/align-anything/hantao/NLPDL/project/NLPDL-project/train/output/dpo/preference_2"
    # "/data/align-anything/hantao/NLPDL/project/NLPDL-project/train/output/dpo/preference_5"
    # "/data/align-anything/hantao/NLPDL/project/NLPDL-project/train/output/dpo/preference_10"
    # "/data/align-anything/hantao/NLPDL/project/NLPDL-project/train/output/dpo/preference_20"
    # "/data/align-anything/hantao/NLPDL/project/NLPDL-project/train/output/dpo/preference_50"
    # "/data/align-anything/hantao/NLPDL/project/NLPDL-project/train/output/dpo/preference_100"
    # "/data/align-anything/hantao/NLPDL/project/NLPDL-project/train/output/sft/supervised_1"
    # "/data/align-anything/hantao/NLPDL/project/NLPDL-project/train/output/sft/supervised_2"
    # "/data/align-anything/hantao/NLPDL/project/NLPDL-project/train/output/sft/supervised_5"
    # "/data/align-anything/hantao/NLPDL/project/NLPDL-project/train/output/sft/supervised_10"
    # "/data/align-anything/hantao/NLPDL/project/NLPDL-project/train/output/sft/supervised_20"
    # "/data/align-anything/hantao/NLPDL/project/NLPDL-project/train/output/sft/supervised_50"
    # "/data/align-anything/hantao/NLPDL/project/NLPDL-project/train/output/sft/supervised_100"
    "/data/align-anything/hantao/NLPDL/project/NLPDL-project/train/output/sft/supervised_3"
    "/data/align-anything/hantao/NLPDL/project/NLPDL-project/train/output/sft/supervised_7"
    "/data/align-anything/hantao/NLPDL/project/NLPDL-project/train/output/sft/supervised_15"
    "/data/align-anything/hantao/NLPDL/project/NLPDL-project/train/output/dpo/preference_3"
    "/data/align-anything/hantao/NLPDL/project/NLPDL-project/train/output/dpo/preference_7"
    "/data/align-anything/hantao/NLPDL/project/NLPDL-project/train/output/dpo/preference_15"
)
INPUT_PATH=/data/align-anything/hantao/NLPDL/project/NLPDL-project/data/safety
FEWSHOT_PATH=/data/align-anything/hantao/NLPDL/project/NLPDL-project/data/safety
OUTPUT_ROOT=/data/align-anything/hantao/NLPDL/project/NLPDL-project/base_gen/outputs

for MODEL in ${MODELS[@]}; do
    MODEL_NAME=$(basename $MODEL)
    python base_vllm.py \
        --model_name_or_path $MODEL \
        --input_path $INPUT_PATH \
        --few_shot_path $FEWSHOT_PATH \
        --few_shot_num 0 \
        --output_dir $OUTPUT_ROOT \
        --output_name ${MODEL_NAME}_0_shot.json
done
