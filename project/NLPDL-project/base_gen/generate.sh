MODEL=/data/align-anything/hantao/models/alpaca-7b-reproduced
INPUT_PATH=/data/align-anything/hantao/NLPDL/project/NLPDL-project/data/safety
FEWSHOT_PATH=/data/align-anything/hantao/NLPDL/project/NLPDL-project/data/safety
FEWSHOT_NUM=(20 50 100)
OUTPUT_ROOT=/data/align-anything/hantao/NLPDL/project/NLPDL-project/base_gen/outputs

for num in ${FEWSHOT_NUM[@]}; do
    python base_vllm.py \
        --model_name_or_path $MODEL \
        --input_path $INPUT_PATH \
        --few_shot_path $FEWSHOT_PATH \
        --few_shot_num $num \
        --output_dir $OUTPUT_ROOT \
        --output_name safety_${num}_shot.json
done
