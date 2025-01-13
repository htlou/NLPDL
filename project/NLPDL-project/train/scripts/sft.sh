#! /bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" &>/dev/null && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export LOGLEVEL="${LOGLEVEL:-WARNING}"

MODEL_NAME_OR_PATH="/data/align-anything/hantao/models/alpaca-7b-reproduced"
OUTPUT_DIR="${ROOT_DIR}/output/dpo"
unset HOSTFILE
ZERO_STAGE=3
OFFLOAD="none"
export WANDB_API_KEY="7e2dcc0c310ebcb7cdcafd5e9320d6be55cf1a33"

mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" &>/dev/null && pwd)"
if [[ ! -f "${OUTPUT_DIR}/.gitignore" ]]; then
	echo '*' >"${OUTPUT_DIR}/.gitignore"
fi

cp -f "$0" "${OUTPUT_DIR}/script.sh"

if [[ -z "${WANDB_API_KEY}" ]]; then
	export WANDB_MODE="offline"
fi

MASTER_PORT_START=10000
MASTER_PORT_END=65535
MASTER_PORT="$(
	comm -23 \
		<(seq "${MASTER_PORT_START}" "${MASTER_PORT_END}" | sort) \
		<(ss -Htan | awk '{ print $4 }' | awk -F ':' '{ print $NF }' | sort -u) |
		shuf | head -n 1
)"

DEEPSPEED_ARGS=()
if [[ -n "${HOSTFILE+x}" ]]; then
	DEEPSPEED_ARGS+=("--hostfile" "${HOSTFILE}")
fi
DEEPSPEED_ARGS+=("--master_port" "${MASTER_PORT}")

exec 1> >(tee "${OUTPUT_DIR}/stdout.log" >&1) 2> >(tee "${OUTPUT_DIR}/stderr.log" >&2)

TRAIN_DATASETS_ROOT="/data/align-anything/hantao/NLPDL/project/NLPDL-project/data/processed"

# NUM_SHOTS=(1 2 5 10 20 50 100)
NUM_SHOTS=(3 7 15)
for NUM_SHOT in ${NUM_SHOTS[@]}; do
	TRAIN_DATASETS="${TRAIN_DATASETS_ROOT}/supervised_train_${NUM_SHOT}.json"
	OUTPUT_DIR="${ROOT_DIR}/output/sft/supervised_${NUM_SHOT}"
	mkdir -p "${OUTPUT_DIR}"
	OUTPUT_DIR="$(cd "${OUTPUT_DIR}" &>/dev/null && pwd)"
	deepspeed "${DEEPSPEED_ARGS[@]}" \
		--module safe_rlhf.finetune \
        --train_datasets nlpdl-supervised::${TRAIN_DATASETS} \
        --model_name_or_path "${MODEL_NAME_OR_PATH}" \
        --max_length 2048 \
        --trust_remote_code True \
        --epochs 3 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 2 \
        --gradient_checkpointing \
        --learning_rate 1e-6 \
        --lr_scheduler_type cosine \
        --lr_warmup_ratio 0.03 \
        --weight_decay 0.05 \
        --seed 42 \
        --output_dir "${OUTPUT_DIR}" \
        --log_type wandb \
        --log_project NLPDL \
        --zero_stage "${ZERO_STAGE}" \
        --offload "${OFFLOAD}" \
        --bf16 True \
        --tf32 True \
        --save_interval 225 \
        --save_16bit
done
