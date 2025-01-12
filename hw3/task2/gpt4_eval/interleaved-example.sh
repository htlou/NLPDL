if [ -z "${BASH_VERSION}" ]; then
	echo "Please use bash to run this script." >&2
	exit 1
fi

set -x

SCRIPT_DIR="$(cd "$(dirname "$0")" &>/dev/null && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
SCRIPT_NAME=$(basename "$0")
SCRIPT_NAME_WITHOUT_EXTENSION="${SCRIPT_NAME%.sh}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

export LOGLEVEL="${LOGLEVEL:-WARNING}"
INPUT_PATH=""
OUTPUT_PATH=""
OUTPUT_FOLDER=""
OUTPUT_NAME=""
MODEL=""
INPUT_TYPE="interleaved-compare"
PLATFORM="openai"
ROOT_DIR=""
output_dir="./output"

while [[ "$#" -gt 0 ]]; do
	arg="$1"
	shift
	case "${arg}" in
		--input-file)
			INPUT_PATH=$1
			shift
			;;
		--folder-name)
			folder_names=("")
			shift
			;;
		--type)
			INPUT_TYPE=$1
			shift
			;;
		*)
			echo "Unknown parameter passed: '${arg}'" >&2
			exit 1
			;;
	esac
done


MASTER_PORT_START=10000
MASTER_PORT_END=65535
MASTER_PORT="$(
	comm -23 \
		<(seq "${MASTER_PORT_START}" "${MASTER_PORT_END}" | sort) \
		<(ss -Htan | awk '{ print $4 }' | awk -F ':' '{ print $NF }' | sort -u) |
		shuf | head -n 1
)"

# ray start --head --port=${MASTER_PORT}
# If master port was used , use a number you like instead, like 6379
# ray stop

for file in "$ROOT_DIR"/*
do
echo "Processing $file"
python3 main.py --debug \
    --openai-api-key-file ${SCRIPT_DIR}/config/openai_api_keys.txt \
    --input-file ${file} \
	--output-dir ${output_dir} \
    --cache-dir ${SCRIPT_DIR}/.cache/${SCRIPT_NAME_WITHOUT_EXTENSION}${file} \
	--num-workers 50 \
	--type ${INPUT_TYPE} \
    --shuffle 
done