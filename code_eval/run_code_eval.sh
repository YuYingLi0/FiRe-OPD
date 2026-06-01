#!/bin/bash
# Generic code evaluation script for checkpoints
# Usage:
#   bash run_code_eval.sh <EXPERIMENT_NAME> <CKPT_BASE> <MODEL_NAME> [STEP1 STEP2 ...]
# Runs HumanEval+, MBPP+, and LiveCodeBench evaluations
set -x

rm -rf ~/.cache/vllm/torch_compile_cache

# Set proxy if needed
# export http_proxy=http://your-proxy:port
# export https_proxy=http://your-proxy:port

EXPERIMENT_NAME=$1
CKPT_BASE=$2
MODEL_NAME=$3
shift 3
STEPS_ARGS=("$@")

if [ -z "$EXPERIMENT_NAME" ] || [ -z "$CKPT_BASE" ] || [ -z "$MODEL_NAME" ]; then
    echo "Usage: $0 <EXPERIMENT_NAME> <CKPT_BASE> <MODEL_NAME> [STEP1 STEP2 ...]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CODE_EVAL_BASE="${SCRIPT_DIR}"
VERL_DIR="$(cd "${SCRIPT_DIR}/../verl" && pwd)"

# Setup evalplus and LiveCodeBench dependencies
export PYTHONPATH="${CODE_EVAL_BASE}/coding/evalplus:${PYTHONPATH}"
pip3 install -r "${CODE_EVAL_BASE}/coding/evalplus/requirements.txt" pebble --quiet


# EvalPlus settings
EVALPLUS_GREEDY=1

# LiveCodeBench settings
LCB_N=4
LCB_TEMPERATURE=1.0
LCB_TOP_P=1.0
LCB_MAX_TOKENS=16384

if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    IFS=',' read -ra GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
else
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
    GPU_LIST=($(seq 0 $((NUM_GPUS - 1))))
fi
TOTAL_GPUS=${#GPU_LIST[@]}

echo "=== Code Eval: ${EXPERIMENT_NAME} ==="
echo "=== Detected ${TOTAL_GPUS} GPUs ==="

merge_checkpoint() {
    local step=$1
    local actor_dir="${CKPT_BASE}/global_step_${step}/actor"
    local merged_dir="${CKPT_BASE}/global_step_${step}/actor/merged_hf"

    if [ -f "${merged_dir}/config.json" ] && [ -f "${merged_dir}/model.safetensors.index.json" -o -f "${merged_dir}/model.safetensors" ]; then
        echo "=== Step ${step}: Merged HF model already exists, skipping ==="
        return 0
    fi

    if [ ! -d "${actor_dir}" ]; then
        echo "=== Step ${step}: actor dir not found at ${actor_dir}, skipping ==="
        return 1
    fi

    echo "=== Step ${step}: Merging FSDP checkpoint ==="
    cd "${VERL_DIR}"
    python3 scripts/legacy_model_merger.py merge \
        --backend fsdp \
        --local_dir "${actor_dir}" \
        --target_dir "${merged_dir}"

    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to merge checkpoint at step ${step}"
        return 1
    fi
    echo "=== Step ${step}: Merge complete ==="
}

run_evalplus() {
    local model_path=$1
    local step=$2
    local output_dir="${SCRIPT_DIR}/eval_outputs_${EXPERIMENT_NAME}/step${step}"
    mkdir -p "${output_dir}"

    cd "${CODE_EVAL_BASE}"

    # HumanEval+
    local he_result="${output_dir}/humaneval_results.json"
    if [ -f "${he_result}" ]; then
        echo "=== HumanEval+ step${step}: Already exists, skipping ==="
    else
        echo "=== Evaluating HumanEval+ step${step} ==="
        export HUMANEVAL_OVERRIDE_PATH="${CODE_EVAL_BASE}/data/HumanEvalPlus.jsonl"
        export MBPP_OVERRIDE_PATH="${CODE_EVAL_BASE}/data/MbppPlus.jsonl"

        python3 coding/evalplus/evalplus/codegen.py --model "${model_path}" \
            --dataset humaneval \
            --backend vllm \
            --trust_remote_code \
            --greedy

        # Find and evaluate results
        sleep 2
        local MODEL_BASE=$(basename "${model_path}")
        local OUTPUT_FILE=$(find "evalplus_results/humaneval" -name "*${MODEL_BASE}*temp_0.0.jsonl" ! -name "*.raw.jsonl" -type f 2>/dev/null | head -n 1)
        if [ -n "${OUTPUT_FILE}" ]; then
            python3 -m evalplus.evaluate --dataset humaneval \
                --samples "${OUTPUT_FILE}" \
                --output_file "${he_result}" \
                --min-time-limit 10.0 \
                --gt-time-limit-factor 8.0
        fi
    fi

    # MBPP+
    local mbpp_result="${output_dir}/mbpp_results.json"
    if [ -f "${mbpp_result}" ]; then
        echo "=== MBPP+ step${step}: Already exists, skipping ==="
    else
        echo "=== Evaluating MBPP+ step${step} ==="
        export HUMANEVAL_OVERRIDE_PATH="${CODE_EVAL_BASE}/data/HumanEvalPlus.jsonl"
        export MBPP_OVERRIDE_PATH="${CODE_EVAL_BASE}/data/MbppPlus.jsonl"

        python3 coding/evalplus/evalplus/codegen.py --model "${model_path}" \
            --dataset mbpp \
            --backend vllm \
            --trust_remote_code \
            --greedy

        sleep 2
        local MODEL_BASE=$(basename "${model_path}")
        local OUTPUT_FILE=$(find "evalplus_results/mbpp" -name "*${MODEL_BASE}*temp_0.0.jsonl" ! -name "*.raw.jsonl" -type f 2>/dev/null | head -n 1)
        if [ -n "${OUTPUT_FILE}" ]; then
            python3 -m evalplus.evaluate --dataset mbpp \
                --samples "${OUTPUT_FILE}" \
                --output_file "${mbpp_result}" \
                --min-time-limit 10.0 \
                --gt-time-limit-factor 8.0
        fi
    fi
}

run_livecodebench() {
    local model_path=$1
    local step=$2
    local output_dir="${SCRIPT_DIR}/eval_outputs_${EXPERIMENT_NAME}/step${step}"
    mkdir -p "${output_dir}"

    local lcb_marker="${output_dir}/lcb_done.marker"
    if [ -f "${lcb_marker}" ]; then
        echo "=== LiveCodeBench step${step}: Already exists, skipping ==="
        return 0
    fi

    echo "=== Evaluating LiveCodeBench step${step} ==="
    cd "${CODE_EVAL_BASE}/coding/LiveCodeBench"

    python -m lcb_runner.runner.main \
        --model "Qwen3-4B-NonThinking" \
        --local_model_path "${model_path}" \
        --trust_remote_code \
        --scenario codegeneration \
        --release_version v6 \
        --tensor_parallel_size ${TOTAL_GPUS} \
        --use_cache \
        --n ${LCB_N} \
        --temperature ${LCB_TEMPERATURE} \
        --max_tokens ${LCB_MAX_TOKENS} \
        --custom_output_save_name "${MODEL_NAME}_step${step}" \
        --top_p ${LCB_TOP_P} \
        --timeout 60 \
        --evaluate

    if [ $? -eq 0 ]; then
        touch "${lcb_marker}"
    fi
}

# Determine steps
if [ ${#STEPS_ARGS[@]} -gt 0 ]; then
    STEPS=("${STEPS_ARGS[@]}")
else
    STEPS=()
    for d in "${CKPT_BASE}"/global_step_*; do
        [ -d "$d" ] || continue
        s=$(basename "$d" | sed 's/global_step_//')
        STEPS+=("$s")
    done
    IFS=$'\n' STEPS=($(sort -n <<<"${STEPS[*]}")); unset IFS
fi

echo "=== Steps to evaluate: ${STEPS[*]} ==="

for step in "${STEPS[@]}"; do
    merge_checkpoint "${step}" || continue
    local_model_path="${CKPT_BASE}/global_step_${step}/actor/merged_hf"

    if [ ! -d "${local_model_path}" ]; then
        echo "ERROR: Merged model not found at ${local_model_path}"
        continue
    fi

    echo ""
    echo "============================================================"
    echo "  Code Eval: ${EXPERIMENT_NAME} step ${step}"
    echo "============================================================"

    run_evalplus "${local_model_path}" "${step}"
    run_livecodebench "${local_model_path}" "${step}"
done

echo ""
echo "============================================================"
echo "  ${EXPERIMENT_NAME}: All code evaluations complete!"
echo "============================================================"
