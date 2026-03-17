#!/bin/bash
set -e

echo ">>> 1. Clone Main Repo..."
git clone https://github.com/Hieurezdev/ace.git
cd ace/

echo ">>> 2. Install dependencies with uv..."
uv sync
uv add vllm wrapt transformers

if [ -z "${MODEL_DIR:-}" ]; then
    if [ -d "/kaggle/temp" ]; then
        MODEL_DIR="/kaggle/temp/model"
    else
        MODEL_DIR="./model"
    fi
fi

if [ -z "${CACHE_DIR:-}" ]; then
    if [ -d "/kaggle/temp" ]; then
        CACHE_DIR="/kaggle/temp/hf_cache"
    else
        CACHE_DIR="./.hf_cache"
    fi
fi

echo ">>> 2.5. Download model weights..."
uv run python install_model.py --save-dir "$MODEL_DIR" --cache-dir "$CACHE_DIR" --min-free-gb 10

echo ">>> 2.6. Update config.json for long context (YARN)..."
uv run python change_config.py --model-path "$MODEL_DIR" --target-context 163840

echo ">>> 3. Start vLLM server..."
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

uv run vllm serve "$MODEL_DIR" \
    --served-model-name Qwen/Qwen2-7B-Instruct \
    --dtype auto \
    --gpu-memory-utilization 0.85 \
    --max-model-len 163840 \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code &

echo ">>> Waiting for vLLM server to be ready..."
until curl -s http://localhost:8000/health > /dev/null 2>&1; do
    echo "... vLLM not ready yet, retrying in 10s"
    sleep 10
done
echo ">>> vLLM server is up!"

echo ">>> 4. Run evaluation..."
uv run python -m eval.finance.run \
    --task_name formula \
    --mode offline \
    --save_path results \
    --api_provider vllm \
    --num_epochs 1 \
    --max_num_rounds 3 \
    --generator_model Qwen/Qwen2-7B-Instruct \
    --reflector_model Qwen/Qwen2-7B-Instruct \
    --curator_model Qwen/Qwen2-7B-Instruct \
    --playbook_token_budget 4000 \
    --max_tokens 2048 \
    --test_workers 5 \
    --eval_steps 50 \
    --save_steps 25

echo ">>> Done!"
