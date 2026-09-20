#!/usr/bin/env bash
set -euo pipefail

# Override this when vLLM runs on another host or port.
export VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8000/v1}"

MODEL="Qwen/Qwen3-4B-Instruct-2507"
RESULTS_ROOT="results/curator_full_operations_rae_ablation"

COMMON_ARGS=(
  --mode offline
  --api_provider vllm
  --num_epochs 1
  --max_num_rounds 3
  --curator_frequency 1
  --generator_model "$MODEL"
  --reflector_model "$MODEL"
  --curator_model "$MODEL"
  --use_lifecycle_curator
  --use_dbscan_merge_candidates
  --use_rae
  --rae_top_k 10
  --playbook_token_budget 4000
  --max_tokens 2048
  --test_workers 5
  --seed 42
  --eval_steps 50
  --save_steps 25
)

run_experiment() {
  local task_name="$1"
  local experiment_name="$2"
  shift 2

  echo ">>> Running ${task_name} / ${experiment_name}"
  uv run python -m eval.finance.run \
    --task_name "$task_name" \
    --save_path "${RESULTS_ROOT}/${task_name}/${experiment_name}" \
    "${COMMON_ARGS[@]}" \
    "$@"
}

for task_name in formula finer_0.5; do
  run_experiment "$task_name" rae_full_operations

  run_experiment "$task_name" rae_fmb_verified_full_operations \
    --use_verified_failure_memory \
    --failure_memory_top_k 10
done

echo ">>> RAE full-operations ablation completed: ${RESULTS_ROOT}"
