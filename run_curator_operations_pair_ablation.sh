#!/usr/bin/env bash
set -euo pipefail

# Override this when vLLM runs on another host or port.
export VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8000/v1}"

MODEL="Qwen/Qwen3-4B-Instruct-2507"
RESULTS_ROOT="${RESULTS_ROOT:-results/curator_operations_pair_ablation}"

COMMON_ARGS=(
  --mode offline
  --api_provider vllm
  --num_epochs 1
  --max_num_rounds 3
  --curator_frequency 1
  --generator_model "$MODEL"
  --reflector_model "$MODEL"
  --curator_model "$MODEL"
  --playbook_token_budget 4000
  --max_tokens 2048
  --test_workers 5
  --seed 42
  --eval_steps 50
  --save_steps 25
)

run_experiment() {
  local task_name="$1"
  local operation_pair="$2"
  shift 2

  echo ">>> Running ${task_name} / ${operation_pair}"
  uv run python -m eval.finance.run \
    --task_name "$task_name" \
    --save_path "${RESULTS_ROOT}/${task_name}/${operation_pair}" \
    "${COMMON_ARGS[@]}" \
    "$@"
}

for task_name in finer_0.5; do
  # ADD is always enabled by default; each run adds exactly two lifecycle
  # operations to measure their combined contribution against that baseline.
  run_experiment "$task_name" update_delete \
    --use_curator_update \
    --use_curator_delete \
    --prune_unused_bullets \
    --prune_unused_interval 50

  run_experiment "$task_name" delete_merge \
    --use_curator_delete \
    --use_curator_merge \
    --prune_unused_bullets \
    --prune_unused_interval 50 \
    --use_dbscan_merge_candidates

  run_experiment "$task_name" merge_update \
    --use_curator_merge \
    --use_curator_update \
    --use_dbscan_merge_candidates
done

echo ">>> Curator operation-pair ablation completed: ${RESULTS_ROOT}"
