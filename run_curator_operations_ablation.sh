#!/usr/bin/env bash
set -euo pipefail

# Override this when vLLM runs on another host or port.
export VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:5000/v1}"

MODEL="Qwen/Qwen3-4B-Instruct-2507"
RESULTS_ROOT="${RESULTS_ROOT:-results/curator_operations_ablation}"

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
  local operation_name="$2"
  shift 2

  echo ">>> Running ${task_name} / ${operation_name}"
  uv run python -m eval.finance.run \
    --task_name "$task_name" \
    --save_path "${RESULTS_ROOT}/${task_name}/${operation_name}" \
    "${COMMON_ARGS[@]}" \
    "$@"
}

for task_name in formula finer_0.5; do
  # Every individual-operation run retains legacy ADD; only the named lifecycle
  # operation is added, which isolates its contribution against ACE's baseline.
  run_experiment "$task_name" update --use_curator_update
  run_experiment "$task_name" delete_prune \
    --use_curator_delete \
    --prune_unused_bullets \
    --prune_unused_interval 50
  run_experiment "$task_name" merge \
    --use_curator_merge \
    --use_dbscan_merge_candidates

  # Full lifecycle enables ADD, UPDATE, DELETE, MERGE, CREATE_META, and PRUNE.
  run_experiment "$task_name" lifecycle_all \
    --use_lifecycle_curator \
    --use_dbscan_merge_candidates
done

echo ">>> Curator operation ablation completed: ${RESULTS_ROOT}"
