#!/usr/bin/env bash
set -euo pipefail

# Invoke from the repository root. Both arms use identical data and model settings.
export VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:5000/v1}"
ACE_MODEL="${ACE_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
RESULTS_ROOT="${RESULTS_ROOT:-results/budgeted_curator_ablation}"
PLAYBOOK_BUDGET="${PLAYBOOK_BUDGET:-8000}"

COMMON_ARGS=(
  --mode offline
  --api_provider vllm
  --num_epochs 1
  --max_num_rounds 3
  --curator_frequency 1
  --generator_model "$ACE_MODEL"
  --reflector_model "$ACE_MODEL"
  --curator_model "$ACE_MODEL"
  --playbook_tokenizer "$ACE_MODEL"
  --playbook_token_budget "$PLAYBOOK_BUDGET"
  --max_tokens 2048
  --test_workers 5
  --seed 42
  --eval_steps 50
  --save_steps 25
)

for task_name in formula finer_0.5; do
  echo ">>> ${task_name}: legacy soft budget"
  uv run python -m eval.finance.run \
    --task_name "$task_name" \
    --save_path "${RESULTS_ROOT}/${task_name}/legacy_soft_budget" \
    "${COMMON_ARGS[@]}" "$@"

  echo ">>> ${task_name}: transactional hard budget"
  uv run python -m eval.finance.run \
    --task_name "$task_name" \
    --save_path "${RESULTS_ROOT}/${task_name}/budgeted_curator" \
    "${COMMON_ARGS[@]}" \
    --use_budgeted_curator --budget_repair_attempts 2 "$@"
done
