#!/usr/bin/env bash
set -euo pipefail

# Run sequentially so all budgets share the same serving-resource conditions.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$SCRIPT_DIR"

export VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8000/v1}"
ACE_MODEL="${ACE_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
RESULTS_ROOT="${RESULTS_ROOT:-results/playbook_budget_sweep}"
read -r -a BUDGETS <<< "${TOKEN_BUDGETS:-4000 8000 10000}"
read -r -a DATASETS <<< "${TASKS:-formula finer_0.5}"

DRY_RUN=0
EXTRA_ARGS=()
for argument in "$@"; do
  case "$argument" in
    --dry-run) DRY_RUN=1 ;;
    --mode|--mode=*|--task_name|--task_name=*|--save_path|--save_path=*|--playbook_token_budget|--playbook_token_budget=*)
      echo "Use TASKS, TOKEN_BUDGETS, and RESULTS_ROOT to configure this offline sweep instead of ${argument}." >&2
      exit 2
      ;;
    *) EXTRA_ARGS+=("$argument") ;;
  esac
done

for budget in "${BUDGETS[@]}"; do
  if [[ ! "$budget" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid token budget '${budget}': TOKEN_BUDGETS must contain positive integers." >&2
    exit 2
  fi
done
if [[ ${#BUDGETS[@]} -eq 0 || ${#DATASETS[@]} -eq 0 ]]; then
  echo "TOKEN_BUDGETS and TASKS must each contain at least one value." >&2
  exit 2
fi

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
  --use_budgeted_curator
  --budget_repair_attempts 2
  --max_tokens 2048
  --test_workers 5
  --seed 42
  --eval_steps 50
  --save_steps 25
)

for task_name in "${DATASETS[@]}"; do
  for budget in "${BUDGETS[@]}"; do
    echo ">>> ${task_name}: hard Playbook budget ${budget} tokens"
    COMMAND=(
      uv run python -m eval.finance.run
      "${COMMON_ARGS[@]}"
    )
    # macOS ships Bash 3.2, where expanding an empty array under nounset fails.
    if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
      COMMAND+=("${EXTRA_ARGS[@]}")
    fi
    COMMAND+=(
      --task_name "$task_name"
      --playbook_token_budget "$budget"
      --save_path "${RESULTS_ROOT}/${task_name}/budget_${budget}"
    )
    if [[ "$DRY_RUN" -eq 1 ]]; then
      printf '%q ' "${COMMAND[@]}"
      printf '\n'
    else
      "${COMMAND[@]}"
    fi
  done
done
