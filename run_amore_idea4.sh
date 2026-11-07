#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results_amore_idea4}"
PROFILE="${AMORE_PROFILE:-baseline}"
if [[ "$PROFILE" != baseline && "$PROFILE" != full_operations ]]; then
  echo "AMORE_PROFILE must be baseline or full_operations" >&2
  exit 2
fi
for task in formula finer_0.5; do
  extra=()
  if [[ "$PROFILE" == full_operations ]]; then
    extra+=(--use_lifecycle_curator --use_dbscan_merge_candidates --use_budgeted_curator)
  fi
  if [[ -n "${MAX_TRAIN_SAMPLES:-}" ]]; then
    extra+=(--max_train_samples "$MAX_TRAIN_SAMPLES")
  fi
  command=(uv run python -m eval.finance.run --task_name "$task" --mode offline
    --save_path "$OUTPUT_ROOT/$PROFILE/$task" --api_provider vllm
    --generator_model "$MODEL" --reflector_model "$MODEL" --curator_model "$MODEL"
    --num_epochs 1 --max_num_rounds 3 --max_tokens 2048 --seed 42
    --test_workers 5 --eval_steps 50 --save_steps 25 --playbook_token_budget 8000
    --use_amore --amore_top_k 4 --amore_consolidation_frequency 50 --amore_max_buffer_tasks 10
    --amore_memory_path "$OUTPUT_ROOT/$PROFILE/$task/amore_memory.json")
  if [[ ${#extra[@]} -gt 0 ]]; then command+=("${extra[@]}"); fi
  if [[ "${DRY_RUN:-0}" == 1 ]]; then printf '%q ' "${command[@]}"; printf '\n'
  else "${command[@]}"; fi
done
