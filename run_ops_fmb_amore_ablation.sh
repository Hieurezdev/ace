#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
read -r -a variants <<< "${VARIANTS:-ops_fmb ops_amore fmb_amore all}"
MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/ops_fmb_amore_ablation}"
if [[ ${#variants[@]} -eq 0 ]]; then echo "VARIANTS must not be empty" >&2; exit 2; fi
if [[ -n "${MAX_TRAIN_SAMPLES:-}" && ! "$MAX_TRAIN_SAMPLES" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_TRAIN_SAMPLES must be positive" >&2; exit 2
fi
for variant in "${variants[@]}"; do
  case "$variant" in ops_fmb|ops_amore|fmb_amore|all) ;; *) echo "Unknown variant: $variant" >&2; exit 2 ;; esac
  for task in formula finer_0.5; do
    if [[ "${DRY_RUN:-0}" != 1 && -e "$OUTPUT_ROOT/$variant/$task" ]]; then
      echo "Output already exists; select a fresh OUTPUT_ROOT: $OUTPUT_ROOT/$variant/$task" >&2; exit 2
    fi
  done
done
for variant in "${variants[@]}"; do
  for task in formula finer_0.5; do
    flags=()
    case "$variant" in ops_fmb|ops_amore|all)
      flags+=(--use_lifecycle_curator --use_dbscan_merge_candidates) ;; esac
    case "$variant" in ops_fmb|fmb_amore|all)
      flags+=(--use_verified_failure_memory --failure_memory_top_k 10) ;; esac
    case "$variant" in ops_amore|fmb_amore|all)
      flags+=(--use_amore --amore_top_k 4 --amore_consolidation_frequency 50
        --amore_max_buffer_tasks 10 --amore_memory_path "$OUTPUT_ROOT/$variant/$task/amore_memory.json") ;; esac
    if [[ -n "${MAX_TRAIN_SAMPLES:-}" ]]; then flags+=(--max_train_samples "$MAX_TRAIN_SAMPLES"); fi
    command=(uv run python -m eval.finance.run --task_name "$task" --mode offline
      --save_path "$OUTPUT_ROOT/$variant/$task" --api_provider vllm
      --generator_model "$MODEL" --reflector_model "$MODEL" --curator_model "$MODEL"
      --num_epochs 1 --max_num_rounds 3 --curator_frequency 1 --playbook_token_budget 8000
      --max_tokens 2048 --test_workers 5 --seed 42 --eval_steps 50 --save_steps 25
      "${flags[@]}")
    # Hard-budget repair is deliberately OFF for every variant: it can itself
    # invoke UPDATE/MERGE, which would confound the lifecycle-operation ablation.
    if [[ "${DRY_RUN:-0}" == 1 ]]; then printf '%q ' "${command[@]}"; printf '\n'
    else "${command[@]}"; fi
  done
done
