#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export RESULTS_ROOT="${RESULTS_ROOT:-results/playbook_budget_sweep_full_operations}"

# Reuse the same budgets, datasets, and training settings as the budget-only arm.
# Full lifecycle enables ADD/UPDATE/MERGE/DELETE/CREATE_META and unused pruning.
# DBSCAN proposes groups to Curator; it does not enable post-Curator hygiene.
exec bash "${SCRIPT_DIR}/run_playbook_budget_sweep.sh" \
  --use_lifecycle_curator \
  --use_dbscan_merge_candidates \
  --curator_dbscan_similarity_threshold 0.9 \
  --prune_unused_interval 50 \
  "$@"
