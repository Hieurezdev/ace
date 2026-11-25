### Active Playbook Learning experiment

Use `--mode offline --use_lifecycle_curator --active_playbook disagreement`.
The optional `--active_playbook_pool_size 6 --active_playbook_checker_cases 3
--active_playbook_max_candidates 3` flags bound the probe pool, checked cases,
and Playbook versions (including HOLD). Start with `--max_train_samples 20`
or a larger `--curator_frequency` to bound model calls.

Compare against `--active_playbook random` with the same seeds, budget and
rollout settings, and against `--active_playbook off` for sequential ACE.
Log fields in `playbook_hygiene.jsonl` separate Generator rollouts, checked
case IDs, checker calls and repairs/regressions for each candidate. Finance
already has labels: selective checking simulates a limited reference budget;
normal adaptation still has access to its ground truth. Final test labels
are not used for candidate selection. Existing hard caps can be enabled with
`--use_budgeted_curator`; out-of-budget candidates are not repaired or probed.
