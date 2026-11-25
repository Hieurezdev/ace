### Order-robust Curator experiment

Use `--mode offline --use_lifecycle_curator --order_curation reconcile --order_curation_batch_size 4`.
For ablations, keep the same train order and budget and compare `off` (sequential),
`naive_batch` (frozen groups, native operation application), and `reconcile`
(frozen groups, conflict-aware edit planning). The two batch modes replay the same
group against the candidate and the original Playbook after any budget repair.
The candidate must repair at least one case and regress none on that group.
Unsupported overlaps are held rather than silently overwriting a bullet.
Per-task `post_train_result` in batch modes is pre-commit; see
`playbook_hygiene.jsonl` for the actual batch decisions and replay counts.
Use the existing `--use_budgeted_curator` flag if a hard cap is required.
