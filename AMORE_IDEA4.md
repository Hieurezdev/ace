# Opt-in AMORE idea4 integration

Adapted from https://github.com/pitorr05/aceMementoExper/tree/71d59fa46cc3d6d36fa2b8203dac3920dee564b9 (Apache-2.0).
This ports the idea4 episodic-memory path, not the upstream Planner/Executor
runner or its unrelated benchmark datasets. Native ACE remains the executor.

## Behavior

- Disabled by default: original FMB, adversarial, operations and budget flags remain independent.
- Training buffers completed task trajectories WITH explicit task/environment feedback.
- Topic-change detection examines previous questions before inserting the incoming task.
- Consolidation triggers on topic boundaries, buffer capacity (10 tasks), every 50
  observations, and at the end of training.
- Compression produces a title and summary for cosine retrieval. The resolver
  compares a Playbook-based hypothesis against the ORIGINAL evidence, not the summary.
- Admission requires nonempty insights and exact evidence quotes; errors are
  explicit and retain the persisted pending buffer. A quote check does not prove
  that the diagnosis is correct. AMORE is not the verified FMB.
- Up to 10 insights per episode are retrieved as untrusted Generator context.
  There is no direct AMORE write into the semantic Playbook. Any later semantic
  learning still uses native Reflector/Curator and its enabled budget checks.
- Evaluation only retrieves completed training episodes. It never consumes test
  labels, calls consolidation models, or mutates the memory checkpoint.
- One canonical JSON checkpoint persists episodes AND pending evidence atomically;
  metadata-only events are saved beside it as *.events.jsonl.
- Episodic context is additional input: a Playbook budget does NOT cap the total
  prompt plus AMORE lessons. No claim of perfect correctness or reduced latency
  is implied.
- Checkpoints contain original task evidence. Keep them private; do not commit
  task data or credentials to Git. AMORE event logs exclude raw evidence.
- Use separate checkpoint paths for datasets, seeds and ablation variants. A
  training run with an existing path resumes it; use a fresh output directory for
  a fresh experiment.

## Runs

From this repository root, with the existing vLLM/environment servers running:

```bash
DRY_RUN=1 bash run_amore_idea4.sh
MAX_TRAIN_SAMPLES=5 bash run_amore_idea4.sh
AMORE_PROFILE=full_operations bash run_amore_idea4.sh
```

The baseline profile adds only AMORE; full_operations also selects native
lifecycle operations, DBSCAN Curator candidates and an 8000-token hard Playbook
budget. The scripts do not implicitly enable FMB or adversarial agents.

Finance CLI: add `--use_amore`, `--amore_memory_path PATH`,
`--amore_top_k 4`, `--amore_consolidation_frequency 50` and
`--amore_max_buffer_tasks 10` to existing offline commands. Without the flag,
these settings do nothing. The supplied script covers Formula and FiNER.
Eval-only requires `--use_amore --amore_memory_path TRAINED_MEMORY.json`
alongside the normal trained Playbook arguments. AMORE online mode is explicitly
unsupported in this initial port. Finance uses the existing Reflector model for
memory processing, and reuses RAE's encoder when enabled.
