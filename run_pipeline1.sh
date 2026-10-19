
# uv run python -m eval.finance.run \
#     --task_name finer_0.5 \
#     --mode offline \
#     --save_path results \
#     --api_provider vllm \
#     --max_num_rounds 3 \
#     --use_verified_failure_memory \
#     --failure_memory_top_k 10 \
#     --generator_model Qwen/Qwen3-4B-Instruct-2507 \
#     --reflector_model Qwen/Qwen3-4B-Instruct-2507 \
#     --curator_model Qwen/Qwen3-4B-Instruct-2507 \
#     --use_lifecycle_curator \
#     --use_dbscan_merge_candidates \
#     --playbook_token_budget 4000 \
#     --max_tokens 2048 \
#     --test_workers 5 \
#     --seed 42 \
#     --eval_steps 50 \
#     --save_steps 25

uv run python -m eval.finance.run \
    --task_name finer_0.5 \
    --mode offline \
    --save_path results \
    --api_provider vllm \
    --max_num_rounds 3 \
    --generator_model Qwen/Qwen3-4B-Instruct-2507 \
    --reflector_model Qwen/Qwen3-4B-Instruct-2507 \
    --curator_model Qwen/Qwen3-4B-Instruct-2507 \
    --use_verified_failure_memory \
    --failure_memory_top_k 10 \
    --use_bulletpoint_analyzer \
    --bulletpoint_analyzer_threshold 0.9 \
    --playbook_token_budget 4000 \
    --max_tokens 2048 \
    --test_workers 5 \
    --seed 42 \
    --eval_steps 50 \
    --save_steps 25


uv run python -m eval.finance.run \
    --task_name formula \
    --mode offline \
    --save_path results \
    --api_provider vllm \
    --max_num_rounds 3 \
    --use_rae \
    --rae_top_k 10 \
    --generator_model Qwen/Qwen3-4B-Instruct-2507 \
    --reflector_model Qwen/Qwen3-4B-Instruct-2507 \
    --curator_model Qwen/Qwen3-4B-Instruct-2507 \
    --use_lifecycle_curator \
    --use_dbscan_merge_candidates \
    --playbook_token_budget 4000 \
    --max_tokens 2048 \
    --test_workers 5 \
    --seed 42 \
    --eval_steps 50 \
    --save_steps 25

uv run python -m eval.finance.run \
    --task_name formula \
    --mode offline \
    --save_path results \
    --api_provider vllm \
    --max_num_rounds 3 \
    --generator_model Qwen/Qwen3-4B-Instruct-2507 \
    --reflector_model Qwen/Qwen3-4B-Instruct-2507 \
    --curator_model Qwen/Qwen3-4B-Instruct-2507 \
    --use_rae \
    --rae_top_k 10 \
    --use_bulletpoint_analyzer \
    --bulletpoint_analyzer_threshold 0.9 \
    --playbook_token_budget 4000 \
    --max_tokens 2048 \
    --test_workers 5 \
    --seed 42 \
    --eval_steps 50 \
    --save_steps 25
