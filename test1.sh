
uv run python -m eval.finance.run \
    --task_name finer_0.5 \
    --mode eval_only \
    --save_path results_playbook \
    --api_provider vllm \
    --initial_playbook_path "final_playbook_finer_0.5_without_zero_evidence.txt" \
    --num_epochs 1 \
    --max_num_rounds 3 \
    --generator_model Qwen/Qwen3-4B-Instruct-2507 \
    --reflector_model Qwen/Qwen3-4B-Instruct-2507 \
    --curator_model Qwen/Qwen3-4B-Instruct-2507 \
    --playbook_token_budget 4000 \
    --max_tokens 2048 \
    --test_workers 5 \
    --seed 42 \
    --eval_steps 50 \
    --save_steps 25


uv run python -m eval.finance.run \
    --task_name formula \
    --mode eval_only \
    --save_path results_playbook \
    --api_provider vllm \
    --initial_playbook_path "final_playbook_formula_without_zero_evidence.txt" \
    --num_epochs 1 \
    --max_num_rounds 3 \
    --generator_model Qwen/Qwen3-4B-Instruct-2507 \
    --reflector_model Qwen/Qwen3-4B-Instruct-2507 \
    --curator_model Qwen/Qwen3-4B-Instruct-2507 \
    --playbook_token_budget 4000 \
    --max_tokens 2048 \
    --test_workers 5 \
    --seed 42 \
    --eval_steps 50 \
    --save_steps 25