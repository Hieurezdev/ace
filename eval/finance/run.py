#!/usr/bin/env python3
"""
Example usage script for the ACE system.

"""
import os
import json
import openai
import argparse
from pathlib import Path
from datetime import datetime
from .data_processor import DataProcessor

from ace import ACE
from ace.core.stress_test import empty_playbook, write_corrupted_playbook
from utils import initialize_clients, set_global_seed

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='ACE System - Refactored')
    
    # Task configuration
    parser.add_argument("--task_name", type=str, required=True,
                        help="Name of the task (e.g., 'finer', 'formula')")
    parser.add_argument("--initial_playbook_path", type=str, default=None,
                        help="Path to initial playbook (optional)")
    parser.add_argument("--mode", type=str, default="offline",
                        choices=["offline", "online", "eval_only"],
                        help="Run mode: 'offline' for offline training with validation, "
                             "'online' for online training and testing on test split, "
                             "'eval_only' for testing only with provided playbook")
    
    # Model configuration
    parser.add_argument("--api_provider", type=str, default="sambanova",
                        choices=["sambanova", "together", "openai", "vllm", "sglang"], help="API provider")
    parser.add_argument("--generator_model", type=str, 
                        default="DeepSeek-V3.1",
                        help="Model for generator")
    parser.add_argument("--reflector_model", type=str,
                        default="DeepSeek-V3.1",
                        help="Model for reflector")
    parser.add_argument("--curator_model", type=str,
                        default="DeepSeek-V3.1",
                        help="Model for curator")
    
    # Training configuration
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--max_num_rounds", type=int, default=3,
                        help="Max reflection rounds for incorrect answers")
    parser.add_argument("--curator_frequency", type=int, default=1,
                        help="Run curator every N steps")
    parser.add_argument("--eval_steps", type=int, default=100,
                        help="Evaluate every N steps")
    parser.add_argument("--online_eval_frequency", type=int, default=15,
                        help="Update playbook every N samples for evaluation in online mode")
    parser.add_argument("--save_steps", type=int, default=50,
                        help="Save intermediate playbooks every N steps")
    parser.add_argument("--resume_from_step", type=int, default=1,
                        help="Resume offline training from this 1-based sample index")
    parser.add_argument("--skip_test_set", action="store_true",
                        help="Skip initial/final test set evaluation")
    
    # System configuration
    parser.add_argument("--max_tokens", type=int, default=4096,
                        help="Max tokens for LLM responses")
    parser.add_argument("--playbook_token_budget", type=int, default=80000,
                        help="Total token budget for playbook")
    parser.add_argument("--test_workers", type=int, default=20,
                        help="Number of parallel workers for testing")
    parser.add_argument("--track_generation_latency", action="store_true",
                        help="In eval_only mode, stream Generator calls and save average TTFT/TPOT metrics")
    
    # Prompt configuration
    parser.add_argument("--json_mode", action="store_true",
                        help="Enable JSON mode for LLM calls")
    parser.add_argument("--no_ground_truth", action="store_true",
                        help="Don't use ground truth in reflection")
    
    # Bulletpoint analyzer configuration
    parser.add_argument("--use_bulletpoint_analyzer", action="store_true",
                        help="Enable bulletpoint analyzer for deduplication and merging")
    parser.add_argument("--bulletpoint_analyzer_threshold", type=float, default=0.90,
                        help="Similarity threshold for bulletpoint analyzer (0-1, default: 0.90)")

    # RAE configuration
    parser.add_argument("--use_rae", action="store_true",
                        help="Enable Retrieval-Augmented Execution at the Generator "
                             "(retrieves Top-K relevant bullets per query via BGE-M3 + FAISS)")
    parser.add_argument("--rae_top_k", type=int, default=10,
                        help="Number of Top-K bullets to retrieve per query when RAE is enabled (default: 10)")

    # Failure Memory (Analogical Reflection) configuration
    failure_memory_group = parser.add_mutually_exclusive_group()
    failure_memory_group.add_argument(
        "--use_failure_memory",
        action="store_true",
        help=(
            "Enable the original legacy Failure Memory Bank. Shares BGE-M3 "
            "with RAE when both are enabled."
        ),
    )
    parser.add_argument("--failure_memory_top_k", type=int, default=3,
                        help="Number of similar past failures to retrieve per reflection step (default: 3)")
    failure_memory_group.add_argument(
        "--use_verified_failure_memory",
        action="store_true",
        help=(
            "Enable schema-v2 verified failure memory with multi-stage retrieval. "
            "The existing --use_failure_memory flag keeps legacy behavior."
        ),
    )
    
    # Adversarial agent configuration
    parser.add_argument("--use_adversarial", action="store_true",
                        help="Enable adversarial agent for active playbook stress testing")
    parser.add_argument("--adversarial_frequency", type=int, default=10,
                        help="Run adversarial episode every N steps (default: 10)")
    parser.add_argument("--adversarial_model", type=str, default=None,
                        help="Model for adversarial agent (defaults to generator model)")
    parser.add_argument("--adversarial_mode", choices=["legacy", "verified"], default="verified",
                        help="Adversarial implementation: original single-call or verified pipeline")
    parser.add_argument("--adversarial_num_candidates", type=int, default=5,
                        help="Number of adversarial candidates generated per episode (default: 5)")
    parser.add_argument("--adversarial_verifier_min_confidence", type=float, default=0.80,
                        help="Minimum verifier confidence for accepting an attack (default: 0.80)")
    parser.add_argument("--adversarial_verifier_max_ambiguity", type=float, default=0.20,
                        help="Maximum ambiguity allowed for an attack (default: 0.20)")
    
    # Output configuration
    parser.add_argument("--save_path", type=str, required=True,
                        help="Directory to save results")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument(
        "--stress_noise_rate", type=float, default=0.0,
        help=(
            "Inject this fraction of controlled harmful Playbook bullets into an "
            "isolated stress-test clone. Set 0 to disable (default)."
        ),
    )
    parser.add_argument(
        "--stress_noise_mode", choices=["replace", "append"], default="replace",
        help="replace keeps Playbook length fixed; append adds harmful distractors.",
    )
    parser.add_argument(
        "--stress_noise_seed", type=int, default=None,
        help="Seed for selecting corrupted bullets; defaults to --seed.",
    )
    parser.add_argument(
        "--stress_noise_schedule", choices=["initial", "interval", "both"], default="initial",
        help="Inject only before training, at fixed training intervals, or both.",
    )
    parser.add_argument(
        "--stress_inject_interval", type=int, default=0,
        help="Inject stress noise every N global offline-training steps when schedule is interval/both.",
    )
    parser.add_argument(
        "--max_train_samples", type=int, default=None,
        help="Limit offline training to the first N processed samples; useful for controlled pilots.",
    )
    
    return parser.parse_args()

def load_data(data_path: str):
    """
    Load and process data from a JSONL file.
    
    Args:
        data_path: Path to the JSONL file
        
    Returns:
        List of dictionaries containing the data
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                data.append(json.loads(line))
    
    print(f"Loaded {len(data)} samples from {data_path}")
    return data

def preprocess_data(task_name, config, mode):
    """
    Load training and test data for the specified task.
    
    Args:
        task_name: Name of the task
        config: Configuration dictionary with data paths
        mode: Run mode ('offline', 'online', or 'eval_only')
    
    Returns:
        Tuple of (train_samples, val_samples, test_samples, data_processor)
        - For offline mode: all three are loaded
        - For online mode: only test_samples
        - For eval_only mode: only test_samples
    """
    processor = DataProcessor(task_name=task_name)
    
    # For online and eval_only modes, only load test data
    if mode in ["online", "eval_only"]:
        train_samples = None
        val_samples = None
        
        if "test_data" in config:
            test_samples = load_data(config["test_data"])
            test_samples = processor.process_task_data(test_samples)
        else:
            raise ValueError(f"{mode} mode requires test data in config.")
        
        if mode == "online":
            print(f"Online mode: Training and testing on {len(test_samples)} examples")
        else:
            print(f"Eval only mode: Testing on {len(test_samples)} examples")
    
    # For offline mode, load train, val, and optionally test data
    else:
        train_samples = load_data(config["train_data"])
        val_samples = load_data(config["val_data"])
        train_samples = processor.process_task_data(train_samples)
        val_samples = processor.process_task_data(val_samples)
        
        if "test_data" in config:
            test_samples = load_data(config["test_data"])
            test_samples = processor.process_task_data(test_samples)
        else:
            test_samples = []
        
        print(f"Offline mode: Training on {len(train_samples)} examples, "
              f"validating on {len(val_samples)}, testing on {len(test_samples)}")
    
    return train_samples, val_samples, test_samples, processor


def load_initial_playbook(path):
    """Load initial playbook if provided."""
    if path and os.path.exists(path):
        with open(path, 'r') as f:
            return f.read()
    return None


def main():
    """Main execution function."""
    args = parse_args()

    if args.track_generation_latency and args.mode != "eval_only":
        raise ValueError("--track_generation_latency is supported only with --mode eval_only")

    set_global_seed(args.seed)
    print(f"Using seed: {args.seed}")
    
    print(f"\n{'='*60}")
    print(f"ACE SYSTEM")
    print(f"{'='*60}")
    print(f"Task: {args.task_name}")
    print(f"Mode: {args.mode.upper().replace('_', ' ')}")
    print(f"Generator Model: {args.generator_model}")
    print(f"{'='*60}\n")
    
    # Load data
    with open("./eval/finance/data/sample_config.json", 'r') as f:
        task_config = json.load(f)

    train_samples, val_samples, test_samples, data_processor = preprocess_data(
        args.task_name, 
        task_config[args.task_name],
        args.mode
    )
    if args.max_train_samples is not None:
        if args.max_train_samples <= 0:
            raise ValueError("--max_train_samples must be positive")
        if args.mode != "offline":
            raise ValueError("--max_train_samples is supported only in offline mode")
        train_samples = train_samples[:args.max_train_samples]
        print(f"Limiting offline training to {len(train_samples)} samples")
    if args.stress_noise_schedule in {"interval", "both"} and args.stress_inject_interval <= 0:
        raise ValueError(
            "--stress_inject_interval must be positive when "
            "--stress_noise_schedule is interval or both"
        )
        
    # Load initial playbook (or use empty if None provided)
    initial_playbook = load_initial_playbook(args.initial_playbook_path)
    if args.stress_noise_rate and args.stress_noise_schedule in {"initial", "both"}:
        if not initial_playbook:
            if args.stress_noise_mode == "replace":
                raise ValueError(
                    "--stress_noise_mode replace requires --initial_playbook_path; "
                    "use append to inject into a newly initialized Playbook"
                )
            initial_playbook = empty_playbook()
        if not initial_playbook:
            raise ValueError("--initial_playbook_path did not contain a Playbook")
        stress_seed = args.seed if args.stress_noise_seed is None else args.stress_noise_seed
        initial_playbook, stress_manifest = write_corrupted_playbook(
            initial_playbook,
            os.path.join(args.save_path, "stress_test_inputs"),
            noise_rate=args.stress_noise_rate,
            mode=args.stress_noise_mode,
            seed=stress_seed,
        )
        print(
            "✓ Stress test enabled: "
            f"mode={stress_manifest['mode']}, "
            f"rate={stress_manifest['noise_rate_realized']:.3f}, "
            f"clone={stress_manifest['playbook_path']}"
        )
    if initial_playbook:
        print(f"Loaded initial playbook from {args.initial_playbook_path}\n")
    else:
        print("Using empty playbook as initial playbook\n")
    
    # Create ACE system
    ace_system = ACE(
        api_provider=args.api_provider,
        generator_model=args.generator_model,
        reflector_model=args.reflector_model,
        curator_model=args.curator_model,
        max_tokens=args.max_tokens,
        initial_playbook=initial_playbook,
        use_bulletpoint_analyzer=args.use_bulletpoint_analyzer,
        bulletpoint_analyzer_threshold=args.bulletpoint_analyzer_threshold,
        use_rae=args.use_rae,
        rae_top_k=args.rae_top_k,
        use_failure_memory=(args.use_failure_memory or args.use_verified_failure_memory),
        failure_memory_top_k=args.failure_memory_top_k,
        failure_memory_mode=("verified" if args.use_verified_failure_memory else "legacy"),
        adversarial_model=args.adversarial_model,
        use_adversarial=args.use_adversarial,
        adversarial_frequency=args.adversarial_frequency,
        adversarial_mode=args.adversarial_mode,
        adversarial_num_candidates=args.adversarial_num_candidates,
        adversarial_verifier_min_confidence=args.adversarial_verifier_min_confidence,
        adversarial_verifier_max_ambiguity=args.adversarial_verifier_max_ambiguity,
    )
    
    # Prepare configuration
    config = {
        'num_epochs': args.num_epochs,
        'max_num_rounds': args.max_num_rounds,
        'curator_frequency': args.curator_frequency,
        'eval_steps': args.eval_steps,
        'online_eval_frequency': args.online_eval_frequency,
        'save_steps': args.save_steps,
        'resume_from_step': args.resume_from_step,
        'playbook_token_budget': args.playbook_token_budget,
        'task_name': args.task_name,
        'mode': args.mode,
        'json_mode': args.json_mode,
        'no_ground_truth': args.no_ground_truth,
        'save_dir': args.save_path,
        'test_workers': args.test_workers,
        'track_generation_latency': args.track_generation_latency,
        'initial_playbook_path': args.initial_playbook_path,
        'stress_noise_rate': args.stress_noise_rate,
        'stress_noise_mode': args.stress_noise_mode,
        'stress_noise_seed': args.stress_noise_seed,
        'stress_noise_schedule': args.stress_noise_schedule,
        'stress_inject_interval': args.stress_inject_interval,
        'max_train_samples': args.max_train_samples,
        'use_bulletpoint_analyzer': args.use_bulletpoint_analyzer,
        'bulletpoint_analyzer_threshold': args.bulletpoint_analyzer_threshold,
        'use_rae': args.use_rae,
        'rae_top_k': args.rae_top_k,
        'use_failure_memory': (args.use_failure_memory or args.use_verified_failure_memory),
        'failure_memory_top_k': args.failure_memory_top_k,
        'failure_memory_mode': ("verified" if args.use_verified_failure_memory else "legacy"),
        'api_provider': args.api_provider,
        'use_adversarial': args.use_adversarial,
        'adversarial_frequency': args.adversarial_frequency,
        'adversarial_mode': args.adversarial_mode,
        'adversarial_num_candidates': args.adversarial_num_candidates,
        'adversarial_verifier_min_confidence': args.adversarial_verifier_min_confidence,
        'adversarial_verifier_max_ambiguity': args.adversarial_verifier_max_ambiguity,
        'seed': args.seed,
    }

    if args.resume_from_step > 1 and args.initial_playbook_path:
        config['resume_run_path'] = str(Path(args.initial_playbook_path).resolve().parent.parent)
    
    # Execute using the unified run method
    run_test_samples = test_samples
    if args.mode == "offline" and (args.skip_test_set or args.resume_from_step > 1):
        print("Skipping test set evaluation for offline run")
        run_test_samples = None

    results = ace_system.run(
        mode=args.mode,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=run_test_samples,
        data_processor=data_processor,
        config=config
    )
        

if __name__ == "__main__":
    main()
