"""
ACE (Agent-Curator-Environment) System
Main orchestrator class for training and testing with playbook-based learning.

This module coordinates three agents:
- Generator: Produces answers using playbook knowledge
- Reflector: Analyzes outputs and tags bullets
- Curator: Updates the playbook based on feedback
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

from .core import Generator, Reflector, Curator, BulletpointAnalyzer, PlaybookRetriever, FailureMemoryBank
from .core import AdversarialAgent
from .core.stress_test import corrupt_playbook
from playbook_utils import *
from logger import *
from utils import *


class ACE:
    """
    Main ACE system orchestrator.
    
    Manages the training loop where:
    1. Generator produces answers using playbook
    2. Reflector analyzes answers and tags bullets
    3. Curator updates playbook based on feedback
    
    """
    
    def __init__(
        self,
        api_provider: str,
        generator_model: str,
        reflector_model: str,
        curator_model: str,
        max_tokens: int = 4096,
        initial_playbook: Optional[str] = None,
        use_bulletpoint_analyzer: bool = False,
        bulletpoint_analyzer_threshold: float = 0.90,
        use_lifecycle_curator: bool = False,
        use_curator_update: bool = False,
        use_curator_delete: bool = False,
        use_curator_merge: bool = False,
        use_curator_create_meta: bool = False,
        use_dbscan_merge: bool = False,
        use_dbscan_merge_candidates: bool = False,
        dbscan_eps: float = 0.12,
        dbscan_min_samples: int = 2,
        curator_dbscan_similarity_threshold: float = 0.90,
        delete_harmful_margin: int = 4,
        delete_min_harmful: int = 3,
        prune_unused_bullets: bool = False,
        prune_unused_interval: int = 50,
        use_rae: bool = False,
        rae_top_k: int = 10,
        rae_retrieval_mode: str = "semantic",
        rae_random_seed: int = 42,
        use_failure_memory: bool = False,
        failure_memory_top_k: int = 3,
        failure_memory_mode: str = "legacy",
        use_adversarial: bool = False,
        adversarial_frequency: int = 10,
        adversarial_model: Optional[str] = None,
        adversarial_num_candidates: int = 5,
        adversarial_verifier_min_confidence: float = 0.80,
        adversarial_verifier_max_ambiguity: float = 0.20,
        adversarial_mode: str = "verified",
    ):
        """
        Initialize the ACE system.
        
        Args:
            api_provider: API provider for LLM calls
            generator_model: Model name for generator
            reflector_model: Model name for reflector
            curator_model: Model name for curator
            max_tokens: Maximum tokens for LLM calls
            initial_playbook: Initial playbook content (optional)
            use_bulletpoint_analyzer: Whether to use bulletpoint analyzer for deduplication
            bulletpoint_analyzer_threshold: Similarity threshold for bulletpoint analyzer (0-1)
            use_lifecycle_curator: Enable Curator UPDATE, DELETE, MERGE, and CREATE_META operations.
            use_curator_update: Enable Curator UPDATE operations only.
            use_curator_delete: Enable Curator DELETE operations only.
            use_curator_merge: Enable Curator MERGE operations only.
            use_curator_create_meta: Enable Curator CREATE_META operations only.
            use_dbscan_merge: Run BulletpointAnalyzer DBSCAN hygiene merge after curation.
            use_dbscan_merge_candidates: Use DBSCAN only to propose candidate groups to Curator MERGE.
            prune_unused_bullets: Periodically remove bullets with zero helpful and harmful evidence.
            prune_unused_interval: Run unused-bullet pruning every N training samples.
                                   Lifecycle Curator enables this by default.
            use_rae: Enable Retrieval-Augmented Execution at the Generator (Top-K bullet retrieval)
            rae_top_k: Number of Top-K bullets to retrieve per query when RAE is enabled
            rae_retrieval_mode: ``semantic`` retrieval or the deterministic
                                ``random`` Top-K ablation control.
            rae_random_seed: Seed for the random Top-K ablation control.
            use_failure_memory: Enable Analogical Reflection — retrieve similar past failures
                                at reflection time to enrich the Reflector's analysis.
                                Shares the BGE-M3 embedding model with RAE when both are enabled.
            failure_memory_top_k: Number of similar past failures to retrieve per reflection step.
            failure_memory_mode: 'legacy' for the original bank or 'verified' for
                                 evidence gating and multi-stage retrieval.
            use_adversarial: Enable adversarial agent for active playbook stress testing.
            adversarial_frequency: Run adversarial episode every N steps (only in train modes).
            adversarial_model: Model name for adversarial agent (defaults to generator model).
            adversarial_num_candidates: Number of attacks generated before verification/selection.
            adversarial_verifier_min_confidence: Minimum verifier confidence for accepting an attack.
            adversarial_verifier_max_ambiguity: Maximum ambiguity allowed for an accepted attack.
            adversarial_mode: 'legacy' for one-call generation or 'verified' for the full pipeline.
        """
        # Initialize API clients
        generator_client, reflector_client, curator_client = initialize_clients(api_provider)

        # Initialize the three agents
        self.generator = Generator(generator_client, api_provider, generator_model, max_tokens)
        self.reflector = Reflector(reflector_client, api_provider, reflector_model, max_tokens)
        self.curator = Curator(curator_client, api_provider, curator_model, max_tokens)
        
        # Initialize bulletpoint analyzer if requested and available
        # DBSCAN is an analyzer clustering mode; enabling it must also enable
        # the analyzer or no clustering/merge pass would ever be executed.
        self.use_bulletpoint_hygiene = use_bulletpoint_analyzer or use_dbscan_merge
        self.use_bulletpoint_analyzer = self.use_bulletpoint_hygiene
        self.bulletpoint_analyzer_threshold = bulletpoint_analyzer_threshold
        self.curator_allowed_operations = ["ADD"]
        if use_lifecycle_curator or use_curator_update:
            self.curator_allowed_operations.append("UPDATE")
        if use_lifecycle_curator or use_curator_delete:
            self.curator_allowed_operations.append("DELETE")
        if use_lifecycle_curator or use_curator_merge:
            self.curator_allowed_operations.append("MERGE")
        if use_lifecycle_curator or use_curator_create_meta:
            self.curator_allowed_operations.append("CREATE_META")
        self.use_dbscan_merge = use_dbscan_merge
        self.use_dbscan_merge_candidates = use_dbscan_merge_candidates
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        if not 0.0 <= curator_dbscan_similarity_threshold <= 1.0:
            raise ValueError("curator_dbscan_similarity_threshold must be between 0 and 1")
        self.curator_dbscan_similarity_threshold = curator_dbscan_similarity_threshold
        self.delete_harmful_margin = delete_harmful_margin
        self.delete_min_harmful = delete_min_harmful
        self.prune_unused_bullets = prune_unused_bullets or use_lifecycle_curator
        if self.prune_unused_bullets and prune_unused_interval <= 0:
            raise ValueError("prune_unused_interval must be positive when prune_unused_bullets is enabled")
        self.prune_unused_interval = prune_unused_interval
        
        if self.use_bulletpoint_hygiene or "MERGE" in self.curator_allowed_operations:
            self.bulletpoint_analyzer = BulletpointAnalyzer(
                curator_client, 
                curator_model, 
                max_tokens
            )
            clustering = "DBSCAN hygiene" if use_dbscan_merge else "pairwise candidates"
            print(f"✓ BulletpointAnalyzer initialized ({clustering}, threshold={bulletpoint_analyzer_threshold})")
        else:
            self.bulletpoint_analyzer = None

        # Initialize PlaybookRetriever (RAE) if requested
        self.use_rae = use_rae
        self.rae_top_k = rae_top_k
        if use_rae:
            self.playbook_retriever = PlaybookRetriever(
                embedding_model_name='BAAI/bge-m3',
                embedding_dim=1024,
                top_k=rae_top_k,
                retrieval_mode=rae_retrieval_mode,
                random_seed=rae_random_seed,
            )
            print(f"✓ PlaybookRetriever (mode={rae_retrieval_mode}, top_k={rae_top_k}) initialized")
        else:
            self.playbook_retriever = None
        
        # Store configuration
        self.generator_client = generator_client
        self.reflector_client = reflector_client
        self.curator_client = curator_client
        self.max_tokens = max_tokens
        
        self.use_adversarial = use_adversarial
        self.adversarial_frequency = adversarial_frequency
        adversarial_model_name = adversarial_model or generator_model
        self.adversarial_agent = (
            AdversarialAgent(
                generator_client,
                api_provider,
                adversarial_model_name,
                max_tokens,
                num_candidates=adversarial_num_candidates,
                verifier_min_confidence=adversarial_verifier_min_confidence,
                verifier_max_ambiguity=adversarial_verifier_max_ambiguity,
                mode=adversarial_mode,
            )
            if use_adversarial else None
        )
        
        # Initialize playbook
        if initial_playbook:
            self.playbook = initial_playbook
        else:
            self.playbook = self._initialize_empty_playbook()

        self.best_playbook = self.playbook
        # Track global bullet ID; continue from the highest existing bullet id
        # so resumed runs never reuse ids from the initial playbook.
        self.next_global_id = get_next_global_id(self.playbook)

        # Pre-build RAE index from the initial playbook so retrieval works from step 1
        # (BGE-M3 is lazy-loaded here: downloaded to ~/.cache/huggingface/ if not cached)
        if self.use_rae and self.playbook_retriever:
            self.playbook_retriever.update_index(self.playbook)

        # Initialize FailureMemoryBank (Analogical Reflection)
        # Shares the BGE-M3 encoder with PlaybookRetriever when RAE is enabled
        # so only one copy of the model is loaded.
        self.use_failure_memory = use_failure_memory
        self.failure_memory_top_k = failure_memory_top_k
        if use_failure_memory:
            shared_encoder = self.playbook_retriever.encode if self.use_rae and self.playbook_retriever else None
            self.failure_memory = FailureMemoryBank(
                encoder=shared_encoder,
                top_k=failure_memory_top_k,
                mode=failure_memory_mode,
            )
            src = "shared BGE-M3 from RAE" if shared_encoder is not None else "standalone BGE-M3"
            print(
                f"✓ FailureMemoryBank initialized (mode={failure_memory_mode}, "
                f"top_k={failure_memory_top_k}, encoder={src})"
            )
        else:
            self.failure_memory = None
    
    def _get_curator_merge_candidates(self, log_dir, call_id):
        if "MERGE" not in self.curator_allowed_operations or not self.bulletpoint_analyzer:
            return []
        return self.bulletpoint_analyzer.discover_merge_candidates(
            playbook=self.playbook,
            threshold=self.bulletpoint_analyzer_threshold,
            clustering="dbscan" if self.use_dbscan_merge_candidates else "pairwise",
            dbscan_eps=1.0 - self.curator_dbscan_similarity_threshold,
            dbscan_min_samples=self.dbscan_min_samples,
            log_dir=log_dir,
            call_id=f"{call_id}_merge_candidates",
        )

    def _initialize_empty_playbook(self) -> str:
        """Initialize an empty playbook with standard sections."""
        return """## STRATEGIES & INSIGHTS

## FORMULAS & CALCULATIONS

## CODE SNIPPETS & TEMPLATES

## COMMON MISTAKES TO AVOID

## PROBLEM-SOLVING HEURISTICS

## CONTEXT CLUES & INDICATORS

## OTHERS"""
    
    def _extract_config_params(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract common configuration parameters.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Dictionary with extracted parameters
        """
        return {
            'num_epochs': config.get('num_epochs', 1),
            'max_num_rounds': config.get('max_num_rounds', 3),
            'curator_frequency': config.get('curator_frequency', 1),
            'eval_steps': config.get('eval_steps', 100),
            'save_steps': config.get('save_steps', 50),
            'resume_from_step': config.get('resume_from_step', 1),
            'resume_run_path': config.get('resume_run_path'),
            'token_budget': config.get('playbook_token_budget', 80000),
            'task_name': config.get('task_name', 'default'),
            'use_json_mode': config.get('json_mode', False),
            'no_ground_truth': config.get('no_ground_truth', False),
            'save_dir': config.get('save_dir', './results'),
            'test_workers': config.get('test_workers', 20),
            'track_generation_latency': config.get('track_generation_latency', False),
            'use_bulletpoint_analyzer': config.get('use_bulletpoint_analyzer', False),
            'bulletpoint_analyzer_threshold': config.get('bulletpoint_analyzer_threshold', 0.90),
            'use_rae': config.get('use_rae', False),
            'rae_top_k': config.get('rae_top_k', 10),
            'rae_retrieval_mode': config.get('rae_retrieval_mode', 'semantic'),
            'rae_random_seed': config.get('rae_random_seed', 42),
            'use_failure_memory': config.get('use_failure_memory', False),
            'failure_memory_top_k': config.get('failure_memory_top_k', 3),
            'failure_memory_mode': config.get('failure_memory_mode', 'legacy'),
            'stress_noise_rate': config.get('stress_noise_rate', 0.0),
            'stress_noise_mode': config.get('stress_noise_mode', 'replace'),
            'stress_noise_seed': config.get('stress_noise_seed'),
            'stress_noise_schedule': config.get('stress_noise_schedule', 'initial'),
            'stress_inject_interval': config.get('stress_inject_interval', 0),
            'use_adversarial': config.get('use_adversarial', False),
            'adversarial_frequency': config.get('adversarial_frequency', 10),
            'adversarial_num_candidates': config.get('adversarial_num_candidates', 5),
            'adversarial_verifier_min_confidence': config.get('adversarial_verifier_min_confidence', 0.80),
            'adversarial_verifier_max_ambiguity': config.get('adversarial_verifier_max_ambiguity', 0.20),
            'adversarial_mode': config.get('adversarial_mode', 'verified'),
        }

    def _inject_interval_stress_noise(
        self,
        *,
        epoch: int,
        step: int,
        config_params: Dict[str, Any],
        log_dir: str,
    ) -> Optional[Dict[str, Any]]:
        """Inject deterministic harmful noise into the in-memory Playbook only."""
        schedule = config_params['stress_noise_schedule']
        interval = int(config_params['stress_inject_interval'])
        noise_rate = float(config_params['stress_noise_rate'])
        if (
            noise_rate <= 0.0
            or schedule not in {'interval', 'both'}
            or interval <= 0
            or step % interval != 0
        ):
            return None

        base_seed = config_params['stress_noise_seed']
        if base_seed is None:
            base_seed = 42
        injection_seed = int(base_seed) + epoch * 1_000_000 + step
        self.playbook, manifest = corrupt_playbook(
            self.playbook,
            noise_rate=noise_rate,
            mode=config_params['stress_noise_mode'],
            seed=injection_seed,
        )
        manifest.update(
            {
                'epoch': epoch,
                'step': step,
                'schedule': schedule,
                'injection_seed': injection_seed,
            }
        )
        event_path = os.path.join(log_dir, 'playbook_stress_events.jsonl')
        with open(event_path, 'a', encoding='utf-8') as file:
            file.write(json.dumps(manifest, ensure_ascii=False) + '\n')
        if self.use_rae and self.playbook_retriever:
            self.playbook_retriever.update_index(self.playbook)
        print(
            '⚠️  Injected interval stress noise: '
            f"epoch={epoch}, step={step}, rate={manifest['noise_rate_realized']:.3f}, "
            f"mode={manifest['mode']}"
        )
        return manifest
    
    def _setup_paths(
        self,
        save_dir: str,
        task_name: str,
        mode: str,
        resume_run_path: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Setup logging paths and directories.
        
        Args:
            save_dir: Base path for saving results
            task_name: task name
            mode: 'offline', 'online', or 'eval_only'
            resume_run_path: Existing run folder to continue writing into
            
        Returns:
            Tuple of (usage_log_path, playbook_dir)
        """
        # Reuse an existing run folder when resuming so new artifacts append to
        # the same location as the initial playbook and prior logs.
        if resume_run_path:
            save_path = resume_run_path
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_folder = f"ace_run_{timestamp}_{task_name}_{mode}"
            save_path = os.path.join(save_dir, run_folder)

        os.makedirs(save_path, exist_ok=True)
        log_dir = os.path.join(save_path, "detailed_llm_logs")
        os.makedirs(log_dir, exist_ok=True)

        if mode == "eval_only":
            return save_path, log_dir

        usage_log_path = os.path.join(save_path, "bullet_usage_log.jsonl")
        playbook_dir = os.path.join(save_path, "intermediate_playbooks")
        os.makedirs(playbook_dir, exist_ok=True)
        
        return save_path, usage_log_path, playbook_dir, log_dir
    
    def run(
        self,
        mode: str,
        train_samples: Optional[List[Dict[str, Any]]] = None,
        val_samples: Optional[List[Dict[str, Any]]] = None,
        test_samples: Optional[List[Dict[str, Any]]] = None,
        data_processor = None,
        config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Main entrypoint for running ACE system in different modes.
        
        Args:
            mode: Run mode - 'offline', 'online', or 'eval_only'
            train_samples: Training samples (required for offline mode)
            val_samples: Validation samples (required for offline mode)
            test_samples: Test samples (required for online and eval_only modes)
            data_processor: Data processor instance for the task
            config: Configuration dictionary
            
        Returns:
            Dictionary with results depending on the mode
        """
        # Validate inputs
        if mode not in ['offline', 'online', 'eval_only']:
            raise ValueError(f"Invalid mode: {mode}. Must be 'offline', 'online', or 'eval_only'")
        
        if mode == 'offline' and (train_samples is None or val_samples is None):
            raise ValueError("Offline mode requires train_samples and val_samples")
        
        if mode == 'online' and test_samples is None:
            raise ValueError("Online mode requires test_samples")
        
        if mode == 'eval_only' and test_samples is None:
            raise ValueError("eval_only mode requires test_samples")
        
        # Extract configuration
        config_params = self._extract_config_params(config)
        task_name = config_params['task_name']
        save_dir = config_params['save_dir']
        resume_run_path = config_params.get('resume_run_path')
        
        # Setup paths based on mode
        if mode == 'eval_only':
            save_path, log_dir = self._setup_paths(
                save_dir, task_name, mode, resume_run_path=resume_run_path
            )
            usage_log_path = None
            playbook_dir = None
        else:
            save_path, usage_log_path, playbook_dir, log_dir = self._setup_paths(
                save_dir, task_name, mode, resume_run_path=resume_run_path
            )

        if self.failure_memory is not None:
            self.failure_memory.set_log_dir(log_dir, task_name=task_name)
        
        # Save configuration
        config_path = os.path.join(save_path, "run_config.json")
        with open(config_path, "w") as f:
            json.dump({
                "task_name": task_name,
                "mode": mode,
                "generator_model": self.generator.model,
                "reflector_model": self.reflector.model,
                "curator_model": self.curator.model,
                "adversarial_model": self.adversarial_agent.model if self.adversarial_agent else None,
                "config": config,
            }, f, indent=2)

        if resume_run_path:
            current_playbook_path = os.path.join(save_path, "current_playbook.txt")
            with open(current_playbook_path, "w") as f:
                f.write(self.playbook)
        
        # Print initial banner
        print(f"\n{'='*60}")
        print(f"ACE SYSTEM - {mode.upper().replace('_', ' ')} MODE")
        print(f"{'='*60}")
        print(f"Task: {task_name}")
        if mode == 'offline':
            print(f"Train samples: {len(train_samples)}")
            print(f"Validation samples: {len(val_samples)}")
            if test_samples:
                print(f"Test samples: {len(test_samples)}")
        elif mode == 'online':
            print(f"Test samples (used for training and testing): {len(test_samples)}")
        else:  # eval_only
            print(f"Test samples: {len(test_samples)}")
        print(f"{'='*60}\n")
        
        # Execute based on mode
        results = {}
        
        if mode == 'offline':
            # OFFLINE MODE WORKFLOW
            # 1. Run initial test if test_samples provided
            if test_samples:
                print(f"\n{'='*60}")
                print(f"INITIAL TEST (before training)")
                print(f"{'='*60}\n")
                initial_test_results = self._run_test(
                    test_samples=test_samples,
                    data_processor=data_processor,
                    playbook=self.playbook,
                    config=config,
                    log_dir=log_dir,
                    save_path=save_path,
                    prefix="initial"
                )
                results['initial_test_results'] = initial_test_results
                print(f"Initial Test Accuracy: {initial_test_results['accuracy']:.3f}\n")
            
            # 2. Run offline training
            print(f"\n{'='*60}")
            print(f"STARTING OFFLINE TRAINING")
            print(f"{'='*60}\n")
            training_results = self._offline_train(
                train_samples=train_samples,
                val_samples=val_samples,
                data_processor=data_processor,
                config=config,
                save_path=save_path,
                usage_log_path=usage_log_path,
                playbook_dir=playbook_dir,
                log_dir=log_dir,
                resume_run_path=resume_run_path
            )
            results['training_results'] = training_results
            
            # 3. Run final test if test_samples provided
            if test_samples:
                print(f"\n{'='*60}")
                print(f"FINAL TEST (with best playbook)")
                print(f"{'='*60}\n")
                final_test_results = self._run_test(
                    test_samples=test_samples,
                    data_processor=data_processor,
                    playbook=self.best_playbook,
                    config=config,
                    log_dir=log_dir,
                    save_path=save_path,
                    prefix="final"
                )
                results['final_test_results'] = final_test_results
                print(f"Final Test Accuracy: {final_test_results['accuracy']:.3f}\n")
        
        elif mode == 'online':
            # ONLINE MODE WORKFLOW
            # 1. Run initial test
            print(f"\n{'='*60}")
            print(f"INITIAL TEST (before training)")
            print(f"{'='*60}\n")
            initial_test_results = self._run_test(
                test_samples=test_samples,
                data_processor=data_processor,
                playbook=self.playbook,
                config=config,
                log_dir=log_dir,
                save_path=save_path,
                prefix="initial"
            )
            results['initial_test_results'] = initial_test_results
            print(f"Initial Test Accuracy: {initial_test_results['accuracy']:.3f}\n")
            
            # 2. Run online training and testing
            print(f"\n{'='*60}")
            print(f"STARTING ONLINE TRAIN AND TEST")
            print(f"{'='*60}\n")
            online_results = self._online_train_and_test(
                test_samples=test_samples,
                data_processor=data_processor,
                config=config,
                save_path=save_path,
                usage_log_path=usage_log_path,
                playbook_dir=playbook_dir,
                log_dir=log_dir
            )
            results['online_test_results'] = online_results
        
        else:  # eval_only
            # EVAL ONLY MODE WORKFLOW
            print(f"\n{'='*60}")
            print(f"RUNNING TEST")
            print(f"{'='*60}\n")
            test_results = self._run_test(
                test_samples=test_samples,
                data_processor=data_processor,
                playbook=self.playbook,
                config=config,
                log_dir=log_dir,
                save_path=save_path,
                prefix="test"
            )
            results['test_results'] = test_results
        
        # Save consolidated results
        final_results_path = os.path.join(save_path, "final_results.json")
        with open(final_results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        # Print final summary
        print(f"\n{'='*60}")
        print(f"RUN COMPLETE")
        print(f"{'='*60}")
        print(f"Mode: {mode.upper().replace('_', ' ')}")
        if mode == 'offline':
            print(f"Best Validation Accuracy: {results['training_results']['best_validation_accuracy']:.3f}")
            if test_samples:
                print(f"Initial Test Accuracy: {results['initial_test_results']['accuracy']:.3f}")
                print(f"Final Test Accuracy: {results['final_test_results']['accuracy']:.3f}")
        elif mode == 'online':
            print(f"Initial Test Accuracy: {results['initial_test_results']['accuracy']:.3f}")
            print(f"Final Test Accuracy: {results['online_test_results']['accuracy']:.3f}")
        else:  # eval_only
            print(f"Test Accuracy: {results['test_results']['accuracy']:.3f}")
        print(f"Results saved to: {save_path}")
        print(f"{'='*60}\n")
        
        return results
    
    def _run_test(
        self,
        test_samples: List[Dict[str, Any]],
        data_processor,
        playbook: str,
        config: Dict[str, Any],
        log_dir: str,
        save_path: str,
        prefix: str = "test"
    ) -> Dict[str, Any]:
        """
        Run testing
        
        Args:
            test_samples: List of test samples
            data_processor: Data processor instance for the task
            playbook: Playbook to use for testing
            config: Configuration dictionary
            log_dir: Directory for detailed logs
            save_path: Path to save results
            prefix: Prefix for saved files (e.g., 'initial', 'final', 'test')
            
        Returns:
            Dictionary with test results
        """
        config_params = self._extract_config_params(config)
        use_json_mode = config_params['use_json_mode']
        test_workers = config_params['test_workers']
        measure_generation_latency = config_params['track_generation_latency']
        
        test_results, test_error_log = evaluate_test_set(
            data_processor,
            self.generator,
            playbook,
            test_samples,
            self.max_tokens,
            log_dir,
            max_workers=test_workers,
            use_json_mode=use_json_mode,
            retriever=self.playbook_retriever,
            measure_generation_latency=measure_generation_latency,
        )

        # Save test results
        test_results_path = os.path.join(save_path, f"{prefix}_test_results.json")
        with open(test_results_path, "w") as f:
            json.dump({
                "test_results": test_results,
                "error_log": test_error_log,
            }, f, indent=2)
        
        return test_results

    def _run_adversarial_episode(
        self,
        step_id: str,
        epoch: int,
        step: int,
        usage_log_path: str,
        log_dir: str,
        config_params: Dict[str, Any],
        total_samples: int,
        base_question: str,
        base_context: str,
        base_target: str,
        data_processor,
    ) -> Optional[Dict[str, Any]]:
        """
        Run a single adversarial episode and update playbook if needed.
        """
        if not self.use_adversarial or not self.adversarial_agent:
            return None

        adversarial_frequency = config_params['adversarial_frequency']
        if adversarial_frequency <= 0 or step % adversarial_frequency != 0:
            return None

        print("\n--- Running Adversarial Agent ---")

        use_json_mode = config_params['use_json_mode']
        no_ground_truth = config_params['no_ground_truth']
        token_budget = config_params['token_budget']
        task_name = config_params['task_name']

        episode_meta = {
            "step_id": step_id,
            "epoch": epoch,
            "step": step,
        }
        log_adversarial_episode(log_dir, {
            **episode_meta,
            "event": "episode_started",
            "task_name": task_name,
        })

        attack, pipeline_info = self.adversarial_agent.generate_attack(
            playbook=self.playbook,
            task_name=task_name,
            recent_question=base_question,
            recent_context=base_context,
            recent_target=base_target,
            use_json_mode=use_json_mode,
            call_id=f"{step_id}_adv_generate",
            log_dir=log_dir,
        )

        if not attack:
            log_adversarial_episode(log_dir, {
                **episode_meta,
                "event": "pipeline_rejected",
                "reason": pipeline_info.get("rejection_reason", "unknown"),
                "pipeline": pipeline_info.get("pipeline"),
                "selected_candidate_id": pipeline_info.get("selected_candidate_id"),
                "reflector_status": "not_run",
                "curator_status": "not_run",
            })
            return None

        adversarial_pipeline = pipeline_info.get("pipeline", "unknown")
        is_legacy_adversarial = adversarial_pipeline == "legacy-single-attack"
        adv_question = attack.get("question", "")
        adv_context = attack.get("context", "")
        adv_target = attack.get("target", "")
        attack_rationale = attack.get("attack_rationale", "")
        vulnerability_hint = attack.get("vulnerability_hint", "")
        candidate_id = attack.get("candidate_id", "")
        attack_category = attack.get("attack_category", "")
        verifier_confidence = attack.get("verifier_confidence", 0.0)
        selection_score = attack.get("selection_score", 0.0)

        adv_response, adv_bullet_ids, executor_call_info = self.generator.generate(
            question=adv_question,
            playbook=self.playbook,
            context=adv_context,
            reflection="(empty)",
            use_json_mode=use_json_mode,
            call_id=f"{step_id}_adv_exec",
            log_dir=log_dir,
            retriever=self.playbook_retriever,
        )

        adv_answer = extract_answer(adv_response)
        adv_correct = data_processor.answer_is_correct(adv_answer, adv_target)
        reflection_content = "(empty)"
        failure_memory_id = None
        log_adversarial_episode(log_dir, {
            **episode_meta,
            "event": "executor_result",
            "pipeline": adversarial_pipeline,
            "candidate_id": candidate_id,
            "question": adv_question,
            "target": adv_target,
            "predicted_answer": adv_answer,
            "is_correct": adv_correct,
            "bullet_ids": adv_bullet_ids,
            "call_id": executor_call_info.get("call_id"),
            "total_time": executor_call_info.get("total_time"),
            "prompt_num_tokens": executor_call_info.get("prompt_num_tokens"),
            "response_num_tokens": executor_call_info.get("response_num_tokens"),
        })

        if not adv_correct:
            playbook_bullets = extract_playbook_bullets(
                self.playbook, adv_bullet_ids
            )
            if is_legacy_adversarial:
                environment_feedback = "Adversarial test: predicted answer does not match adversarial target."
                if attack_rationale:
                    environment_feedback += f" Intended trap: {attack_rationale}"
                if vulnerability_hint:
                    environment_feedback += f" Vulnerability hint: {vulnerability_hint}"
            else:
                environment_feedback = (
                    "Adversarial test: predicted answer does not match the independently "
                    f"verified target (verifier confidence={verifier_confidence:.3f}). "
                    "Diagnose the failure independently."
                )

            reflection_content, bullet_tags, reflector_call_info = self.reflector.reflect(
                question=adv_question,
                reasoning_trace=adv_response,
                predicted_answer=adv_answer,
                ground_truth=adv_target if not no_ground_truth else None,
                environment_feedback=environment_feedback,
                bullets_used=playbook_bullets,
                use_ground_truth=not no_ground_truth,
                use_json_mode=use_json_mode,
                call_id=f"{step_id}_adv_reflect",
                log_dir=log_dir,
                failure_memory=self.failure_memory,
            )
            log_adversarial_episode(log_dir, {
                **episode_meta,
                "event": "reflector_result",
                "pipeline": adversarial_pipeline,
                "status": "completed",
                "candidate_id": candidate_id,
                "reflection": reflection_content,
                "bullet_tags": bullet_tags,
                "call_id": reflector_call_info.get("call_id"),
                "total_time": reflector_call_info.get("total_time"),
                "prompt_num_tokens": reflector_call_info.get("prompt_num_tokens"),
                "response_num_tokens": reflector_call_info.get("response_num_tokens"),
            })

            if bullet_tags:
                self.playbook = update_bullet_counts(
                    self.playbook, bullet_tags
                )

            if self.failure_memory is not None and reflection_content not in ("(empty)", ""):
                try:
                    parsed = json.loads(reflection_content) if isinstance(reflection_content, str) else {}
                except (json.JSONDecodeError, TypeError):
                    parsed = {}
                if self.failure_memory.mode == "verified":
                    failure_memory_id = self.failure_memory.add_verified(
                        question=adv_question,
                        predicted_answer=adv_answer,
                        ground_truth=adv_target,
                        error_identification=parsed.get("error_identification", ""),
                        root_cause=parsed.get("root_cause_analysis", ""),
                        key_insight=parsed.get("key_insight", ""),
                        verification={
                            "verified": not is_legacy_adversarial,
                            "confidence": verifier_confidence,
                            "oracle_type": "adversarial_verifier",
                        },
                        evidence=[
                            f"verified_target={adv_target}",
                            f"observed_answer={adv_answer}",
                            f"selection_score={selection_score}",
                        ],
                        source="adversarial",
                        task_id=step_id,
                        playbook_refs=list(adv_bullet_ids),
                        vulnerability_id=attack.get("vulnerability_id", ""),
                        candidate_id=candidate_id,
                    )
                else:
                    failure_memory_id = self.failure_memory.add(
                        question=adv_question,
                        predicted_answer=adv_answer,
                        ground_truth=adv_target,
                        error_identification=parsed.get("error_identification", ""),
                        root_cause=parsed.get("root_cause_analysis", ""),
                        key_insight=parsed.get("key_insight", ""),
                    )

            print("--- Running Curator for Adversarial Report ---")
            stats = get_playbook_stats(self.playbook)
            if is_legacy_adversarial:
                question_context = (
                    f"Adversarial question: {adv_question}\n"
                    f"Context: {adv_context}\n"
                    f"Attack rationale: {attack_rationale}\n"
                    f"Vulnerability hint: {vulnerability_hint}"
                )
            else:
                question_context = (
                    f"Adversarial question: {adv_question}\n"
                    f"Context: {adv_context}\n"
                    f"Verified target: {adv_target}\n"
                    f"Attack category: {attack_category}"
                )
            self.playbook, self.next_global_id, curator_operations, curator_call_info = self.curator.curate(
                current_playbook=self.playbook,
                recent_reflection=reflection_content,
                question_context=question_context,
                current_step=step,
                total_samples=total_samples,
                token_budget=token_budget,
                playbook_stats=stats,
                use_ground_truth=not no_ground_truth,
                use_json_mode=use_json_mode,
                call_id=f"{step_id}_adv_curate",
                log_dir=log_dir,
                next_global_id=self.next_global_id,
                allowed_operations=self.curator_allowed_operations,
                delete_harmful_margin=self.delete_harmful_margin,
                delete_min_harmful=self.delete_min_harmful,
                merge_candidates=self._get_curator_merge_candidates(log_dir, f"{step_id}_adv_curate"),
            )
            if self.failure_memory is not None and self.failure_memory.mode == "verified":
                self.failure_memory.record_curator_result(
                    failure_memory_id,
                    curator_operations,
                    applied=bool(curator_operations),
                )
            log_adversarial_episode(log_dir, {
                **episode_meta,
                "event": "curator_result",
                "pipeline": adversarial_pipeline,
                "status": "completed" if curator_operations else "completed_no_operations",
                "candidate_id": candidate_id,
                "operations": curator_operations,
                "playbook_stats_before": stats,
                "playbook_stats_after": get_playbook_stats(self.playbook),
                "call_id": curator_call_info.get("call_id"),
                "total_time": curator_call_info.get("total_time"),
                "prompt_num_tokens": curator_call_info.get("prompt_num_tokens"),
                "response_num_tokens": curator_call_info.get("response_num_tokens"),
            })

            if self.use_bulletpoint_analyzer and self.bulletpoint_analyzer:
                self.playbook = self.bulletpoint_analyzer.analyze(
                    playbook=self.playbook,
                    threshold=self.bulletpoint_analyzer_threshold,
                    merge=True,
                    clustering="dbscan" if self.use_dbscan_merge else "pairwise",
                    dbscan_eps=self.dbscan_eps,
                    dbscan_min_samples=self.dbscan_min_samples,
                    log_dir=log_dir,
                    call_id=f"{step_id}_adv_hygiene",
                )

            if self.use_rae and self.playbook_retriever:
                self.playbook_retriever.update_index(self.playbook)
        else:
            log_adversarial_episode(log_dir, {
                **episode_meta,
                "event": "reflector_skipped",
                "pipeline": adversarial_pipeline,
                "status": "not_run",
                "candidate_id": candidate_id,
                "reason": "executor_answer_correct",
            })
            log_adversarial_episode(log_dir, {
                **episode_meta,
                "event": "curator_skipped",
                "pipeline": adversarial_pipeline,
                "status": "not_run",
                "candidate_id": candidate_id,
                "reason": "executor_answer_correct",
            })

        adversarial_sample = {
            "question": adv_question,
            "context": adv_context,
            "target": adv_target,
            "attack_rationale": attack_rationale,
            "vulnerability_hint": vulnerability_hint,
            "source": "adversarial",
            "candidate_id": candidate_id,
            "attack_category": attack_category,
            "verifier_confidence": verifier_confidence,
            "selection_score": selection_score,
        }
        log_adversarial_episode(
            log_dir,
            {
                **episode_meta,
                "event": "episode_completed",
                "pipeline": adversarial_pipeline,
                "question": adv_question,
                "context": adv_context,
                "target": adv_target,
                "predicted_answer": adv_answer,
                "is_correct": adv_correct,
                "attack_rationale": attack_rationale,
                "vulnerability_hint": vulnerability_hint,
                "bullet_ids": adv_bullet_ids,
                "candidate_id": candidate_id,
                "attack_category": attack_category,
                "verifier_confidence": verifier_confidence,
                "selection_score": selection_score,
            },
        )
        log_bullet_usage(
            usage_log_path, epoch, step, adversarial_sample, adv_bullet_ids,
            playbook=self.playbook,
            reflection_content=None if adv_correct else reflection_content,
            is_correct=adv_correct,
        )

        return {
            "question": adv_question,
            "context": adv_context,
            "target": adv_target,
            "predicted_answer": adv_answer,
            "is_correct": adv_correct,
            "attack_rationale": attack_rationale,
            "vulnerability_hint": vulnerability_hint,
            "candidate_id": candidate_id,
            "attack_category": attack_category,
            "verifier_confidence": verifier_confidence,
            "selection_score": selection_score,
        }
    
    def _train_single_sample(
        self,
        task_dict: Dict[str, Any],
        data_processor,
        step_id: str,
        epoch: int,
        step: int,
        usage_log_path: str,
        log_dir: str,
        config_params: Dict[str, Any],
        total_samples: int
    ) -> Tuple[str, str, Dict[str, Any]]:
        """
        Train on a single sample with reflection and curation.
        
        Args:
            task_dict: Sample dictionary with question, context, target
            data_processor: Data processor for evaluation
            step_id: Identifier string for this step (e.g., "train_e_1_s_10" or "online_train_w_1_s_5")
            epoch: Current epoch number
            step: Current step number
            usage_log_path: Path for bullet usage logging
            log_dir: Path for logging directory
            config_params: Configuration parameters dictionary
            total_samples: Total number of samples in dataset
            
        Returns:
            Tuple of (pre_train_answer, post_train_answer, tracking_dict)
        """
        # Extract configuration
        max_num_rounds = config_params['max_num_rounds']
        curator_frequency = config_params['curator_frequency']
        token_budget = config_params['token_budget']
        use_json_mode = config_params['use_json_mode']
        no_ground_truth = config_params['no_ground_truth']
        
        # Extract sample data
        question = task_dict.get("question", "")
        context = task_dict.get("context", "")
        target = task_dict.get("target", "")
        
        # STEP 1: Initial generation (pre-train)
        print("Generating initial answer...")
        gen_response, bullet_ids, call_info = self.generator.generate(
            question=question,
            playbook=self.playbook,
            context=context,
            reflection="(empty)",
            use_json_mode=use_json_mode,
            call_id=f"{step_id}_gen_initial",
            log_dir=log_dir,
            retriever=self.playbook_retriever
        )
        
        # Extract answer and check correctness
        final_answer = extract_answer(gen_response)
        is_correct = data_processor.answer_is_correct(final_answer, target)
        pre_train_answer = final_answer
        
        print(f"Correct: {is_correct}")
        
        # Log bullet usage
        log_bullet_usage(usage_log_path, epoch, step, task_dict, bullet_ids,
                       playbook=self.playbook, is_correct=is_correct)
        
        # Track pre-train result
        tracking_dict = {
            "pre_train_result": {
                "final_answer": final_answer,
                "is_correct": is_correct,
                "playbook_num_tokens": count_tokens(self.playbook),
                "playbook_length": len(self.playbook)
            }
        }
        
        reflection_content = "(empty)"
        failure_memory_id = None
        
        # STEP 2: Reflection and regeneration
        if not is_correct:
            # For incorrect answers, iterate reflection rounds. The verified
            # bank stores only after reflection has produced a root cause.
            for round_num in range(max_num_rounds):
                print(f"Reflection round {round_num + 1}/{max_num_rounds}")
                
                # Get bullets for reflector
                playbook_bullets = extract_playbook_bullets(
                    self.playbook, bullet_ids
                )
                
                # Reflect on error (with analogical context if available)
                reflection_content, bullet_tags, _ = self.reflector.reflect(
                    question=question,
                    reasoning_trace=gen_response,
                    predicted_answer=final_answer,
                    ground_truth=target if not no_ground_truth else None,
                    environment_feedback="Predicted answer does not match ground truth",
                    bullets_used=playbook_bullets,
                    use_ground_truth=not no_ground_truth,
                    use_json_mode=use_json_mode,
                    call_id=f"{step_id}_round_{round_num}",
                    log_dir=log_dir,
                    failure_memory=self.failure_memory,
                )
                
                # Update bullet counts
                if bullet_tags:
                    self.playbook = update_bullet_counts(
                        self.playbook, bullet_tags
                    )
                
                # Regenerate with reflection
                gen_response, bullet_ids, _ = self.generator.generate(
                    question=question,
                    playbook=self.playbook,
                    context=context,
                    reflection=reflection_content,
                    use_json_mode=use_json_mode,
                    call_id=f"{step_id}_post_reflect_round_{round_num}",
                    log_dir=log_dir,
                    retriever=self.playbook_retriever
                )
                
                final_answer = extract_answer(gen_response)
                
                if data_processor.answer_is_correct(final_answer, target):
                    print(f"Corrected after reflection round {round_num + 1}!")
                    is_correct = True
                    break

            # Store distilled insights from the last reflection into memory
            if self.failure_memory is not None and reflection_content not in ("(empty)", ""):
                try:
                    parsed = json.loads(reflection_content) if isinstance(reflection_content, str) else {}
                except (json.JSONDecodeError, TypeError):
                    parsed = {}
                if self.failure_memory.mode == "verified":
                    failure_memory_id = self.failure_memory.add_verified(
                        question=question,
                        predicted_answer=pre_train_answer,
                        ground_truth=target,
                        error_identification=parsed.get("error_identification", ""),
                        root_cause=parsed.get("root_cause_analysis", ""),
                        key_insight=parsed.get("key_insight", ""),
                        verification={
                            "verified": True,
                            "confidence": 1.0,
                            "oracle_type": "finance_ground_truth",
                        },
                        evidence=[
                            f"ground_truth={target}",
                            f"initial_answer={pre_train_answer}",
                        ],
                        source="finance",
                        task_id=step_id,
                        playbook_refs=list(bullet_ids),
                    )
                else:
                    failure_memory_id = self.failure_memory.add(
                        question=question,
                        predicted_answer=pre_train_answer,
                        ground_truth=target,
                        error_identification=parsed.get("error_identification", ""),
                        root_cause=parsed.get("root_cause_analysis", ""),
                        key_insight=parsed.get("key_insight", ""),
                    )

        else:
            # For correct answers - still run reflector to tag helpful bullets
            playbook_bullets = extract_playbook_bullets(
                self.playbook, bullet_ids
            )
            
            reflection_content, bullet_tags, _ = self.reflector.reflect(
                question=question,
                reasoning_trace=gen_response,
                predicted_answer=final_answer,
                ground_truth=target if not no_ground_truth else None,
                environment_feedback="Predicted answer matches ground truth",
                bullets_used=playbook_bullets,
                use_ground_truth=not no_ground_truth,
                use_json_mode=use_json_mode,
                call_id=f"{step_id}_reflect_on_correct",
                log_dir=log_dir,
                failure_memory=None,  # no memory lookup for correct answers
            )
            
            # Update bullet counts
            if bullet_tags:
                self.playbook = update_bullet_counts(
                    self.playbook, bullet_tags
                )
            
            # Log with reflection
            log_bullet_usage(usage_log_path, epoch, step, task_dict, bullet_ids,
                           playbook=self.playbook, 
                           reflection_content=reflection_content,
                           is_correct=is_correct)
        
        # STEP 3: Curator - Periodically update playbook
        if step % curator_frequency == 0:
            print(f"\n--- Running Curator at step {step} ---")
            
            stats = get_playbook_stats(self.playbook)
            
            self.playbook, self.next_global_id, operations, _ = self.curator.curate(
                current_playbook=self.playbook,
                recent_reflection=reflection_content,
                question_context=context,
                current_step=step,
                total_samples=total_samples,
                token_budget=token_budget,
                playbook_stats=stats,
                use_ground_truth=not no_ground_truth,
                use_json_mode=use_json_mode,
                call_id=step_id,
                log_dir=log_dir,
                next_global_id=self.next_global_id,
                allowed_operations=self.curator_allowed_operations,
                delete_harmful_margin=self.delete_harmful_margin,
                delete_min_harmful=self.delete_min_harmful,
                merge_candidates=self._get_curator_merge_candidates(log_dir, step_id),
            )
            if self.failure_memory is not None and self.failure_memory.mode == "verified":
                self.failure_memory.record_curator_result(
                    failure_memory_id,
                    operations,
                    applied=bool(operations),
                )
            
            # Run bulletpoint analyzer if enabled
            if self.use_bulletpoint_analyzer and self.bulletpoint_analyzer:
                print(f"  Running BulletpointAnalyzer (threshold={self.bulletpoint_analyzer_threshold})...")
                self.playbook = self.bulletpoint_analyzer.analyze(
                    playbook=self.playbook,
                    threshold=self.bulletpoint_analyzer_threshold,
                    merge=True,
                    clustering="dbscan" if self.use_dbscan_merge else "pairwise",
                    dbscan_eps=self.dbscan_eps,
                    dbscan_min_samples=self.dbscan_min_samples,
                    log_dir=log_dir,
                    call_id=f"{step_id}_hygiene",
                )

            # Rebuild RAE index with the updated playbook
            if self.use_rae and self.playbook_retriever:
                self.playbook_retriever.update_index(self.playbook)
        
        # STEP 4: Post-curator generation
        gen_response, _, _ = self.generator.generate(
            question=question,
            playbook=self.playbook,
            context=context,
            reflection="(empty)",
            use_json_mode=use_json_mode,
            call_id=f"{step_id}_post_curate",
            log_dir=log_dir,
            retriever=self.playbook_retriever
        )
        
        final_answer = extract_answer(gen_response)
        post_train_answer = final_answer
        
        post_train_is_correct = data_processor.answer_is_correct(final_answer, target)
        tracking_dict["post_train_result"] = {
            "final_answer": final_answer,
            "is_correct": post_train_is_correct,
            "playbook_num_tokens": count_tokens(self.playbook),
            "playbook_length": len(self.playbook)
        }

        adversarial_result = self._run_adversarial_episode(
            step_id=step_id,
            epoch=epoch,
            step=step,
            usage_log_path=usage_log_path,
            log_dir=log_dir,
            config_params=config_params,
            total_samples=total_samples,
            base_question=question,
            base_context=context,
            base_target=target,
            data_processor=data_processor,
        )
        if adversarial_result is not None:
            tracking_dict["adversarial_result"] = adversarial_result
        
        return pre_train_answer, post_train_answer, tracking_dict
    
    def _offline_train(
        self,
        train_samples: List[Dict[str, Any]],
        val_samples: List[Dict[str, Any]],
        data_processor,
        config: Dict[str, Any],
        save_path: str,
        usage_log_path: str,
        playbook_dir: str,
        log_dir: str,
        resume_run_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run offline training
        
        Args:
            train_samples: List of training samples
            val_samples: List of validation samples
            data_processor: Data processor instance for the task
            config: Configuration dictionary
            save_path: Path to save results
            usage_log_path: Path for bullet usage logging
            playbook_dir: Directory for intermediate playbooks
            log_dir: Directory for detailed logs
            
        Returns:
            Dictionary with training results
        """
        # Extract configuration using helper
        config_params = self._extract_config_params(config)
        task_name = config_params['task_name']
        num_epochs = config_params['num_epochs']
        eval_steps = config_params['eval_steps']
        save_steps = config_params['save_steps']
        resume_from_step = max(1, int(config_params.get('resume_from_step', 1)))
        test_workers = config_params['test_workers']
        use_json_mode = config_params['use_json_mode']
        curator_frequency = config_params['curator_frequency']

        if resume_from_step > len(train_samples) + 1:
            raise ValueError(
                f"resume_from_step={resume_from_step} exceeds dataset length {len(train_samples)}"
            )

        start_index = max(resume_from_step - 1, 0)
        train_samples_to_run = train_samples[start_index:]
        
        # Initialize tracking
        results = []
        pre_train_post_train_results = []
        error_logs = []
        best_accuracy = 0.0
        self.best_playbook = self.playbook

        print(f"Total epochs: {num_epochs}")
        print(f"Train samples per epoch: {len(train_samples)}")
        if start_index > 0:
            print(f"Resuming from sample {resume_from_step} (skipping first {start_index} samples)")
            print(f"Remaining samples per epoch: {len(train_samples_to_run)}")
        print(f"Val samples: {len(val_samples)}")
        print(f"Curator frequency: every {curator_frequency} steps")
        print(f"Evaluation frequency: every {eval_steps} steps\n")
        
        # Training loop
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch}/{num_epochs}")
            print(f"{'='*60}")
            
            epoch_answers_pre_train = []
            epoch_targets_pre_train = []
            epoch_answers_post_train = []
            epoch_targets_post_train = []
            
            for step, task_dict in enumerate(train_samples_to_run, start=resume_from_step):
                print(f"\n--- Step {step}/{len(train_samples)} ---")
                self._inject_interval_stress_noise(
                    epoch=epoch,
                    step=(epoch - 1) * len(train_samples) + step,
                    config_params=config_params,
                    log_dir=log_dir,
                )
                
                target = task_dict.get("target", "")
                
                # Use helper method for training single sample
                pre_train_answer, post_train_answer, tracking_dict = self._train_single_sample(
                    task_dict=task_dict,
                    data_processor=data_processor,
                    step_id=f"train_e_{epoch}_s_{step}",
                    epoch=epoch,
                    step=step,
                    usage_log_path=usage_log_path,
                    log_dir=log_dir,
                    config_params=config_params,
                    total_samples=len(train_samples)
                )
                
                # Collect answers for accuracy calculation
                epoch_answers_pre_train.append(pre_train_answer)
                epoch_targets_pre_train.append(target)
                epoch_answers_post_train.append(post_train_answer)
                epoch_targets_post_train.append(target)
                
                # Track pre-train and post-train results
                pre_train_post_train_result = {
                    "epoch": epoch,
                    "step": step,
                    "target": target,
                    **tracking_dict
                }
                pre_train_post_train_results.append(pre_train_post_train_result)

                global_step = (epoch - 1) * len(train_samples) + step
                if (
                    self.prune_unused_bullets
                    and global_step % self.prune_unused_interval == 0
                ):
                    playbook_before_prune = self.playbook
                    self.playbook, pruned_bullet_ids = prune_zero_evidence_bullets(self.playbook)
                    if pruned_bullet_ids:
                        print(
                            f"🧹 Pruned {len(pruned_bullet_ids)} unused bullets at "
                            f"global step {global_step}"
                        )
                        if self.use_rae and self.playbook_retriever:
                            self.playbook_retriever.update_index(self.playbook)
                    log_playbook_hygiene(
                        Path(log_dir).parent if log_dir else None,
                        {
                            "event": "unused_bullet_prune",
                            "epoch": epoch,
                            "step": step,
                            "global_step": global_step,
                            "interval": self.prune_unused_interval,
                            "pruned_bullet_ids": pruned_bullet_ids,
                            "pruned_count": len(pruned_bullet_ids),
                            "playbook_changed": self.playbook != playbook_before_prune,
                            "remaining_bullets": get_playbook_stats(self.playbook)["total_bullets"],
                        },
                    )
                
                # Save intermediate playbook
                if step % save_steps == 0:
                    intermediate_path = os.path.join(
                        playbook_dir, f"epoch_{epoch}_step_{step}_playbook.txt"
                    )
                    with open(intermediate_path, "w") as f:
                        f.write(self.playbook)
                
                # Periodic evaluation
                if step % eval_steps == 0:
                    print(f"\n{'='*40}")
                    print(f"EVALUATION AT EPOCH {epoch}, STEP {step}")
                    print(f"{'='*40}")
                    
                    # Compute training accuracies
                    pre_train_accuracy = data_processor.evaluate_accuracy(
                        epoch_answers_pre_train, epoch_targets_pre_train
                    )
                    post_train_accuracy = data_processor.evaluate_accuracy(
                        epoch_answers_post_train, epoch_targets_post_train
                    )
                    
                    # Validation evaluation
                    val_results = {}
                    if val_samples:
                        val_results, val_error_log = evaluate_test_set(
                            data_processor, self.generator, self.playbook,
                            val_samples, self.max_tokens, log_dir,
                            max_workers=test_workers, use_json_mode=use_json_mode,
                            retriever=self.playbook_retriever
                        )
                    
                    result = {
                        "epoch": epoch,
                        "step": step,
                        "train_result": {
                            "pre_train_accuracy": pre_train_accuracy,
                            "post_train_accuracy": post_train_accuracy
                        },
                        "val_result": val_results,
                        "playbook_num_tokens": count_tokens(self.playbook),
                        "playbook_length": len(self.playbook),
                        "playbook_stats": get_playbook_stats(self.playbook)
                    }
                    results.append(result)
                    error_logs.append({
                        "epoch": epoch,
                        "step": step,
                        "val_results": val_results,
                        "error_log": val_error_log
                    })

                    # Track best playbook
                    if val_results:
                        acc = val_results["accuracy"]
                        if acc > best_accuracy:
                            best_accuracy = acc
                            self.best_playbook = self.playbook
                            print(f"🎉 New best accuracy: {best_accuracy:.3f}")
                    
                    # Save results
                    results_path = os.path.join(save_path, "train_results.json")
                    with open(results_path, "w") as f:
                        json.dump({
                            "best_accuracy": best_accuracy,
                            "results": results,
                        }, f, indent=2)
                    
                    error_logs_path = os.path.join(save_path, "val_results.json")
                    with open(error_logs_path, "w") as f:
                        json.dump(error_logs, f, indent=2)
            
            # End of epoch - save final playbook
            epoch_playbook_path = os.path.join(
                playbook_dir, f"epoch_{epoch}_final_playbook.txt"
            )
            with open(epoch_playbook_path, "w") as f:
                f.write(self.playbook)

        # Save training results
        results_path = os.path.join(save_path, "train_results.json")
        with open(results_path, "w") as f:
            json.dump({
                "best_accuracy": best_accuracy,
                "results": results,
            }, f, indent=2)
        
        pre_train_post_train_results_path = os.path.join(save_path, "pre_train_post_train_results.json")
        with open(pre_train_post_train_results_path, "w") as f:
            json.dump(pre_train_post_train_results, f, indent=2)
        
        # Save final playbook
        final_playbook_path = os.path.join(save_path, f"final_playbook.txt")
        with open(final_playbook_path, "w") as f:
            f.write(self.playbook)

        if resume_run_path:
            current_playbook_path = os.path.join(save_path, "current_playbook.txt")
            with open(current_playbook_path, "w") as f:
                f.write(self.playbook)
        
        # Save best playbook
        best_playbook_path = os.path.join(save_path, f"best_playbook.txt")
        with open(best_playbook_path, "w") as f:
            f.write(self.best_playbook)
        
        print(f"\n{'='*60}")
        print(f"OFFLINE TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Best Validation Accuracy: {best_accuracy:.3f}")
        print(f"{'='*60}\n")

        return {"best_validation_accuracy": best_accuracy}

    
    def test(
        self,
        test_samples: List[Dict[str, Any]],
        data_processor,
        playbook,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run testing with the playbook (backward compatibility wrapper).
        
        Args:
            test_samples: List of test samples
            data_processor: Data processor instance for the task
            playbook: Playbook to be used for generator
            config: Configuration dictionary
            
        Returns:
            Dictionary with test results
        """
        # Temporarily set the playbook
        old_playbook = self.playbook
        self.playbook = playbook
        
        # Use the run method
        results = self.run(
            mode='eval_only',
            test_samples=test_samples,
            data_processor=data_processor,
            config=config
        )
        
        # Restore old playbook
        self.playbook = old_playbook
        
        # Return in the old format for backward compatibility
        return {
            "test_results": results['test_results'],
            "error_log": results.get('test_error_log', {}),
            "playbook": playbook
        }
    
    def _online_train_and_test(
        self,
        test_samples: List[Dict[str, Any]],
        data_processor,
        config: Dict[str, Any],
        save_path: str,
        usage_log_path: str,
        playbook_dir: str,
        log_dir: str
    ) -> Dict[str, Any]:
        """
        Run online training and testing
        
        Args:
            test_samples: List of samples to train and test on
            data_processor: Data processor instance for the task
            config: Configuration dictionary
            save_path: Path to save results
            usage_log_path: Path for bullet usage logging
            playbook_dir: Directory for intermediate playbooks
            log_dir: Directory for detailed logs
            
        Returns:
            Dictionary with training results, test results, and final playbook
        """
        # Extract configuration using helper
        config_params = self._extract_config_params(config)
        num_epochs = config_params['num_epochs']
        
        # Validate configuration
        if num_epochs != 1:
            raise ValueError(f"online_train_and_test requires num_epochs=1, got {num_epochs}")
        
        # Extract additional parameters
        curator_frequency = config_params['curator_frequency']
        task_name = config_params['task_name']
        save_steps = config_params['save_steps']
        use_json_mode = config_params['use_json_mode']
        test_workers = config_params['test_workers']
        online_eval_frequency = config.get('online_eval_frequency', 100)  # Get from config
        
        # Initialize tracking
        train_results = []
        pre_train_post_train_results = []
        
        # Test tracking - accumulate across all windows
        correct_count_sample_based = 0
        correct_count = 0
        total_count = 0
        all_test_errors = []
        window_test_results = []
        print(f"Total samples: {len(test_samples)}")
        print(f"Window size: {online_eval_frequency}")
        print(f"Number of windows: {(len(test_samples) + online_eval_frequency - 1) // online_eval_frequency}")
        print(f"Curator frequency: every {curator_frequency} steps")
        
        # Split samples into windows
        num_windows = (len(test_samples) + online_eval_frequency - 1) // online_eval_frequency
        
        epoch = 1  # Always 1 epoch
        global_step = 0
        
        for window_idx in range(num_windows):
            start_idx = window_idx * online_eval_frequency
            end_idx = min((window_idx + 1) * online_eval_frequency, len(test_samples))
            window_samples = test_samples[start_idx:end_idx]
            
            print(f"\n{'='*60}")
            print(f"WINDOW {window_idx + 1}/{num_windows}")
            print(f"Samples {start_idx} to {end_idx - 1}")
            print(f"{'='*60}")
            
            # =================================================================
            # STEP 1: TEST on window with current playbook (before training)
            # =================================================================
            print(f"\n--- Testing window {window_idx + 1} with current playbook ---")
            
            # Use evaluate_test_set for parallel evaluation
            window_test_results_dict, window_test_error_log = evaluate_test_set(
                data_processor,
                self.generator,
                self.playbook,
                window_samples,
                self.max_tokens,
                log_dir,
                max_workers=test_workers,
                use_json_mode=use_json_mode,
                retriever=self.playbook_retriever
            )
            
            # Extract results
            window_accuracy = window_test_results_dict['accuracy']
            window_correct = window_test_results_dict['correct']
            window_total = window_test_results_dict['total']
            correct_count_sample_based += window_correct
            correct_count += window_accuracy * window_total
            total_count += window_total
            
            # Add errors with window and global index information
            for error in window_test_error_log['errors']:
                all_test_errors.append({
                    "window": window_idx + 1,
                    "global_index": start_idx + error['index'],
                    "prediction": error['prediction'],
                    "ground_truth": error['ground_truth']
                })
            
            window_test_results.append({
                "window": window_idx + 1,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "window_accuracy": window_accuracy,
                "window_correct": window_correct,
                "window_total": window_total
            })
            
            # Calculate cumulative test accuracy so far
            cumulative_test_accuracy = correct_count / total_count
            
            print(f"Window {window_idx + 1} test accuracy: {window_accuracy:.3f}")
            print(f"Cumulative test accuracy so far: {cumulative_test_accuracy:.3f} "
                  f"({total_count} samples)")
            
            # =================================================================
            # STEP 2: TRAIN on window (same as offline_train)
            # =================================================================
            print(f"\n--- Training on window {window_idx + 1} ---")
            
            epoch_answers_pre_train = []
            epoch_targets_pre_train = []
            epoch_answers_post_train = []
            epoch_targets_post_train = []
            
            for local_step, task_dict in enumerate(window_samples):
                global_step += 1
                local_step += 1
                
                print(f"\n--- Window {window_idx + 1}, Step {local_step}/{len(window_samples)} "
                      f"(Global step {global_step}) ---")
                
                target = task_dict.get("target", "")
                
                # Use helper method for training single sample
                pre_train_answer, post_train_answer, tracking_dict = self._train_single_sample(
                    task_dict=task_dict,
                    data_processor=data_processor,
                    step_id=f"online_train_s_{global_step}",
                    epoch=epoch,
                    step=global_step,
                    usage_log_path=usage_log_path,
                    log_dir=log_dir,
                    config_params=config_params,
                    total_samples=len(test_samples)
                )
                
                # Collect answers for accuracy calculation
                epoch_answers_pre_train.append(pre_train_answer)
                epoch_targets_pre_train.append(target)
                epoch_answers_post_train.append(post_train_answer)
                epoch_targets_post_train.append(target)
                
                # Track pre-train and post-train results
                pre_train_post_train_result = {
                    "window": window_idx + 1,
                    "global_step": global_step,
                    "target": target,
                    **tracking_dict
                }
                pre_train_post_train_results.append(pre_train_post_train_result)
                
                # Save intermediate playbook
                if global_step % save_steps == 0:
                    intermediate_path = os.path.join(
                        playbook_dir, f"step_{global_step}_playbook.txt"
                    )
                    with open(intermediate_path, "w") as f:
                        f.write(self.playbook)
            
            # End of window - compute training accuracies for this window
            pre_train_accuracy = data_processor.evaluate_accuracy(
                epoch_answers_pre_train, epoch_targets_pre_train
            )
            post_train_accuracy = data_processor.evaluate_accuracy(
                epoch_answers_post_train, epoch_targets_post_train
            )
            
            window_train_result = {
                "window": window_idx + 1,
                "global_step": global_step,
                "train_result": {
                    "pre_train_accuracy": pre_train_accuracy,
                    "post_train_accuracy": post_train_accuracy
                },
                "cumulative_test_accuracy": cumulative_test_accuracy,
                "playbook_num_tokens": count_tokens(self.playbook),
                "playbook_length": len(self.playbook),
                "playbook_stats": get_playbook_stats(self.playbook)
            }
            train_results.append(window_train_result)
            
            print(f"\nWindow {window_idx + 1} training complete:")
            print(f"  Pre-train accuracy: {pre_train_accuracy:.3f}")
            print(f"  Post-train accuracy: {post_train_accuracy:.3f}")
            
            # Save window playbook
            window_playbook_path = os.path.join(
                playbook_dir, f"window_{window_idx + 1}_final_playbook.txt"
            )
            with open(window_playbook_path, "w") as f:
                f.write(self.playbook)
        
        # All windows complete
        print(f"\n{'='*60}")
        print(f"ONLINE TRAIN AND TEST COMPLETE")
        print(f"{'='*60}")
        
        # Calculate final cumulative test accuracy
        assert total_count == len(test_samples)
        final_test_accuracy = correct_count / total_count
        
        test_results = {
            "accuracy": final_test_accuracy,
            "correct": correct_count_sample_based,
            "total": total_count,
            "window_results": window_test_results
        }
        
        test_error_log = {
            "accuracy": final_test_accuracy,
            "errors": all_test_errors
        }

        # Save test results
        test_results_path = os.path.join(save_path, "test_results.json")
        with open(test_results_path, "w") as f:
            json.dump({
                "test_accuracy": final_test_accuracy,
                "test_results": test_results,
                "test_error_log": test_error_log
            }, f, indent=2)
        
        # Save training results (per window)
        train_results_path = os.path.join(save_path, "train_results.json")
        with open(train_results_path, "w") as f:
            json.dump({"train_results": train_results}, f, indent=2)
        
        # Save pre-train/post-train results
        pre_train_post_train_results_path = os.path.join(save_path, "pre_train_post_train_results.json")
        with open(pre_train_post_train_results_path, "w") as f:
            json.dump(pre_train_post_train_results, f, indent=2)
        
        # Save final playbook
        final_playbook_path = os.path.join(save_path, f"final_playbook.txt")
        with open(final_playbook_path, "w") as f:
            f.write(self.playbook)
        
        print(f"\n{'='*60}")
        print(f"ONLINE TRAINING AND TESTING COMPLETE")
        print(f"{'='*60}")
        print(f"Final Test Accuracy: {final_test_accuracy:.3f}")
        print(f"{'='*60}\n")
        
        return {
            "accuracy": final_test_accuracy,
            "correct": correct_count_sample_based,
            "total": total_count,
        }
