"""Verified adversarial curriculum for ACE.

Pipeline: vulnerability mining -> multi-candidate generation -> independent
verification -> deterministic selection of the highest-value valid attack.
"""

import json
from typing import Any, Dict, List, Optional, Tuple

from ..prompts.adversarial import (
    ATTACK_GENERATOR_PROMPT,
    ATTACK_VERIFIER_PROMPT,
    VULNERABILITY_MINER_PROMPT,
)
from llm import timed_llm_call
from logger import log_adversarial_episode
from playbook_utils import extract_json_from_text


class AdversarialAgent:
    """Create verified mock queries that expose playbook weaknesses."""

    def __init__(
        self,
        api_client,
        api_provider,
        model: str,
        max_tokens: int = 4096,
        num_candidates: int = 5,
        max_vulnerabilities: int = 5,
        verifier_min_confidence: float = 0.80,
        verifier_max_ambiguity: float = 0.20,
    ):
        self.api_client = api_client
        self.api_provider = api_provider
        self.model = model
        self.max_tokens = max_tokens
        self.num_candidates = max(1, num_candidates)
        self.max_vulnerabilities = max(1, max_vulnerabilities)
        self.verifier_min_confidence = self._score(verifier_min_confidence)
        self.verifier_max_ambiguity = self._score(verifier_max_ambiguity)

    @staticmethod
    def _score(value: Any, default: float = 0.0) -> float:
        try:
            return min(1.0, max(0.0, float(value)))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _clean_text(value: Any) -> str:
        return "" if value is None else str(value).strip()

    @staticmethod
    def _strict_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() == "true"
        return False

    def _call_json(
        self,
        prompt: str,
        role: str,
        call_id: str,
        log_dir: Optional[str],
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        response, call_info = timed_llm_call(
            self.api_client,
            self.api_provider,
            self.model,
            prompt,
            role=role,
            call_id=call_id,
            max_tokens=self.max_tokens,
            log_dir=log_dir,
            use_json_mode=True,
        )
        parsed = extract_json_from_text(response)
        if not isinstance(parsed, dict):
            log_adversarial_episode(log_dir, {
                "step_id": call_id,
                "event": "parse_failure",
                "stage": role,
                "model": self.model,
                "response_preview": self._clean_text(response)[:800],
            })
            return None, call_info
        return parsed, call_info

    def mine_vulnerabilities(
        self,
        playbook: str,
        task_name: str,
        recent_question: str,
        recent_context: str,
        recent_target: str,
        call_id: str,
        log_dir: Optional[str],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        prompt = VULNERABILITY_MINER_PROMPT.format(
            playbook=playbook,
            task_name=task_name,
            recent_question=recent_question,
            recent_context=recent_context,
            recent_target=recent_target,
            max_vulnerabilities=self.max_vulnerabilities,
        )
        parsed, info = self._call_json(prompt, "adversarial_miner", call_id, log_dir)
        raw_items = parsed.get("vulnerabilities", []) if parsed else []
        vulnerabilities = []
        for index, item in enumerate(raw_items[:self.max_vulnerabilities]):
            if not isinstance(item, dict) or not self._clean_text(item.get("description")):
                continue
            vulnerabilities.append({
                "id": self._clean_text(item.get("id")) or f"v{index + 1}",
                "type": self._clean_text(item.get("type")) or "unspecified",
                "description": self._clean_text(item.get("description")),
                "evidence": self._clean_text(item.get("evidence")),
                "severity": self._score(item.get("severity"), 0.5),
                "testability": self._score(item.get("testability"), 0.5),
                "target_bullet_ids": item.get("target_bullet_ids", []),
            })
        return vulnerabilities, info

    def generate_candidates(
        self,
        vulnerabilities: List[Dict[str, Any]],
        task_name: str,
        recent_question: str,
        recent_context: str,
        recent_target: str,
        call_id: str,
        log_dir: Optional[str],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        prompt = ATTACK_GENERATOR_PROMPT.format(
            task_name=task_name,
            vulnerabilities=json.dumps(vulnerabilities, ensure_ascii=False, indent=2),
            recent_question=recent_question,
            recent_context=recent_context,
            recent_target=recent_target,
            num_candidates=self.num_candidates,
        )
        parsed, info = self._call_json(prompt, "adversarial_generator", call_id, log_dir)
        raw_items = parsed.get("candidates", []) if parsed else []
        candidates = []
        required = ("question", "target", "target_derivation")
        for index, item in enumerate(raw_items[:self.num_candidates]):
            if not isinstance(item, dict) or any(not self._clean_text(item.get(k)) for k in required):
                continue
            candidate = {
                "candidate_id": self._clean_text(item.get("candidate_id")) or f"c{index + 1}",
                "vulnerability_id": self._clean_text(item.get("vulnerability_id")),
                "question": self._clean_text(item.get("question")),
                "context": self._clean_text(item.get("context")),
                "target": self._clean_text(item.get("target")),
                "target_derivation": self._clean_text(item.get("target_derivation")),
                "attack_rationale": self._clean_text(item.get("attack_rationale")),
                "vulnerability_hint": self._clean_text(item.get("vulnerability_hint")),
                "attack_category": self._clean_text(item.get("attack_category")) or "unspecified",
                "novelty": self._score(item.get("novelty"), 0.5),
                "learning_value": self._score(item.get("learning_value"), 0.5),
            }
            candidates.append(candidate)
        return candidates, info

    def verify_candidates(
        self,
        candidates: List[Dict[str, Any]],
        task_name: str,
        recent_target: str,
        call_id: str,
        log_dir: Optional[str],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        prompt = ATTACK_VERIFIER_PROMPT.format(
            task_name=task_name,
            recent_target=recent_target,
            candidates=json.dumps(candidates, ensure_ascii=False, indent=2),
        )
        parsed, info = self._call_json(prompt, "adversarial_verifier", call_id, log_dir)
        raw_items = parsed.get("verifications", []) if parsed else []
        by_id = {self._clean_text(v.get("candidate_id")): v for v in raw_items if isinstance(v, dict)}
        verified = []
        for candidate in candidates:
            verdict = by_id.get(candidate["candidate_id"], {})
            confidence = self._score(verdict.get("confidence"))
            ambiguity = self._score(verdict.get("ambiguity"), 1.0)
            independent_target = self._clean_text(verdict.get("independent_target"))
            target_matches = self._strict_bool(verdict.get("target_matches"))
            accepted = (
                self._strict_bool(verdict.get("valid"))
                and target_matches
                and bool(independent_target)
                and confidence >= self.verifier_min_confidence
                and ambiguity <= self.verifier_max_ambiguity
            )
            verified.append({
                **candidate,
                "verified": accepted,
                "verifier_confidence": confidence,
                "ambiguity": ambiguity,
                "independent_target": independent_target,
                "verification_reason": self._clean_text(verdict.get("reason")),
            })
        return verified, info

    @staticmethod
    def select_attack(verified_candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Choose a valid, novel, high-learning-value candidate deterministically."""
        accepted = [candidate for candidate in verified_candidates if candidate.get("verified")]
        if not accepted:
            return None

        def selection_score(candidate: Dict[str, Any]) -> float:
            score = (
                0.45 * float(candidate.get("verifier_confidence", 0.0))
                + 0.25 * float(candidate.get("learning_value", 0.0))
                + 0.20 * float(candidate.get("novelty", 0.0))
                - 0.10 * float(candidate.get("ambiguity", 1.0))
            )
            return score

        selected = max(accepted, key=lambda candidate: (selection_score(candidate), candidate["candidate_id"]))
        selected = dict(selected)
        selected["selection_score"] = round(selection_score(selected), 6)
        return selected

    def generate_attack(
        self,
        playbook: str,
        task_name: str,
        recent_question: str,
        recent_context: str,
        recent_target: str,
        use_json_mode: bool = False,  # kept for backwards compatibility
        call_id: str = "adv",
        log_dir: Optional[str] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """Run the complete verified adversarial pipeline."""
        del use_json_mode  # pipeline stages always require structured JSON
        stage_info: Dict[str, Any] = {"pipeline": "mine-generate-verify-select"}

        vulnerabilities, stage_info["miner"] = self.mine_vulnerabilities(
            playbook, task_name, recent_question, recent_context, recent_target,
            f"{call_id}_mine", log_dir,
        )
        if not vulnerabilities:
            stage_info["rejection_reason"] = "no_valid_vulnerabilities"
            return None, stage_info

        candidates, stage_info["generator"] = self.generate_candidates(
            vulnerabilities, task_name, recent_question, recent_context, recent_target,
            f"{call_id}_generate", log_dir,
        )
        if not candidates:
            stage_info["rejection_reason"] = "no_valid_candidates"
            return None, stage_info

        verified, stage_info["verifier"] = self.verify_candidates(
            candidates, task_name, recent_target, f"{call_id}_verify", log_dir,
        )
        selected = self.select_attack(verified)
        log_adversarial_episode(log_dir, {
            "step_id": call_id,
            "event": "pipeline_selection",
            "vulnerabilities": vulnerabilities,
            "candidates": verified,
            "selected_candidate_id": selected.get("candidate_id") if selected else None,
        })
        if selected is None:
            stage_info["rejection_reason"] = "no_candidate_passed_verification"
            return None, stage_info

        stage_info["selected_candidate_id"] = selected["candidate_id"]
        return selected, stage_info
