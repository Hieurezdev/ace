"""Failure memory for ACE with legacy and evidence-verified modes."""

from __future__ import annotations

import re
import json
import os
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import numpy as np

try:
    import faiss

    MEMORY_AVAILABLE = True
except ImportError:
    faiss = None
    MEMORY_AVAILABLE = False
    print("Warning: faiss not available for FailureMemoryBank.")


# General failure taxonomy shared by finance, classification, and adversarial
# tasks. These types are eligible for verified reflection-memory retrieval.
VERIFIED_FAILURE_TYPES = {
    "PLAYBOOK_GAP",
    "PLAYBOOK_MISAPPLICATION",
    "REASONING_ERROR",
    "CALCULATION_ERROR",
    "RETRIEVAL_ERROR",
    "VERIFICATION_ERROR",
    "INSTRUCTION_FOLLOWING_ERROR",
}

# These failures are logged by their owning pipeline but must not enter active
# reflection retrieval or trigger a Curator playbook update.
DIAGNOSTIC_ONLY_FAILURE_TYPES = {
    "MODEL_FORMAT_ERROR",
    "EXECUTION_ERROR",
    "ENVIRONMENT_ERROR",
    "INVALID_ATTACK",
    "AMBIGUOUS_INPUT",
}


class FailureMemoryBank:
    """In-memory failure bank.

    ``legacy`` preserves the original semantic Top-K behavior. ``verified``
    stores an evidence-bearing schema and retrieves in stages: eligibility
    filtering, semantic candidate generation, then lexical/root-cause and
    historical-usefulness reranking.
    """

    def __init__(
        self,
        encoder: Optional[Callable[[List[str]], np.ndarray]] = None,
        embedding_dim: int = 1024,
        top_k: int = 3,
        embedding_model_name: str = "BAAI/bge-m3",
        mode: str = "legacy",
        min_verifier_confidence: float = 0.8,
        min_retrieval_score: float = 0.2,
        candidate_multiplier: int = 4,
    ):
        if mode not in {"legacy", "verified"}:
            raise ValueError("failure memory mode must be 'legacy' or 'verified'")
        self.mode = mode
        self.embedding_dim = embedding_dim
        self.default_top_k = top_k
        self.embedding_model_name = embedding_model_name
        self.min_verifier_confidence = min_verifier_confidence
        self.min_retrieval_score = min_retrieval_score
        self.candidate_multiplier = max(1, candidate_multiplier)
        self._external_encoder = encoder
        self._standalone_model: Optional[Any] = None
        self._index: Optional[Any] = None
        self._entries: List[Dict[str, Any]] = []
        self._next_id = 1
        self._event_log_path: Optional[str] = None
        self._snapshot_path: Optional[str] = None

    def set_log_dir(self, log_dir: str, task_name: str = "") -> None:
        """Enable JSONL event logging and a durable public-memory snapshot."""
        os.makedirs(log_dir, exist_ok=True)
        self._event_log_path = os.path.join(log_dir, "failure_memory_events.jsonl")
        self._snapshot_path = os.path.join(
            os.path.dirname(log_dir), "failure_memory_v2.jsonl"
        )
        self._log_event(
            "initialized",
            {
                "task_name": task_name,
                "mode": self.mode,
                "top_k": self.default_top_k,
                "min_verifier_confidence": self.min_verifier_confidence,
                "min_retrieval_score": self.min_retrieval_score,
                "candidate_multiplier": self.candidate_multiplier,
            },
        )
        self._persist_snapshot()

    def _log_event(self, event: str, payload: Dict[str, Any]) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "mode": self.mode,
            "bank_size": self.size,
            **payload,
        }
        if self._event_log_path:
            with open(self._event_log_path, "a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _persist_snapshot(self) -> None:
        if not self._snapshot_path:
            return
        temporary_path = self._snapshot_path + ".tmp"
        with open(temporary_path, "w", encoding="utf-8") as file:
            for entry in self.entries:
                file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        os.replace(temporary_path, self._snapshot_path)

    def _load_standalone_model(self) -> None:
        if self._standalone_model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer

            print(f"[FailureMemory] Loading standalone embedding model: {self.embedding_model_name}")
            self._standalone_model = SentenceTransformer(self.embedding_model_name)
        except ImportError:
            print("⚠️  FailureMemoryBank: sentence-transformers unavailable; memory disabled.")

    def _encode(self, texts: List[str]) -> Optional[np.ndarray]:
        if self._external_encoder is not None:
            values = self._external_encoder(texts)
            values = np.asarray(values, dtype=np.float32)
            norms = np.linalg.norm(values, axis=1, keepdims=True)
            return values / np.maximum(norms, 1e-12)
        self._load_standalone_model()
        if self._standalone_model is None:
            return None
        return self._standalone_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

    def _rebuild_index(self) -> None:
        if not MEMORY_AVAILABLE or not self._entries:
            self._index = None
            return
        embeddings = np.stack([entry["_emb"] for entry in self._entries])
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings.astype(np.float32))
        self._index = index

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9_]+", (text or "").lower()))

    @classmethod
    def _lexical_score(cls, query: str, entry: Dict[str, Any]) -> float:
        query_tokens = cls._tokens(query)
        memory_tokens = cls._tokens(
            " ".join(
                [
                    entry.get("question", ""),
                    entry.get("root_cause", ""),
                    entry.get("error_identification", ""),
                    entry.get("key_insight", ""),
                ]
            )
        )
        if not query_tokens or not memory_tokens:
            return 0.0
        return len(query_tokens & memory_tokens) / len(query_tokens | memory_tokens)

    @staticmethod
    def _usefulness_score(entry: Dict[str, Any]) -> float:
        helpful = int(entry.get("times_helpful", 0))
        harmful = int(entry.get("times_harmful", 0))
        return (helpful + 1) / (helpful + harmful + 2)

    @staticmethod
    def _public(entry: Dict[str, Any]) -> Dict[str, Any]:
        return {key: value for key, value in entry.items() if not key.startswith("_")}

    def add(
        self,
        question: str,
        predicted_answer: str,
        ground_truth: str,
        error_identification: str = "",
        root_cause: str = "",
        key_insight: str = "",
    ) -> Optional[str]:
        """Store using the original schema. Disabled in verified mode."""
        if self.mode == "verified":
            print("[FailureMemory] Rejected unverified legacy add() in verified mode.")
            self._log_event("store_rejected", {"reason": "legacy_add_in_verified_mode"})
            return None
        embedding = self._encode([question])
        if embedding is None:
            return None
        failure_id = f"fmb-{self._next_id:06d}"
        self._next_id += 1
        self._entries.append(
            {
                "failure_id": failure_id,
                "question": question,
                "predicted_answer": predicted_answer,
                "ground_truth": ground_truth,
                "error_identification": error_identification,
                "root_cause": root_cause,
                "key_insight": key_insight,
                "_emb": embedding[0],
            }
        )
        self._rebuild_index()
        self._log_event("failure_stored", {"failure": self._public(self._entries[-1])})
        self._persist_snapshot()
        print(f"[FailureMemory] Stored legacy failure {failure_id} | bank_size={self.size}")
        return failure_id

    def add_verified(
        self,
        *,
        question: str,
        predicted_answer: str,
        ground_truth: str,
        error_identification: str,
        root_cause: str,
        key_insight: str,
        verification: Dict[str, Any],
        evidence: List[str],
        failure_type: str = "PLAYBOOK_GAP",
        source: str = "standard",
        task_id: str = "",
        playbook_refs: Optional[List[str]] = None,
        vulnerability_id: str = "",
        candidate_id: str = "",
    ) -> Optional[str]:
        """Store only a verified, evidence-grounded playbook failure."""
        if self.mode != "verified":
            return self.add(
                question,
                predicted_answer,
                ground_truth,
                error_identification,
                root_cause,
                key_insight,
            )
        try:
            confidence = float(verification.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        eligible = (
            verification.get("verified") is True
            and confidence >= self.min_verifier_confidence
            and failure_type in VERIFIED_FAILURE_TYPES
            and bool(evidence)
            and bool(ground_truth)
            and predicted_answer != ground_truth
        )
        failed_checks = []
        if verification.get("verified") is not True:
            failed_checks.append("not_verified")
        if confidence < self.min_verifier_confidence:
            failed_checks.append("confidence_below_threshold")
        if failure_type not in VERIFIED_FAILURE_TYPES:
            failed_checks.append("unsupported_failure_type")
        if not evidence:
            failed_checks.append("missing_evidence")
        if not ground_truth:
            failed_checks.append("missing_ground_truth")
        if predicted_answer == ground_truth:
            failed_checks.append("no_observed_mismatch")
        self._log_event(
            "verification_gate",
            {
                "task_id": task_id,
                "source": source,
                "failure_type": failure_type,
                "confidence": confidence,
                "accepted": eligible,
                "failed_checks": failed_checks,
                "evidence_count": len(evidence),
            },
        )
        if not eligible:
            print(
                "[FailureMemory] Rejected failure: verification/evidence/type gate failed "
                f"(type={failure_type}, confidence={confidence:.3f})."
            )
            return None

        retrieval_text = "\n".join(
            filter(None, [question, error_identification, root_cause, key_insight])
        )
        embedding = self._encode([retrieval_text])
        if embedding is None:
            return None
        failure_id = f"fmb-{self._next_id:06d}"
        self._next_id += 1
        now = datetime.now(timezone.utc).isoformat()
        entry = {
            "schema_version": 2,
            "failure_id": failure_id,
            "task_id": task_id,
            "source": source,
            "failure_type": failure_type,
            "status": "verified",
            "question": question,
            "predicted_answer": predicted_answer,
            "ground_truth": ground_truth,
            "expected_outcome": ground_truth,
            "observed_outcome": predicted_answer,
            "error_identification": error_identification,
            "root_cause": root_cause,
            "key_insight": key_insight,
            "evidence": list(evidence),
            "verification": {**verification, "confidence": confidence},
            "playbook_refs": list(playbook_refs or []),
            "vulnerability_id": vulnerability_id,
            "candidate_id": candidate_id,
            "curator_operations": [],
            "curator_applied": False,
            "times_retrieved": 0,
            "times_helpful": 0,
            "times_harmful": 0,
            "created_at": now,
            "updated_at": now,
            "_emb": embedding[0],
        }
        self._entries.append(entry)
        self._rebuild_index()
        self._log_event("failure_stored", {"failure": self._public(entry)})
        self._persist_snapshot()
        print(f"[FailureMemory] Stored verified failure {failure_id} | bank_size={self.size}")
        return failure_id

    def record_curator_result(
        self,
        failure_id: Optional[str],
        operations: List[Dict[str, Any]],
        applied: bool,
    ) -> bool:
        """Attach Curator output to the failure that caused the update."""
        if not failure_id:
            return False
        for entry in self._entries:
            if entry.get("failure_id") == failure_id:
                entry["curator_operations"] = list(operations or [])
                entry["curator_applied"] = bool(applied)
                entry["status"] = "curated" if applied else entry.get("status", "verified")
                entry["updated_at"] = datetime.now(timezone.utc).isoformat()
                self._log_event(
                    "curator_result_attached",
                    {
                        "failure_id": failure_id,
                        "applied": bool(applied),
                        "operation_count": len(operations or []),
                        "operations": list(operations or []),
                        "status": entry["status"],
                    },
                )
                self._persist_snapshot()
                return True
        self._log_event(
            "curator_result_orphaned",
            {"failure_id": failure_id, "operation_count": len(operations or [])},
        )
        return False

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        if not self._entries:
            self._log_event("retrieval_skipped", {"reason": "empty_bank", "query": query})
            return []
        k = min(top_k or self.default_top_k, len(self._entries))
        query_embedding = self._encode([query])
        if query_embedding is None:
            self._log_event("retrieval_skipped", {"reason": "encoding_unavailable", "query": query})
            return []

        self._log_event("retrieval_started", {"query": query, "requested_top_k": k})

        if self.mode == "legacy":
            if not MEMORY_AVAILABLE or self._index is None:
                return []
            scores, indices = self._index.search(query_embedding, k)
            results = []
            for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
                item = self._public(self._entries[idx])
                item.update({"similarity": float(score), "rank": rank})
                results.append(item)
            self._log_event(
                "retrieval_completed",
                {
                    "query": query,
                    "results": [
                        {"failure_id": item.get("failure_id"), "rank": item["rank"], "similarity": item["similarity"]}
                        for item in results
                    ],
                },
            )
            return results

        eligible = [
            entry
            for entry in self._entries
            if entry.get("status") in {"verified", "curated", "validated"}
            and entry.get("failure_type") in VERIFIED_FAILURE_TYPES
            and entry.get("verification", {}).get("verified") is True
        ]
        if not eligible:
            self._log_event(
                "eligibility_filter",
                {"query": query, "total": len(self._entries), "eligible": 0},
            )
            return []

        self._log_event(
            "eligibility_filter",
            {"query": query, "total": len(self._entries), "eligible": len(eligible)},
        )

        semantic = np.asarray([float(np.dot(query_embedding[0], e["_emb"])) for e in eligible])
        pool_size = min(len(eligible), max(k, k * self.candidate_multiplier))
        pool_indices = np.argsort(-semantic)[:pool_size]
        self._log_event(
            "semantic_candidates",
            {
                "query": query,
                "pool_size": pool_size,
                "candidates": [
                    {
                        "failure_id": eligible[int(idx)].get("failure_id"),
                        "semantic_score": float(semantic[int(idx)]),
                    }
                    for idx in pool_indices
                ],
            },
        )
        reranked = []
        for idx in pool_indices:
            entry = eligible[int(idx)]
            semantic_score = float(semantic[int(idx)])
            lexical_score = self._lexical_score(query, entry)
            usefulness_score = self._usefulness_score(entry)
            retrieval_score = (
                0.65 * semantic_score
                + 0.25 * lexical_score
                + 0.10 * usefulness_score
            )
            if retrieval_score >= self.min_retrieval_score:
                reranked.append((retrieval_score, semantic_score, lexical_score, entry))
        reranked.sort(key=lambda item: item[0], reverse=True)
        self._log_event(
            "candidates_reranked",
            {
                "query": query,
                "minimum_score": self.min_retrieval_score,
                "candidates": [
                    {
                        "failure_id": entry.get("failure_id"),
                        "retrieval_score": score,
                        "semantic_score": semantic_score,
                        "lexical_score": lexical_score,
                        "usefulness_score": self._usefulness_score(entry),
                    }
                    for score, semantic_score, lexical_score, entry in reranked
                ],
            },
        )

        results = []
        for rank, (score, semantic_score, lexical_score, entry) in enumerate(reranked[:k], start=1):
            entry["times_retrieved"] += 1
            item = self._public(entry)
            item.update(
                {
                    "rank": rank,
                    "similarity": semantic_score,
                    "lexical_score": lexical_score,
                    "retrieval_score": score,
                }
            )
            results.append(item)
        if results:
            print(
                f"[FailureMemory/verified] Retrieved {len(results)} of {len(eligible)} eligible "
                f"failures; top_score={results[0]['retrieval_score']:.3f}"
            )
        self._log_event(
            "retrieval_completed",
            {
                "query": query,
                "results": [
                    {
                        "failure_id": item.get("failure_id"),
                        "rank": item["rank"],
                        "retrieval_score": item["retrieval_score"],
                    }
                    for item in results
                ],
            },
        )
        self._persist_snapshot()
        return results

    @staticmethod
    def format_for_prompt(similar_failures: List[Dict[str, Any]]) -> str:
        if not similar_failures:
            return "(No sufficiently relevant verified failures found)"
        lines: List[str] = []
        for failure in similar_failures:
            score = failure.get("retrieval_score", failure.get("similarity", 0.0))
            lines.append(
                f"--- Similar Failure #{failure['rank']} "
                f"(id={failure.get('failure_id', 'legacy')}, score={score:.3f}) ---"
            )
            lines.append(f"Question: {failure.get('question', '')}")
            lines.append(f"Wrong Answer: {failure.get('predicted_answer', '')}")
            if failure.get("ground_truth"):
                lines.append(f"Ground Truth: {failure['ground_truth']}")
            if failure.get("error_identification"):
                lines.append(f"Error: {failure['error_identification']}")
            if failure.get("root_cause"):
                lines.append(f"Root Cause: {failure['root_cause']}")
            if failure.get("key_insight"):
                lines.append(f"Key Insight: {failure['key_insight']}")
            if failure.get("evidence"):
                lines.append(f"Verified Evidence: {failure['evidence']}")
            if failure.get("curator_operations"):
                lines.append(f"Curator Fix: {failure['curator_operations']}")
            lines.append("")
        return "\n".join(lines).strip()

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> List[Dict[str, Any]]:
        """Public snapshot for tests and experiment analysis."""
        return [self._public(entry) for entry in self._entries]
