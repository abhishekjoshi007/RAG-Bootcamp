from __future__ import annotations

from pathlib import Path

from .audit import AuditLog
from .grounding import CitationCheck, check_citations
from .indexing import InMemoryIndex
from .llm import LocalGroundedGenerator
from .models import Recommendation, SearchResult
from .prompts import build_recommendation_prompt
from .query import build_query
from .retrieval import Retriever


class CuriaRagPipeline:
    def __init__(
        self,
        index: InMemoryIndex,
        audit_path: Path | None = None,
        source_quotas: dict[str, int] | None = None,
    ) -> None:
        self.retriever = Retriever(index)
        self.generator = LocalGroundedGenerator()
        self.audit = AuditLog(audit_path) if audit_path else None
        self.source_quotas = source_quotas

    def run(self, unit: dict, k: int = 8) -> dict:
        query = build_query(unit)
        evidence = self.retriever.retrieve(
            query,
            k=k,
            source_quotas=self.source_quotas,
        )
        prompt = build_recommendation_prompt(unit, evidence)
        recommendation = self.generator.generate(unit, evidence)
        citation_check = check_citations(recommendation, evidence)
        audit_id = None
        if self.audit:
            audit_id = self.audit.write(unit, prompt, evidence, recommendation, citation_check)
        return {
            "query": query,
            "prompt": prompt,
            "evidence": self._serialize_evidence(evidence),
            "recommendation": self._serialize_recommendation(recommendation),
            "citation_check": self._serialize_citation_check(citation_check),
            "audit_id": audit_id,
        }

    def _serialize_evidence(self, evidence: list[SearchResult]) -> list[dict]:
        return [
            {
                "chunk_id": result.chunk.chunk_id,
                "parent_id": result.chunk.parent_id,
                "title": result.chunk.title,
                "source": result.chunk.source,
                "date": result.chunk.date.isoformat(),
                "similarity": round(result.similarity, 4),
                "score": round(result.score, 4),
                "text": result.chunk.text,
            }
            for result in evidence
        ]

    def _serialize_recommendation(self, recommendation: Recommendation) -> dict:
        return {
            "signal_strength": recommendation.signal_strength,
            "summary": recommendation.summary,
            "emerging_topics": recommendation.emerging_topics,
            "evidence_ids": recommendation.evidence_ids,
        }

    def _serialize_citation_check(self, citation_check: CitationCheck) -> dict:
        return {
            "passed": citation_check.passed,
            "cited_ids": citation_check.cited_ids,
            "retrieved_ids": citation_check.retrieved_ids,
            "missing_ids": citation_check.missing_ids,
        }
