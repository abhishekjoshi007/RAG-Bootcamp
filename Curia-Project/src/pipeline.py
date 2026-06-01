from __future__ import annotations

import os
from pathlib import Path

from .agent_d_roadmap import RoadmapAgent
from .audit import AuditLog
from .cache import CacheLayer
from .config import LLM_MODEL, RETRIEVAL_K, USE_CACHE
from .grounding import CitationCheck, check_citations
from .indexing import FaissIndex, InMemoryIndex
from .llm import LocalGroundedGenerator, OpenAIGenerator
from .models import Recommendation, SearchResult
from .prompts import build_recommendation_prompt
from .query import build_query, build_query_from_learner
from .query_hash import LearnerQuery
from .retrieval import Retriever

_Generator = LocalGroundedGenerator | OpenAIGenerator


class CuriaRagPipeline:
    def __init__(
        self,
        index: FaissIndex | InMemoryIndex,
        audit_path: Path | None = None,
        source_quotas: dict[str, int] | None = None,
        generator: _Generator | None = None,
        cache: CacheLayer | None = None,
    ) -> None:
        self.retriever = Retriever(index)
        if generator is not None:
            self.generator: _Generator = generator
        elif os.environ.get("OPENAI_API_KEY"):
            self.generator = OpenAIGenerator()
        else:
            self.generator = LocalGroundedGenerator()
        self.audit = AuditLog(audit_path) if audit_path else None
        self.source_quotas = source_quotas
        if cache is not None:
            self.cache: CacheLayer | None = cache
        elif audit_path is not None:
            self.cache = CacheLayer(audit_path)
        else:
            self.cache = None
        self.agent_d = RoadmapAgent()

    def run(self, unit_or_query, k: int = RETRIEVAL_K) -> dict:
        if isinstance(unit_or_query, LearnerQuery):
            return self._run_learner(unit_or_query, k=k)
        return self._run_unit(unit_or_query, k=k)

    # ---------- Backward-compatible unit path (unchanged behavior) ----------
    def _run_unit(self, unit: dict, k: int = 8) -> dict:
        query = build_query(unit)
        evidence = self.retriever.retrieve(query, k=k, source_quotas=self.source_quotas)
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

    # ---------- LearnerQuery path (cache-aware, §5.2) ----------
    def _run_learner(self, learner_query: LearnerQuery, k: int = RETRIEVAL_K) -> dict:
        query_hash = learner_query.hash()
        if USE_CACHE and self.cache is not None:
            cached = self.cache.get_recommendation(query_hash)
            if cached is not None:
                return {**cached, "cache_status": "hit", "query_hash": query_hash}

        query_text = build_query_from_learner(learner_query)
        evidence = self.retriever.retrieve(query_text, k=k, source_quotas=self.source_quotas)

        relevant_skills = self._infer_relevant_skills(evidence)
        agent_a = agent_b = agent_c = resources = {}
        if self.cache is not None:
            agent_a = {s: self.cache.get_agent_a(s) for s in relevant_skills}
            agent_b = {
                s: {h: self.cache.get_agent_b(s, h) for h in (3, 12, 24)}
                for s in relevant_skills
            }
            agent_c = {s: self.cache.get_agent_c(s) for s in relevant_skills}
            resources = {
                s: self.cache.get_resources_for_skill(s, top_k=5) for s in relevant_skills
            }

        roadmap = self.agent_d.build_roadmap(
            learner_query=learner_query,
            evidence=evidence,
            agent_a=agent_a,
            agent_b=agent_b,
            agent_c=agent_c,
            resources=resources,
        )

        unit = self._unit_from_learner(learner_query, query_hash)
        prompt = build_recommendation_prompt(unit, evidence)
        recommendation = self.generator.generate(unit, evidence)
        citation_check = check_citations(recommendation, evidence)
        evidence_ids = [result.chunk.parent_id for result in evidence]
        recommendation_dict = self._serialize_recommendation(recommendation)

        if USE_CACHE and self.cache is not None and citation_check.passed:
            self.cache.set_recommendation(
                query_hash=query_hash,
                normalized_query=learner_query.normalized(),
                recommendation=recommendation_dict,
                evidence_ids=evidence_ids,
                llm_model=LLM_MODEL,
                citation_check_ok=True,
            )
            self.cache.link_recommendation_skills(query_hash, relevant_skills)

        audit_id = None
        if self.audit:
            audit_id = self.audit.write(unit, prompt, evidence, recommendation, citation_check)

        return {
            "recommendation": recommendation_dict,
            "evidence_ids": evidence_ids,
            "citation_check": self._serialize_citation_check(citation_check),
            "roadmap": roadmap,
            "audit_id": audit_id,
            "cache_status": "miss",
            "query_hash": query_hash,
        }

    @staticmethod
    def _unit_from_learner(learner_query: LearnerQuery, query_hash: str) -> dict:
        return {
            "id": query_hash,
            "title": learner_query.goal or learner_query.query_text or learner_query.program,
            "description": learner_query.query_text or learner_query.goal,
            "current_topics": list(learner_query.completed_skills),
        }

    def _infer_relevant_skills(self, evidence: list[SearchResult]) -> list[str]:
        from .forecasting import _TREND_DEFS

        tracked = set(_TREND_DEFS.keys())
        found: list[str] = []
        for result in evidence:
            haystack = f"{result.chunk.title} {result.chunk.text}".lower()
            for skill in tracked:
                if skill in haystack and skill not in found:
                    found.append(skill)
        return found

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
