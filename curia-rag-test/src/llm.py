from __future__ import annotations

import json
import re
from collections import Counter

from .embedding import normalize_terms
from .models import Recommendation, SearchResult


STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "for",
    "from",
    "have",
    "into",
    "that",
    "the",
    "their",
    "this",
    "with",
    "applicants",
    "build",
    "candidates",
    "common",
    "engineer",
    "hiring",
    "includes",
    "project",
    "requires",
    "services",
    "should",
    "systems",
    "team",
    "toolkit",
    "will",
}

KEY_PHRASES = [
    "retrieval-augmented generation",
    "vector databases",
    "prompt evaluation",
    "model monitoring",
    "hallucination rates",
    "guardrails",
    "claim-level faithfulness",
    "citation precision",
    "adversarial robustness",
    "grounding checks",
    "software bill of materials",
    "sbom generation",
    "provenance verification",
    "dependency vulnerability scanning",
    "package integrity",
    "policy-as-code",
    "distributed tracing",
    "service-level objectives",
    "serverless functions",
    "resilience testing",
    "incident response",
    "slo burn rates",
]


def parse_json_response(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned.removeprefix("```json").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```").strip()
    if cleaned.endswith("```"):
        cleaned = cleaned.removesuffix("```").strip()
    return json.loads(cleaned)


class LocalGroundedGenerator:
    def generate(self, unit: dict, evidence: list[SearchResult]) -> Recommendation:
        if not evidence:
            return Recommendation(
                signal_strength="low",
                summary="The retrieved corpus did not provide enough evidence for a grounded recommendation.",
                emerging_topics=[],
                evidence_ids=[],
            )

        top = evidence[: min(4, len(evidence))]
        evidence_ids = []
        for result in top:
            if result.chunk.parent_id not in evidence_ids:
                evidence_ids.append(result.chunk.parent_id)

        avg_score = sum(result.score for result in top) / len(top)
        signal = "high" if avg_score >= 0.28 else "medium" if avg_score >= 0.14 else "low"
        topics = self._extract_topics(unit, top)
        topic_text = ", ".join(topics[:5]) if topics else "the retrieved technical evidence"
        citation_text = ", ".join(evidence_ids[:3])
        summary = (
            f"Retrieved evidence for {unit['title']} emphasizes {topic_text} ({citation_text}). "
            f"The signal strength is {signal} because the highest-ranked sources overlap with the unit description "
            f"and come from {len({item.chunk.source for item in top})} source type(s) ({citation_text})."
        )
        return Recommendation(
            signal_strength=signal,
            summary=summary,
            emerging_topics=topics[:8],
            evidence_ids=evidence_ids,
        )

    def _extract_topics(self, unit: dict, evidence: list[SearchResult]) -> list[str]:
        unit_terms = set(normalize_terms(unit["title"] + " " + unit.get("description", "")))
        combined_text = " ".join(result.chunk.text.lower() for result in evidence)
        topics = [phrase for phrase in KEY_PHRASES if phrase in combined_text]
        counts: Counter[str] = Counter()
        for result in evidence:
            for term in normalize_terms(result.chunk.text):
                if len(term) > 3 and term not in STOPWORDS and term not in unit_terms:
                    counts[term.replace("_", " ")] += 1
        for term, _ in counts.most_common(12):
            clean = self._clean_topic(term)
            if clean not in topics:
                topics.append(clean)
        return topics

    def _clean_topic(self, term: str) -> str:
        return re.sub(r"\s+", " ", term).strip()
