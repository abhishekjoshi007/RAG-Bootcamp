from __future__ import annotations

from .models import SearchResult


def format_evidence(results: list[SearchResult]) -> str:
    lines: list[str] = []
    for index, result in enumerate(results, start=1):
        chunk = result.chunk
        lines.append(
            f"[{index}] id={chunk.parent_id} | chunk={chunk.chunk_id} | "
            f"source={chunk.source} | date={chunk.date.isoformat()} | "
            f"similarity={result.similarity:.3f} | score={result.score:.3f}\n"
            f"    title={chunk.title}\n"
            f"    {chunk.text}"
        )
    return "\n\n".join(lines)


def build_recommendation_prompt(unit: dict, evidence: list[SearchResult]) -> str:
    return f"""You are evaluating whether current computing curriculum should be updated.

CS2023 unit:
Title: {unit["title"]}
Description: {unit.get("description", "")}
Current topics: {", ".join(unit.get("current_topics", []))}

Retrieved evidence:
{format_evidence(evidence)}

Return only a JSON object with these fields:
- signal_strength: one of "high", "medium", "low"
- summary: 2-3 sentences. Cite evidence ids after each factual claim.
- emerging_topics: array of concise topic strings
- evidence_ids: array of cited document ids

Do not introduce facts that are not supported by the retrieved evidence.
"""
