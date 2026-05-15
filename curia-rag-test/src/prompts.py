from __future__ import annotations

from .models import SearchResult


def format_evidence(results: list[SearchResult]) -> str:
    lines: list[str] = []
    for index, result in enumerate(results, start=1):
        chunk = result.chunk
        lines.append(
            f"[{index}] SOURCE_ID={chunk.parent_id} | "
            f"source={chunk.source} | date={chunk.date.isoformat()} | "
            f"score={result.score:.3f}\n"
            f"    title={chunk.title}\n"
            f"    {chunk.text}"
        )
    return "\n\n".join(lines)


def build_recommendation_prompt(unit: dict, evidence: list[SearchResult]) -> str:
    source_ids = [r.chunk.parent_id for r in evidence]
    return f"""You are evaluating whether current computing curriculum should be updated.

CS2023 unit:
Title: {unit["title"]}
Description: {unit.get("description", "")}
Current topics: {", ".join(unit.get("current_topics", []))}

Retrieved evidence (cite using SOURCE_ID values exactly as shown):
{format_evidence(evidence)}

Available SOURCE_IDs you may cite: {source_ids}

Return ONLY a JSON object with these exact fields:
- signal_strength: one of "high", "medium", "low"
- summary: 2-3 sentences. After every factual claim write the SOURCE_ID in parentheses, e.g. (jp_001).
- emerging_topics: array of concise topic strings (max 8)
- evidence_ids: array of SOURCE_ID strings you cited (must be from the available list above)

Rules:
- Only cite SOURCE_IDs from the available list above — never invent new IDs.
- Do not introduce facts not present in the retrieved evidence.
"""
