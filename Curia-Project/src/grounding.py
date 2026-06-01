from __future__ import annotations

import re
from dataclasses import dataclass

from .models import Recommendation, SearchResult


@dataclass(frozen=True)
class CitationCheck:
    passed: bool
    cited_ids: list[str]
    retrieved_ids: list[str]
    missing_ids: list[str]


# Source IDs include repo/job IDs such as jp_001 and arXiv-derived IDs such as
# axhist_2601.00150v3. Keep dots/hyphens inside the token so inline citation
# parsing does not turn a valid dotted ID into a false missing prefix.
ID_RE = re.compile(r"\b[a-z]{2,}[a-z0-9-]*(?:_[a-z0-9_.-]+)+\b", re.IGNORECASE)


def check_citations(
    recommendation: Recommendation,
    evidence: list[SearchResult],
) -> CitationCheck:
    retrieved_ids = sorted({result.chunk.parent_id for result in evidence})
    explicit_ids = set(recommendation.evidence_ids)
    inline_ids = set(ID_RE.findall(recommendation.summary))
    cited_ids = sorted(explicit_ids | inline_ids)
    missing_ids = sorted(cited_id for cited_id in cited_ids if cited_id not in retrieved_ids)
    return CitationCheck(
        passed=not missing_ids,
        cited_ids=cited_ids,
        retrieved_ids=retrieved_ids,
        missing_ids=missing_ids,
    )
