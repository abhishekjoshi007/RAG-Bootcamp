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

# arXiv IDs look like ax_2501.00701v3 or axhist_2501.00701v3 — same paper at
# different revisions. The corpus snapshot pins one version per paper; a model
# citing a different version of the same paper is not a hallucinated citation.
_ARXIV_VERSION_RE = re.compile(r"^(ax|axhist)_(\d{4}\.\d{4,5})v\d+$", re.IGNORECASE)


def _canonical_id(source_id: str) -> str:
    """Strip arXiv version suffix so v1/v2/v4 of the same paper compare equal."""
    match = _ARXIV_VERSION_RE.match(source_id)
    if match:
        return f"{match.group(1).lower()}_{match.group(2)}"
    return source_id


def check_citations(
    recommendation: Recommendation,
    evidence: list[SearchResult],
) -> CitationCheck:
    retrieved_ids = sorted({result.chunk.parent_id for result in evidence})
    explicit_ids = set(recommendation.evidence_ids)
    inline_ids = set(ID_RE.findall(recommendation.summary))
    cited_ids = sorted(explicit_ids | inline_ids)
    retrieved_canonical = {_canonical_id(rid) for rid in retrieved_ids}
    missing_ids = sorted(
        cited_id for cited_id in cited_ids
        if _canonical_id(cited_id) not in retrieved_canonical
    )
    return CitationCheck(
        passed=not missing_ids,
        cited_ids=cited_ids,
        retrieved_ids=retrieved_ids,
        missing_ids=missing_ids,
    )
