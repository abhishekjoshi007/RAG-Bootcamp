from __future__ import annotations

import re

from .models import Chunk, Document


TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def detokenize(tokens: list[str]) -> str:
    text = " ".join(tokens)
    return (
        text.replace(" ,", ",")
        .replace(" .", ".")
        .replace(" ;", ";")
        .replace(" :", ":")
        .replace("( ", "(")
        .replace(" )", ")")
    )


def chunk_document(doc: Document, max_tokens: int = 160, overlap: int = 30) -> list[Chunk]:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if overlap < 0 or overlap >= max_tokens:
        raise ValueError("overlap must be >= 0 and smaller than max_tokens")

    tokens = tokenize(doc.text)
    if not tokens:
        return []

    chunks: list[Chunk] = []
    step = max_tokens - overlap
    for start in range(0, len(tokens), step):
        chunk_tokens = tokens[start : start + max_tokens]
        chunks.append(
            Chunk(
                chunk_id=f"{doc.id}_c{len(chunks)}",
                parent_id=doc.id,
                title=doc.title,
                source=doc.source,
                date=doc.date,
                text=detokenize(chunk_tokens),
                chunk_index=len(chunks),
                metadata=doc.metadata,
            )
        )
        if start + max_tokens >= len(tokens):
            break
    return chunks


def chunk_documents(
    docs: list[Document], max_tokens: int = 160, overlap: int = 30
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_document(doc, max_tokens=max_tokens, overlap=overlap))
    return chunks
