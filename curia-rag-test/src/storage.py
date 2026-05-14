from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from .chunking import chunk_documents
from .indexing import InMemoryIndex
from .models import Document


def load_documents(corpus_dir: Path) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(corpus_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        docs.append(
            Document(
                id=payload["id"],
                title=payload["title"],
                source=payload["source"],
                date=date.fromisoformat(payload["date"]),
                text=payload["text"],
                metadata=payload.get("metadata", {}),
            )
        )
    return docs


def load_units(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def get_unit(units: list[dict], unit_id: str) -> dict:
    for unit in units:
        if unit["id"] == unit_id:
            return unit
    raise KeyError(f"Unknown unit id: {unit_id}")


def build_index_from_corpus(corpus_dir: Path) -> InMemoryIndex:
    docs = load_documents(corpus_dir)
    chunks = chunk_documents(docs)
    return InMemoryIndex.build(chunks)
