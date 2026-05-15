from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from src.config import CORPUS_DIR, EMBED_MODEL, INDEX_PATH
from src.storage import build_index_from_corpus


def main() -> None:
    print(f"Building FAISS index with {EMBED_MODEL} …")
    index = build_index_from_corpus(CORPUS_DIR, model_name=EMBED_MODEL)
    index.save(INDEX_PATH)
    print(f"Indexed {len(index.chunks)} chunks → {INDEX_PATH}")


if __name__ == "__main__":
    main()
