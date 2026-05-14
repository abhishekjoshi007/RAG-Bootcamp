from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.storage import build_index_from_corpus


def main() -> None:
    index = build_index_from_corpus(ROOT / "data" / "corpus")
    out_path = ROOT / "audit" / "local_index.pkl"
    index.save(out_path)
    print(f"Indexed {len(index.chunks)} chunks")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
