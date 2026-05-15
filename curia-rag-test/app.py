from __future__ import annotations

import html
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

from src.config import AUDIT_DB_PATH, CORPUS_DIR, INDEX_PATH, SOURCE_QUOTAS, UNITS_FILE
from src.indexing import FaissIndex
from src.pipeline import CuriaRagPipeline
from src.storage import build_index_from_corpus, get_unit, load_units


ROOT = Path(__file__).resolve().parent
UNITS = load_units(UNITS_FILE)


def load_pipeline() -> CuriaRagPipeline:
    if INDEX_PATH.exists():
        index = FaissIndex.load(INDEX_PATH)
    else:
        print("Building FAISS index on first run (downloads all-mpnet-base-v2 once) …")
        index = build_index_from_corpus(CORPUS_DIR)
        index.save(INDEX_PATH)
    return CuriaRagPipeline(
        index,
        audit_path=AUDIT_DB_PATH,
        source_quotas=SOURCE_QUOTAS,
    )


PIPELINE = load_pipeline()


CSS = """
body { margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #18202a; background: #f7f7f4; }
main { max-width: 1180px; margin: 0 auto; padding: 28px; }
header { display: flex; align-items: baseline; justify-content: space-between; gap: 20px; border-bottom: 1px solid #d8d6ce; padding-bottom: 16px; }
h1 { font-size: 28px; margin: 0; letter-spacing: 0; }
.subtle { color: #65707d; }
form { display: flex; gap: 12px; align-items: center; margin: 24px 0; flex-wrap: wrap; }
select, button { font: inherit; padding: 10px 12px; border: 1px solid #b9bebf; border-radius: 6px; background: white; }
button { background: #184c5b; color: white; cursor: pointer; border-color: #184c5b; }
.grid { display: grid; grid-template-columns: 0.9fr 1.1fr; gap: 20px; align-items: start; }
section, .card { background: #ffffff; border: 1px solid #d8d6ce; border-radius: 8px; padding: 18px; }
h2 { font-size: 18px; margin: 0 0 12px; }
h3 { font-size: 15px; margin: 0 0 8px; }
.badge { display: inline-block; padding: 4px 8px; border-radius: 999px; background: #d9ece8; color: #184c5b; font-weight: 650; }
.evidence { display: grid; gap: 12px; }
.meta { font-size: 13px; color: #65707d; margin-bottom: 8px; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; background: #f2f3ef; padding: 12px; border-radius: 6px; }
@media (max-width: 860px) { .grid { grid-template-columns: 1fr; } main { padding: 18px; } }
"""


def render_page(unit_id: str = "cs_ai_01") -> str:
    unit = get_unit(UNITS, unit_id)
    result = PIPELINE.run(unit)
    recommendation = result["recommendation"]
    options = "\n".join(
        f'<option value="{item["id"]}" {"selected" if item["id"] == unit_id else ""}>{html.escape(item["title"])}</option>'
        for item in UNITS
    )
    evidence_html = "\n".join(
        f"""
        <article class="card">
          <h3>{html.escape(item["title"])}</h3>
          <div class="meta">{html.escape(item["parent_id"])} | {html.escape(item["source"])} | {html.escape(item["date"])} | score {item["score"]}</div>
          <p>{html.escape(item["text"])}</p>
        </article>
        """
        for item in result["evidence"]
    )
    topics = ", ".join(recommendation["emerging_topics"]) or "None"
    citation_status = "passed" if result["citation_check"]["passed"] else "failed"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CURIA RAG Test</title>
  <style>{CSS}</style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>CURIA RAG Test</h1>
        <div class="subtle">Retrieval, citation checks, and audit logging for curriculum recommendations</div>
      </div>
      <span class="badge">audit #{result["audit_id"]}</span>
    </header>
    <form method="post">
      <label for="unit_id">CS2023 unit</label>
      <select id="unit_id" name="unit_id">{options}</select>
      <button type="submit">Run RAG</button>
    </form>
    <div class="grid">
      <section>
        <h2>Recommendation</h2>
        <p><strong>Signal:</strong> {html.escape(recommendation["signal_strength"])}</p>
        <p>{html.escape(recommendation["summary"])}</p>
        <p><strong>Emerging topics:</strong> {html.escape(topics)}</p>
        <p><strong>Citation check:</strong> {html.escape(citation_status)}</p>
        <h2>Query</h2>
        <pre>{html.escape(result["query"])}</pre>
      </section>
      <section>
        <h2>Retrieved Evidence</h2>
        <div class="evidence">{evidence_html}</div>
      </section>
    </div>
  </main>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self._send(render_page())

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode()
        unit_id = parse_qs(body).get("unit_id", ["cs_ai_01"])[0]
        self._send(render_page(unit_id))

    def _send(self, body: str) -> None:
        encoded = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 8000), Handler)
    print("CURIA RAG Test running at http://127.0.0.1:8000")
    server.serve_forever()


if __name__ == "__main__":
    main()
