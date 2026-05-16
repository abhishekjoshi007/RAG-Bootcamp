#!/usr/bin/env bash
# =============================================================================
# CURIA RAG Pipeline — full end-to-end runner
#
# Steps
#   1. Load environment variables from .env
#   2. Install / verify Python dependencies
#   3. Ingest corpus from all 12 sources
#   4. Build FAISS index (all-mpnet-base-v2)
#   5. Run RAG pipeline for all 3 CS2023 units (OpenAI generation)
#   6. Run all 4 evaluation scripts
#   7. Save all output to results/<timestamp>/
#
# Usage
#   chmod +x run_pipeline.sh
#   ./run_pipeline.sh                 # full run
#   ./run_pipeline.sh --skip-ingest   # skip corpus fetch, reuse existing docs
#   ./run_pipeline.sh --skip-ingest --skip-index  # just pipeline + eval
# =============================================================================

set -euo pipefail

# --------------------------------------------------------------------------
# Flags
# --------------------------------------------------------------------------
SKIP_INGEST=false
SKIP_INDEX=false

for arg in "$@"; do
  case $arg in
    --skip-ingest) SKIP_INGEST=true ;;
    --skip-index)  SKIP_INDEX=true  ;;
    *)
      echo "Unknown argument: $arg"
      echo "Usage: $0 [--skip-ingest] [--skip-index]"
      exit 1
      ;;
  esac
done

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="results/${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

LOG_FILE="${RESULTS_DIR}/run.log"

# --------------------------------------------------------------------------
# Logging helpers
# --------------------------------------------------------------------------
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log()   { echo -e "${CYAN}[$(date +%H:%M:%S)]${NC} $*" | tee -a "$LOG_FILE"; }
ok()    { echo -e "${GREEN}[$(date +%H:%M:%S)] ✓ $*${NC}" | tee -a "$LOG_FILE"; }
warn()  { echo -e "${YELLOW}[$(date +%H:%M:%S)] ⚠ $*${NC}" | tee -a "$LOG_FILE"; }
fail()  { echo -e "${RED}[$(date +%H:%M:%S)] ✗ $*${NC}" | tee -a "$LOG_FILE"; exit 1; }
header(){ echo -e "\n${BOLD}${CYAN}══════════════════════════════════════════${NC}"; \
          echo -e "${BOLD}${CYAN}  $*${NC}"; \
          echo -e "${BOLD}${CYAN}══════════════════════════════════════════${NC}\n"; }

# --------------------------------------------------------------------------
# Step 0 — Load .env
# --------------------------------------------------------------------------
header "Step 0 · Environment"

if [[ -f ".env" ]]; then
  set -o allexport
  # shellcheck disable=SC1091
  source .env
  set +o allexport
  ok "Loaded .env"
else
  warn ".env not found — copy .env.example to .env and fill in your keys"
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  fail "OPENAI_API_KEY is not set. Add it to .env and re-run."
fi
ok "OPENAI_API_KEY present"

[[ -n "${USAJOBS_API_KEY:-}" ]] && ok "USAJOBS_API_KEY present" || warn "USAJOBS_API_KEY not set — skipping federal jobs"
[[ -n "${GITHUB_TOKEN:-}"    ]] && ok "GITHUB_TOKEN present (higher rate limit)" || warn "GITHUB_TOKEN not set — GitHub limited to 60 req/hr"

# Prevent HuggingFace tokenizer fork deadlocks and macOS MPS/BLAS segfaults
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
ok "Runtime guards set (TOKENIZERS_PARALLELISM=false, OMP_NUM_THREADS=1)"

# --------------------------------------------------------------------------
# Step 1 — Python dependencies
# --------------------------------------------------------------------------
header "Step 1 · Dependencies"

PYTHON=$(command -v python3 || command -v python || fail "python3 not found")
log "Using Python: $PYTHON ($($PYTHON --version 2>&1))"

log "Installing / verifying requirements …"
$PYTHON -m pip install -q -r requirements.txt 2>&1 | tail -5 | tee -a "$LOG_FILE"
ok "Dependencies ready"

# --------------------------------------------------------------------------
# Step 2 — Corpus ingestion
# --------------------------------------------------------------------------
header "Step 2 · Corpus Ingestion"

if $SKIP_INGEST; then
  warn "--skip-ingest: using existing docs in data/corpus/"
  CORPUS_COUNT=$(ls data/corpus/*.json 2>/dev/null | wc -l | tr -d ' ')
  log "Existing documents: $CORPUS_COUNT"
else
  log "Fetching from all 12 sources …"
  log "  Job postings: Greenhouse (~4 000 companies), Lever (~3 000 companies),"
  log "  We Work Remotely, RemoteOK, Remotive, Arbeitnow, The Muse, HN Hiring, USAJOBS"
  log "  Technical:    arXiv (cs.*), Stack Exchange, GitHub READMEs"
  echo ""

  $PYTHON scripts/ingest_corpus.py --no-rebuild-index 2>&1 | tee -a "$LOG_FILE" | grep -E "Fetch|→|Done|ERROR|WARNING" || true

  CORPUS_COUNT=$(ls data/corpus/*.json 2>/dev/null | wc -l | tr -d ' ')
  ok "Corpus now contains $CORPUS_COUNT documents"
fi

echo "{\"corpus_doc_count\": $CORPUS_COUNT}" > "${RESULTS_DIR}/corpus_stats.json"

# --------------------------------------------------------------------------
# Step 3 — Build FAISS index
# --------------------------------------------------------------------------
header "Step 3 · FAISS Index (all-mpnet-base-v2)"

if $SKIP_INDEX && [[ -f "audit/faiss_index.pkl" ]]; then
  warn "--skip-index: reusing existing audit/faiss_index.pkl"
else
  log "Embedding $CORPUS_COUNT documents and building IndexFlatIP …"
  $PYTHON scripts/build_index.py 2>&1 | tee -a "$LOG_FILE"
  ok "FAISS index saved to audit/faiss_index.pkl"
fi

# --------------------------------------------------------------------------
# Step 4 — RAG pipeline (all 3 CS2023 units)
# --------------------------------------------------------------------------
header "Step 4 · RAG Pipeline (OpenAI gpt-4o-mini)"

UNITS=("cs_ai_01" "cs_sec_01" "cs_cloud_01")
UNIT_LABELS=("Generative AI and LLMs" "Software Supply Chain Security" "Cloud Native Systems")

for i in "${!UNITS[@]}"; do
  UNIT_ID="${UNITS[$i]}"
  UNIT_LABEL="${UNIT_LABELS[$i]}"
  OUT_FILE="${RESULTS_DIR}/pipeline_${UNIT_ID}.json"

  log "Running unit: $UNIT_ID — $UNIT_LABEL"
  if $PYTHON scripts/run_pipeline.py --unit-id "$UNIT_ID" --k 8 > "$OUT_FILE" 2>>"$LOG_FILE"; then
    SIGNAL=$(python3 -c "import json; d=json.load(open('${OUT_FILE}')); print(d['recommendation']['signal_strength'])" 2>/dev/null || echo "?")
    CITATION=$(python3 -c "import json; d=json.load(open('${OUT_FILE}')); print('passed' if d['citation_check']['passed'] else 'FAILED')" 2>/dev/null || echo "?")
    ok "$UNIT_ID → signal=$SIGNAL | citation_check=$CITATION | saved to $(basename $OUT_FILE)"
  else
    warn "$UNIT_ID pipeline failed — check $LOG_FILE"
  fi
done

# --------------------------------------------------------------------------
# Step 5 — Evaluation
# --------------------------------------------------------------------------
header "Step 5 · Evaluation"

# 5a — Retrieval eval (Recall@k, MRR, nDCG)
log "5a · Retrieval eval …"
$PYTHON eval/run_retrieval_eval.py 2>>"$LOG_FILE" \
  | tee "${RESULTS_DIR}/eval_retrieval.json" \
  | grep -E '"aggregate"|"recall|"mrr' | head -10 || true
ok "Retrieval eval saved"

# 5b — Faithfulness / citation grounding eval
log "5b · Faithfulness eval …"
$PYTHON eval/run_faithfulness_eval.py 2>>"$LOG_FILE" \
  | tee "${RESULTS_DIR}/eval_faithfulness.json" \
  | grep -E '"aggregate"|"citation|"claim' | head -10 || true
ok "Faithfulness eval saved"

# 5c — Answer relevance eval
log "5c · Relevance eval …"
$PYTHON eval/run_relevance_eval.py 2>>"$LOG_FILE" \
  | tee "${RESULTS_DIR}/eval_relevance.json" \
  | grep -E '"mean_rating|"passed' | head -5 || true
ok "Relevance eval saved"

# 5d — Adversarial robustness eval
log "5d · Adversarial eval …"
$PYTHON eval/run_adversarial_eval.py 2>>"$LOG_FILE" \
  | tee "${RESULTS_DIR}/eval_adversarial.json" \
  | grep -E '"avg_recall|"max_recall|"passed' | head -5 || true
ok "Adversarial eval saved"

# --------------------------------------------------------------------------
# Step 6 — Summary report
# --------------------------------------------------------------------------
header "Step 6 · Summary"

# Pass RESULTS_DIR via env var so the single-quoted heredoc body can read it
# (single-quoted heredoc prevents shell expansion — env var is the safe way in)
CURIA_RESULTS_DIR="$RESULTS_DIR" $PYTHON << 'PYEOF' | tee "${RESULTS_DIR}/summary.json" | python3 -m json.tool
import json, os, pathlib, sys
sys.path.insert(0, ".")

R = pathlib.Path(os.environ["CURIA_RESULTS_DIR"])

# Pull targets directly from config so they stay in sync with the codebase
from src.config import (
    EVAL_TARGET_RECALL_8,
    EVAL_TARGET_CITATION_PRECISION,
    EVAL_TARGET_CLAIM_GROUNDING,
    EVAL_TARGET_RELEVANCE_MEAN,
    EVAL_TARGET_ADVERSARIAL_DROP,
)


def load_last_json(name):
    p = R / name
    if not p.exists():
        return {}
    for line in reversed(p.read_text().splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except Exception:
            pass
    return {}


retrieval    = load_last_json("eval_retrieval.json")
faithfulness = load_last_json("eval_faithfulness.json")
relevance    = load_last_json("eval_relevance.json")
adversarial  = load_last_json("eval_adversarial.json")

agg_retrieval    = retrieval.get("aggregate", {})
agg_faithfulness = faithfulness.get("aggregate", {})
mean_relevance   = relevance.get("summary", {}).get("mean_rating")
max_drop         = adversarial.get("adversarial_summary", {}).get("max_recall_drop")

# Check each metric against its target
checks = {
    "recall@8":          (agg_retrieval.get("recall@8"),       EVAL_TARGET_RECALL_8),
    "citation_precision":(agg_faithfulness.get("citation_precision"), EVAL_TARGET_CITATION_PRECISION),
    "claim_grounding":   (agg_faithfulness.get("claim_grounding"),    EVAL_TARGET_CLAIM_GROUNDING),
    "relevance_mean":    (mean_relevance,                       EVAL_TARGET_RELEVANCE_MEAN),
    "adversarial_drop":  (None if max_drop is None else 1 - max_drop, 1 - EVAL_TARGET_ADVERSARIAL_DROP),
}

metrics_passed = {k: (v is not None and v >= t) for k, (v, t) in checks.items()}

summary = {
    "results_dir": str(R),
    "retrieval":   agg_retrieval,
    "faithfulness": agg_faithfulness,
    "relevance_mean": mean_relevance,
    "adversarial_max_drop": max_drop,
    "targets": {
        "recall@8_min":            EVAL_TARGET_RECALL_8,
        "citation_precision_min":  EVAL_TARGET_CITATION_PRECISION,
        "claim_grounding_min":     EVAL_TARGET_CLAIM_GROUNDING,
        "relevance_mean_min":      EVAL_TARGET_RELEVANCE_MEAN,
        "adversarial_max_drop":    EVAL_TARGET_ADVERSARIAL_DROP,
    },
    "targets_passed": metrics_passed,
    "all_targets_met": all(metrics_passed.values()),
}
print(json.dumps(summary, indent=2))
PYEOF

echo ""
ok "All results saved to: ${RESULTS_DIR}/"
echo ""
echo -e "${BOLD}Files written:${NC}"
ls -1 "$RESULTS_DIR"
echo ""
echo -e "${BOLD}Full log:${NC} $LOG_FILE"
echo ""
