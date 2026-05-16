#!/usr/bin/env bash
# =============================================================================
# CURIA Module Verification — runs all 12 test steps in order.
# Stops immediately on first failure.
#
# Usage:
#   chmod +x scripts/verify_modules.sh
#   ./scripts/verify_modules.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Load env vars
if [[ -f ".env" ]]; then
  set -o allexport; source .env; set +o allexport
fi
export TOKENIZERS_PARALLELISM=false

BOLD='\033[1m'; GREEN='\033[0;32m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'

STEPS=(
  "test_01_config.py   | Config — params, env overrides, path types"
  "test_02_models.py   | Models — frozen dataclasses, immutability"
  "test_03_chunking.py | Chunking — split, overlap, parent_id"
  "test_04_embedding.py| Embedding — shape (768), L2 norm, semantic similarity"
  "test_05_indexing.py | FAISS Index — build, search, save/load round-trip"
  "test_06_retrieval.py| Retrieval — recency scoring, source quotas"
  "test_07_query.py    | Query — construction, topic toggle, HyDE"
  "test_08_prompts.py  | Prompts — evidence block, JSON instructions"
  "test_09_grounding.py| Grounding — citation check, hallucination detection"
  "test_10_audit.py    | Audit — SQLite write, read-back, ID increment"
  "test_11_llm.py      | LLM — OpenAI call, JSON parse, env guard"
  "test_12_pipeline.py | Pipeline — full end-to-end, all 3 CS2023 units"
)

echo -e "\n${BOLD}${CYAN}CURIA Module Verification${NC}"
echo -e "${CYAN}══════════════════════════════════════════${NC}\n"

PASSED=0; FAILED=0; STEP=1

for entry in "${STEPS[@]}"; do
  FILE=$(echo "$entry" | cut -d'|' -f1 | tr -d ' ')
  DESC=$(echo "$entry" | cut -d'|' -f2)

  echo -e "${BOLD}Step $STEP — $DESC${NC}"
  if python3 -m pytest "tests/$FILE" -q --tb=short 2>&1 | tail -3; then
    echo -e "${GREEN}✓ Step $STEP passed${NC}\n"
    ((PASSED++))
  else
    echo -e "${RED}✗ Step $STEP FAILED — stopping here${NC}"
    ((FAILED++))
    exit 1
  fi
  ((STEP++))
done

echo -e "${BOLD}${GREEN}══════════════════════════════════════════"
echo -e "  All $PASSED steps passed. System verified."
echo -e "══════════════════════════════════════════${NC}\n"
