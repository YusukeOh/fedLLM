#!/usr/bin/env bash
# ─── Phase 1: FOMC document collection + Fed vocabulary build ───
#
# Run inside tmux to survive SSH disconnection:
#   tmux new -s phase1
#   bash scripts/run_phase1_collection.sh
#   # Ctrl-B D to detach
#
# Re-attach later:
#   tmux attach -t phase1

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"
source venv/bin/activate

LOG_DIR="$PROJECT_DIR/results/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/phase1_collection_${TIMESTAMP}.log"

echo "========================================" | tee "$LOG_FILE"
echo "Phase 1: FOMC Document Collection"       | tee -a "$LOG_FILE"
echo "Started: $(date)"                        | tee -a "$LOG_FILE"
echo "Log: $LOG_FILE"                          | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# ── Step 1: Collect FOMC statements and minutes ──
echo ""                                         | tee -a "$LOG_FILE"
echo "[Step 1/3] Collecting FOMC documents..."  | tee -a "$LOG_FILE"
echo ""                                         | tee -a "$LOG_FILE"

python src/data/fomc.py \
    --output-dir data/raw/fomc \
    --start-year 1994 \
    --doc-types statements minutes \
    2>&1 | tee -a "$LOG_FILE"

STMT_COUNT=$(find data/raw/fomc/statements -name "*.txt" 2>/dev/null | wc -l)
MINS_COUNT=$(find data/raw/fomc/minutes -name "*.txt" 2>/dev/null | wc -l)
echo ""                                                              | tee -a "$LOG_FILE"
echo "Collected: $STMT_COUNT statements, $MINS_COUNT minutes"        | tee -a "$LOG_FILE"

# ── Step 2: Build Fed vocabulary (GPT-2 tokenizer, for dev) ──
echo ""                                                              | tee -a "$LOG_FILE"
echo "[Step 2/3] Building Fed vocabulary (GPT-2)..."                 | tee -a "$LOG_FILE"
echo ""                                                              | tee -a "$LOG_FILE"

python -c "
import logging, json
logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
from transformers import GPT2Tokenizer
from src.data.fed_vocabulary import load_fomc_texts, build_fed_vocabulary, save_vocabulary

texts = load_fomc_texts('data/raw/fomc', doc_types=('statements', 'minutes'))
if not texts:
    print('WARNING: No FOMC texts found. Skipping vocabulary build.')
    exit(0)

tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2')
vocab = build_fed_vocabulary(texts, tokenizer, min_doc_freq=5, max_vocab_size=5000)
save_vocabulary(vocab, 'data/vocabulary/fed_vocab_gpt2.json')

print(f'Top 30 Fed tokens:')
for tid, tok in zip(vocab['token_ids'][:30], vocab['token_strings'][:30]):
    print(f'  {tid:6d}  {tok}')
" 2>&1 | tee -a "$LOG_FILE"

# ── Step 3: Quick data availability check ──
echo ""                                                              | tee -a "$LOG_FILE"
echo "[Step 3/3] FRED-MD data availability check..."                 | tee -a "$LOG_FILE"
echo ""                                                              | tee -a "$LOG_FILE"

python scripts/check_data_availability.py 2>&1 | tee -a "$LOG_FILE"

# ── Summary ──
echo ""                                                              | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Phase 1 collection complete!"             | tee -a "$LOG_FILE"
echo "Finished: $(date)"                        | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo ""                                                              | tee -a "$LOG_FILE"
echo "Artifacts:"                                                    | tee -a "$LOG_FILE"
echo "  FOMC statements: data/raw/fomc/statements/"                  | tee -a "$LOG_FILE"
echo "  FOMC minutes:    data/raw/fomc/minutes/"                     | tee -a "$LOG_FILE"
echo "  Fed vocabulary:  data/vocabulary/fed_vocab_gpt2.json"        | tee -a "$LOG_FILE"
echo "  Log file:        $LOG_FILE"                                  | tee -a "$LOG_FILE"
