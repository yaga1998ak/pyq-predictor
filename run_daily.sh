#!/bin/bash
# Single entry point for the n8n daily job.
# Emits one line of JSON on stdout: {"ok":bool,"pdf":path,"date":str,...}
#
# Pipeline: organs -> build (adaptive) -> VALIDATE -> pdf
# The validate step is a hard gate: a malformed paper is never emailed (§65).
set -uo pipefail
cd "$(dirname "$0")"

DATE="${1:-$(date +%F)}"
PY=./.venv/bin/python
LOG=out/daily/run_${DATE}.log
MD=out/daily/SSC_CGL_2026_Daily_${DATE}.md
mkdir -p out/daily

{
  echo "=== SSC CGL daily run $DATE ==="
  echo "--- organ routing ---";        $PY src/organs.py 2>&1
  echo "--- newspapers: PDF facts ---";          $PY src/newspaper_ingest.py --dir data/newspapers 2>&1 || true
  echo "--- newspapers: Downloads sweep ---";    $PY src/newspaper_ingest.py --dir "$HOME/Downloads" 2>&1 || true
  echo "--- current affairs: web pull ---";     $PY src/current_affairs.py --fetch 2>&1 || true
  echo "--- YAGA: think ---";                    $PY src/yaga.py 2>&1
  echo "--- YAGA: obsidian memory ---";          $PY src/yaga_memory.py 2>&1 || true
  echo "--- composition constraints ---";        $PY src/brain_compose.py 2>&1
  echo "--- ops brain: observe/decide/act ---";  $PY src/brain.py --apply 2>&1
  echo "--- coverage ledger ---";      $PY src/adaptive.py --report 2>&1
  echo "--- building paper ---";       $PY src/daily_run.py --date "$DATE" 2>&1
} > "$LOG" 2>&1

# hard gate
if ! $PY src/validate_daily.py --md "$MD" >> "$LOG" 2>&1; then
  REASON=$($PY src/validate_daily.py --md "$MD" 2>/dev/null \
           | python3 -c "import sys,json;print('; '.join(json.load(sys.stdin).get('errors',[]))[:300])" 2>/dev/null)
  printf '{"ok":false,"pdf":null,"date":"%s","note":"VALIDATION FAILED: %s","log":"%s"}\n' \
         "$DATE" "$REASON" "$LOG"
  exit 1
fi

# only render the PDF once the paper has passed
$PY src/daily_run.py --date "$DATE" --pdf >> "$LOG" 2>&1
PDF=$(grep -o '^PDF_PATH=.*' "$LOG" | tail -1 | cut -d= -f2-)

if [ -n "$PDF" ] && [ -f "$PDF" ]; then
  printf '{"ok":true,"pdf":"%s","date":"%s","bytes":%s,"log":"%s"}\n' \
         "$PDF" "$DATE" "$(wc -c < "$PDF" | tr -d ' ')" "$LOG"
else
  ERR=$(tail -5 "$LOG" | tr '\n' ' ' | sed 's/"/\\"/g')
  printf '{"ok":false,"pdf":null,"date":"%s","note":"%s","log":"%s"}\n' "$DATE" "$ERR" "$LOG"
  exit 1
fi
