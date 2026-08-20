#!/bin/bash
# INDEPENDENT DELIVERY GUARANTEE — deliberately dumb, deliberately separate.
#
# n8n does the intelligent work (brain, adaptation, generation). This does not
# depend on n8n, on the brain, or on today's build succeeding. It answers ONE
# question: did a paper reach the inbox today? If not, it sends the next
# validated PDF off the pre-built queue.
#
# Every failure mode above it - n8n down, pm2 dead, generator crashed, pool dry,
# validation failing - degrades buffer depth rather than skipping a day.
# Runs from launchd, so it survives reboots and n8n being uninstalled entirely.
set -uo pipefail
cd "$(dirname "$0")"

TODAY=$(date +%F)
MARK="out/daily/.sent_${TODAY}"
LOG="out/daily/watchdog.log"
PY=./.venv/bin/python
mkdir -p out/daily

log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }

# already delivered today -> nothing to do
[ -f "$MARK" ] && exit 0

# 1) preferred: today's own paper, if the pipeline produced one
PDF="out/daily/SSC_CGL_2026_Daily_${TODAY}.pdf"

# 2) fallback: next validated paper off the pre-built queue
if [ ! -f "$PDF" ]; then
  PDF=$($PY - <<'PYEOF' 2>/dev/null
import json, pathlib
q = pathlib.Path("out/delivery_queue.json")
if q.exists():
    d = json.loads(q.read_text())
    for e in d.get("pending", []):
        if pathlib.Path(e["pdf"]).exists():
            print(e["pdf"]); break
PYEOF
)
  [ -n "$PDF" ] && log "today's build missing - using queued paper $PDF"
fi

# 3) last resort: most recent PDF that exists at all
if [ -z "${PDF:-}" ] || [ ! -f "$PDF" ]; then
  PDF=$(ls -t out/daily/*.pdf 2>/dev/null | head -1)
  [ -n "$PDF" ] && log "queue empty - resending most recent paper $PDF"
fi

if [ -z "${PDF:-}" ] || [ ! -f "$PDF" ]; then
  log "CRITICAL: no PDF available by any route"
  exit 1
fi

if $PY src/send_mail.py --pdf "$PDF" --date "$TODAY" >> "$LOG" 2>&1; then
  touch "$MARK"
  log "delivered $(basename "$PDF")"
  # consume it from the queue if that is where it came from
  $PY - "$PDF" <<'PYEOF' 2>/dev/null
import json, pathlib, sys
p = sys.argv[1]; q = pathlib.Path("out/delivery_queue.json")
if q.exists():
    d = json.loads(q.read_text())
    keep = [e for e in d.get("pending", []) if e["pdf"] != p]
    if len(keep) != len(d.get("pending", [])):
        d["sent"] = d.get("sent", []) + [{"pdf": p}]
        d["pending"] = keep
        q.write_text(json.dumps(d, indent=2))
PYEOF
  exit 0
fi

log "SEND FAILED for $PDF"
exit 1
