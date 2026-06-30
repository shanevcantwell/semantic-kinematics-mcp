#!/usr/bin/env bash
# Resumable full-corpus nv_embed @4096-d run.
#
# Idempotent and crash-resilient: safe to re-run or re-launch at any time.
# embed_corpus.py checkpoints per request-group (append mode, skip-already-done),
# so a killed or crashed run resumes from exactly where it stopped. This wrapper
# adds auto-restart with a TRUTHFUL completion signal: it counts distinct
# successfully-embedded items (embed_status.py), not raw checkpoint lines or the
# process exit code — because embed_corpus.py exits 0 even when items are marked
# ``_failed``. It stops on full success OR when successes stop increasing (so a
# persistent per-item failure neither spins forever nor is silently abandoned).
#
# Override via env:
#   CORPUS  input JSONL corpus (default: the thought-vault chunks)
#   CKPT    checkpoint == output JSONL (one {chunk_id, embedding} per line)
#   LOG     append-only run log
#
# Monitor:  tail -f "$LOG"
#           "$PY" scripts/embed_status.py "$CORPUS" --checkpoint "$CKPT"  # done failed pending total
# Stop:     kill the process; re-run this script later to resume.
set -uo pipefail

SK="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CORPUS="${CORPUS:-/srv/dev/shanevcantwell/thought-vault-integration/output/vectors/chunks.jsonl}"
CKPT="${CKPT:-$SK/.runs/nv4096/corpus_4096.jsonl}"
LOG="${LOG:-$SK/.runs/nv4096/embed.log}"
PY="$SK/.venv/bin/python"

mkdir -p "$(dirname "$CKPT")" "$(dirname "$LOG")"

# Echoes: "done failed pending total" (zeros on any error, so completion never
# false-positives on a transient status failure).
status() {
    "$PY" "$SK/scripts/embed_status.py" "$CORPUS" --checkpoint "$CKPT" 2>/dev/null || echo "0 0 0 0"
}

stall=0
last_done=-1
while true; do
    read -r done failed pending total <<<"$(status)"
    echo "[wrapper $(date -u +%FT%TZ)] starting pass; done=$done failed=$failed pending=$pending total=$total" | tee -a "$LOG"

    "$PY" "$SK/scripts/embed_corpus.py" "$CORPUS" \
        --checkpoint "$CKPT" --backend nv_embed >>"$LOG" 2>&1
    rc=$?

    read -r done failed pending total <<<"$(status)"
    echo "[wrapper $(date -u +%FT%TZ)] pass exited rc=$rc; done=$done failed=$failed pending=$pending total=$total" | tee -a "$LOG"

    # Fully embedded only when every embeddable item has a valid vector.
    if [ "$total" -gt 0 ] && [ "$done" -ge "$total" ]; then
        echo "[wrapper] DONE: $done/$total embedded in $CKPT" | tee -a "$LOG"
        break
    fi

    # Not complete: a crash (rc!=0) OR persistent per-item failures (rc==0 with
    # _failed items that do not clear on retry). Retry either way, but stop once
    # successes stop increasing -- never spin, never silently abandon.
    if [ "$done" -le "$last_done" ]; then
        stall=$((stall + 1))
    else
        stall=0
    fi
    last_done="$done"
    if [ "$stall" -ge 3 ]; then
        echo "[wrapper] STOP: no new successes across 3 passes (last rc=$rc); $done/$total done, $failed persistent failures, $pending pending. Needs attention." | tee -a "$LOG"
        exit 1
    fi
    echo "[wrapper] resuming in 10s (stall=$stall)" | tee -a "$LOG"
    sleep 10
done
