#!/usr/bin/env bash
# Resumable full-corpus nv_embed @4096-d run.
#
# TVI runs default to the verified canonical DB query door.  Set CORPUS for an
# explicit generic/non-TVI JSONL run instead; CORPUS and CORPUS_DB are mutually
# exclusive.  The checkpoint remains derived model-keyed JSONL vector state.
#
# Canonical overrides:
#   CORPUS_DB      promoted canonical corpus.db
#   INPUT_KEYS     whitespace-separated canonical input keys (default: base)
#   CANONICAL_ROOT optional trusted canonical root
# Generic override:
#   CORPUS         input JSONL corpus
# Common overrides:
#   CKPT, LOG, PY, MAXREQ, MAXCHUNK, SLEEP_SECONDS
set -uo pipefail

SK="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_CORPUS_DB="/srv/dev/shanevcantwell/thought-vault-integration/output/canonical/corpus.db"
CKPT="${CKPT:-$SK/.runs/nv4096/corpus_4096.jsonl}"
LOG="${LOG:-$SK/.runs/nv4096/embed.log}"
PY="${PY:-$SK/.venv/bin/python}"
MAXREQ="${MAXREQ:-8000}"
MAXCHUNK="${MAXCHUNK:-8000}"
SLEEP_SECONDS="${SLEEP_SECONDS:-10}"
STATUS_RETRIES="${STATUS_RETRIES:-3}"
STATUS_RETRY_DELAY_SECONDS="${STATUS_RETRY_DELAY_SECONDS:-1}"

if ! [[ "$STATUS_RETRIES" =~ ^[1-9][0-9]*$ ]]; then
    echo "[wrapper] ERROR: STATUS_RETRIES must be a positive integer" >&2
    exit 2
fi
if ! [[ "$STATUS_RETRY_DELAY_SECONDS" =~ ^[0-9]+$ ]]; then
    echo "[wrapper] ERROR: STATUS_RETRY_DELAY_SECONDS must be a non-negative integer" >&2
    exit 2
fi

if [[ -n "${CORPUS:-}" && -n "${CORPUS_DB:-}" ]]; then
    echo "[wrapper] ERROR: CORPUS and CORPUS_DB are mutually exclusive" >&2
    exit 2
fi

source_args=()
source_mode=""
if [[ -n "${CORPUS:-}" ]]; then
    source_mode="generic_jsonl"
    source_args+=("$CORPUS")
else
    source_mode="canonical_db"
    CORPUS_DB="${CORPUS_DB:-$DEFAULT_CORPUS_DB}"
    INPUT_KEYS="${INPUT_KEYS:-base}"
    read -r -a input_key_values <<<"$INPUT_KEYS"
    if [[ "${#input_key_values[@]}" -eq 0 ]]; then
        echo "[wrapper] ERROR: INPUT_KEYS must contain at least one key" >&2
        exit 2
    fi
    source_args+=(--corpus-db "$CORPUS_DB")
    if [[ -n "${CANONICAL_ROOT:-}" ]]; then
        source_args+=(--canonical-root "$CANONICAL_ROOT")
    fi
    for input_key in "${input_key_values[@]}"; do
        source_args+=(--input-key "$input_key")
    done
fi
if [[ -n "${TVI_CANONICAL_CORPUS_EXECUTABLE:-}" ]]; then
    source_args+=(--tvi-executable "$TVI_CANONICAL_CORPUS_EXECUTABLE")
fi

mkdir -p "$(dirname "$CKPT")" "$(dirname "$LOG")"

# Exit 3 from embed_status.py is the only retryable class: the canonical query
# process failed to complete. Integrity refusals use exit 2 and stop immediately.
run_status() {
    local status_output rc attempt delay
    for ((attempt = 1; attempt <= STATUS_RETRIES; attempt++)); do
        status_output=$("$PY" "$SK/scripts/embed_status.py" \
            "${source_args[@]}" --checkpoint "$CKPT" "$@" 2>>"$LOG")
        rc=$?
        if [[ "$rc" -eq 0 ]]; then
            printf '%s\n' "$status_output"
            return 0
        fi
        if [[ "$rc" -ne 3 ]]; then
            echo "[wrapper] ERROR: embed status refused (exit=$rc)" | tee -a "$LOG" >&2
            return "$rc"
        fi
        if (( attempt == STATUS_RETRIES )); then
            echo "[wrapper] ERROR: embed status transport failed after $attempt attempt(s)" | tee -a "$LOG" >&2
            return "$rc"
        fi
        delay=$((STATUS_RETRY_DELAY_SECONDS * (1 << (attempt - 1))))
        echo "[wrapper] status transport retry $((attempt + 1))/$STATUS_RETRIES in ${delay}s" | tee -a "$LOG" >&2
        sleep "$delay"
    done
}

# Verified-open/query failures are fatal after bounded transport retries; never
# fallback from DB to JSONL and never replace an integrity error with zeros.
if source_identity=$(run_status --source-summary); then
    :
else
    rc=$?
    echo "[wrapper] ERROR: could not verify source mode=$source_mode" | tee -a "$LOG" >&2
    exit "$rc"
fi
echo "[wrapper $(date -u +%FT%TZ)] source mode=$source_mode $source_identity" | tee -a "$LOG"

read_status() {
    local status_output
    status_output=$(run_status) || return $?
    if ! [[ "$status_output" =~ ^[0-9]+[[:space:]]+[0-9]+[[:space:]]+[0-9]+[[:space:]]+[0-9]+[[:space:]]*$ ]]; then
        echo "[wrapper] ERROR: embed status returned an invalid progress record" | tee -a "$LOG" >&2
        return 2
    fi
    printf '%s\n' "$status_output"
}

stall=0
last_done=-1
while true; do
    status_output=$(read_status) || {
        rc=$?
        echo "[wrapper] ERROR: cannot read embed status before the next pass" | tee -a "$LOG" >&2
        exit "$rc"
    }
    read -r done failed pending total <<<"$status_output"
    echo "[wrapper $(date -u +%FT%TZ)] starting pass; done=$done failed=$failed pending=$pending total=$total" | tee -a "$LOG"

    "$PY" "$SK/scripts/embed_corpus.py" "${source_args[@]}" \
        --checkpoint "$CKPT" --backend nv_embed \
        --max-tokens-per-request "$MAXREQ" --max-tokens-per-chunk "$MAXCHUNK" >>"$LOG" 2>&1
    rc=$?

    status_output=$(read_status) || {
        rc=$?
        echo "[wrapper] ERROR: cannot read embed status after the pass" | tee -a "$LOG" >&2
        exit "$rc"
    }
    read -r done failed pending total <<<"$status_output"
    echo "[wrapper $(date -u +%FT%TZ)] pass exited rc=$rc; done=$done failed=$failed pending=$pending total=$total" | tee -a "$LOG"

    if [[ "$total" -gt 0 && "$done" -ge "$total" ]]; then
        echo "[wrapper] DONE: $done/$total embedded in $CKPT" | tee -a "$LOG"
        break
    fi

    if [[ "$done" -le "$last_done" ]]; then
        stall=$((stall + 1))
    else
        stall=0
    fi
    last_done="$done"
    if [[ "$stall" -ge 3 ]]; then
        echo "[wrapper] STOP: no new successes across 3 passes (last rc=$rc); $done/$total done, $failed persistent failures, $pending pending. Needs attention." | tee -a "$LOG"
        exit 1
    fi
    echo "[wrapper] resuming in ${SLEEP_SECONDS}s (stall=$stall)" | tee -a "$LOG"
    sleep "$SLEEP_SECONDS"
done
