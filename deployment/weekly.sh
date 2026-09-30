#!/usr/bin/env bash
# Weekly QC entry point for cron, Airflow, or a Databricks job.
#
#   QC_URI=abfss://.../gold/fact QC_STORE=/data/qc.db ./weekly.sh
#
# Credentials are read from the environment (QC_STORAGE_OPTIONS), never passed
# on the command line where `ps` could see them.
#
# Exit codes (from `qc weekly`):
#   0  PASS / PASS_WITH_EXPLANATION
#   2  INVESTIGATE  (alert-worthy)
#   3  DATA_CONTRACT_FAILURE
#   4  INCOMPLETE   (required evidence unavailable)
#   75 LOCKED       (another run holds the lock)
#   1  execution error
set -euo pipefail
umask 077

: "${QC_URI:?set QC_URI to the refreshed table uri}"
QC_BIN="${QC_BIN:-qc}"
QC_OUT="${QC_OUT:-$(pwd)/reports/weekly}"
QC_STAGE="${QC_STAGE:-warehouse}"
QC_LOG_DIR="${QC_LOG_DIR:-$QC_OUT/logs}"
QC_LOCK="${QC_LOCK:-$QC_OUT/.weekly.lock}"

mkdir -p "$QC_OUT" "$QC_LOG_DIR"
LOG_FILE="$QC_LOG_DIR/weekly-$(date -u +%Y%m%dT%H%M%SZ).log"

# Refuse to overlap with a concurrent run; the Python lock is the real guard,
# this keeps cron from queueing duplicate work.
exec 9>"$QC_LOCK"
if ! flock -n 9; then
    echo "weekly: another run holds $QC_LOCK; exiting" | tee -a "$LOG_FILE"
    exit 75
fi

args=(weekly --uri "$QC_URI" --stage "$QC_STAGE" --out "$QC_OUT")

[[ -n "${QC_CONFIG:-}" ]] && args+=(--config "$QC_CONFIG")
[[ -n "${QC_STORE:-}" ]] && args+=(--store "$QC_STORE")
[[ -n "${QC_EXPECTATIONS:-}" ]] && args+=(--expectations "$QC_EXPECTATIONS")
[[ -n "${QC_REFERENCE_URI:-}" ]] && args+=(--reference-uri "$QC_REFERENCE_URI")
[[ -n "${QC_REFERENCE_SPEC:-}" ]] && args+=(--reference-spec "$QC_REFERENCE_SPEC")
[[ -n "${QC_REFERENCE_VERSION:-}" ]] && args+=(--reference-version "$QC_REFERENCE_VERSION")
[[ -n "${QC_REFERENCE_STAGE:-}" ]] && args+=(--reference-stage "$QC_REFERENCE_STAGE")
[[ -n "${QC_NOTIFY:-}" ]] && args+=(--notify "$QC_NOTIFY")

set +e
"$QC_BIN" "${args[@]}" "$@" 2>&1 | tee -a "$LOG_FILE"
code=${PIPESTATUS[0]}
set -e
echo "weekly: exit=$code log=$LOG_FILE"
exit "$code"
