#!/usr/bin/env bash
# Weekly QC entry point for cron, Airflow, or a Databricks job.
#
#   QC_URI=abfss://.../gold/fact QC_STORE=/data/qc.db ./weekly.sh
#
# Exit codes (from `qc weekly`):
#   0  PASS / PASS_WITH_EXPLANATION / ALREADY_PROCESSED
#   2  INVESTIGATE  (alert-worthy)
#   3  DATA_CONTRACT_FAILURE
#   1  execution error
set -euo pipefail

: "${QC_URI:?set QC_URI to the refreshed table uri}"
QC_BIN="${QC_BIN:-qc}"
QC_OUT="${QC_OUT:-reports/weekly}"
QC_STAGE="${QC_STAGE:-warehouse}"
QC_MIN_SAMPLES="${QC_MIN_SAMPLES:-9}"

args=(weekly --uri "$QC_URI" --stage "$QC_STAGE" --out "$QC_OUT"
      --min-samples "$QC_MIN_SAMPLES")

[[ -n "${QC_CONFIG:-}" ]] && args+=(--config "$QC_CONFIG")
[[ -n "${QC_STORE:-}" ]] && args+=(--store "$QC_STORE")
[[ -n "${QC_CALIBRATION_STORE:-}" ]] && args+=(--calibration-store "$QC_CALIBRATION_STORE")
[[ -n "${QC_EXPECTATIONS:-}" ]] && args+=(--expectations "$QC_EXPECTATIONS")
[[ -n "${QC_REFERENCE_URI:-}" ]] && args+=(--reference-uri "$QC_REFERENCE_URI")
[[ -n "${QC_REFERENCE_SPEC:-}" ]] && args+=(--reference-spec "$QC_REFERENCE_SPEC")
[[ -n "${QC_STORAGE_OPTIONS:-}" ]] && args+=(--storage-options "$QC_STORAGE_OPTIONS")
[[ "${QC_ALLOW_INVESTIGATE:-0}" == "1" ]] && args+=(--allow-investigate)

exec "$QC_BIN" "${args[@]}" "$@"
