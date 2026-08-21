#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
KV_DIR="${KV_DIR:-$ROOT_DIR/kv_compression}"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

MODEL_CONFIG_NAME="${1:-qwen2_14B_config}"
shift || true
EXTRA_ARGS=("$@")

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
MIN_NEW_TOKENS="${MIN_NEW_TOKENS:-128}"
COMMON_GENERATE_ARGS=(--max_new_tokens "$MAX_NEW_TOKENS" --min_new_tokens "$MIN_NEW_TOKENS")

OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/kv_mode_outputs}"
mkdir -p "$OUT_DIR"

log() {
  printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"
}

run_mode() {
  local mode_name="$1"
  local log_name="$2"
  shift 2
  log "Running $mode_name"
  set +e
  python "$KV_DIR/main_kv_cache.py" --model_config_name "$MODEL_CONFIG_NAME" "${COMMON_GENERATE_ARGS[@]}" "$@" "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$OUT_DIR/$log_name"
  local status=$?
  set -e
  printf '{"mode":"%s","exit_code":%d}\n' "$mode_name" "$status" > "$OUT_DIR/${log_name%.log}_status.json"
}

run_mode "mode1 torch_dynamic_cache native" "mode1_torch_dynamic_cache_native.log" --max_length 800 --kv_layout segmented_tiled --batch_size 4 # 51s
run_mode "mode3 compressed_materialize_decode native" "mode3_compressed_materialize_decode_native.log" --compress_kv_cache --kv_adjust_update_steps 1000000000 --max_length 800 --kv_layout segmented_tiled --batch_size 4 # 59s
