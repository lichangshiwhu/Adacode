#!/usr/bin/env sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
KV_DIR="${KV_DIR:-$ROOT_DIR/kv_compression}"
export PYTHONPATH="$ROOT_DIR:$KV_DIR:${PYTHONPATH:-}"

MODEL_CONFIG_NAME="${MODEL_CONFIG_NAME:-deepseek_v2_lite_config}"
KV_CACHE_IMPL="${KV_CACHE_IMPL:-mla_float11}"
MLA_CACHE_MODE="${MLA_CACHE_MODE:-auto}"
KV_LAYOUT="${KV_LAYOUT:-segmented_tiled}"
# 128 for long inputs
# 16 for short inputs
BLOCK_SIZE="${BLOCK_SIZE:-128}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_SEQUENCE="${NUM_SEQUENCE:-64}"
MAX_LENGTH="${MAX_LENGTH:-20480}"
# 129 for long 
# 1025 for short
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1025}"
OUTPUT_PATH="${OUTPUT_PATH:-$ROOT_DIR/kv_short_BLOCK_SIZE16_inputs_DeepSeek-V2-Lite.json}"
FLOAT11_NEED_ADJUST=1
MLA_COMPRESS_ROPE=1
FLOAT11_UPDATE_STEPS="${FLOAT11_UPDATE_STEPS:-128}"

set -- \
  python "$KV_DIR/main_lossless_comp.py" \
  --model_config_name "$MODEL_CONFIG_NAME" \
  --kv_cache_impl "$KV_CACHE_IMPL" \
  --mla_cache_mode "$MLA_CACHE_MODE" \
  --kv_layout "$KV_LAYOUT" \
  --block_size "$BLOCK_SIZE" \
  --batch_size "$BATCH_SIZE" \
  --num_seqence "$NUM_SEQUENCE" \
  --max_length "$MAX_LENGTH" \
  --min_new_tokens "$MAX_NEW_TOKENS" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --output_path "$OUTPUT_PATH" \
  "$@"

if [ "${MLA_COMPRESS_ROPE:-0}" != "1" ]; then
  set -- "$@" --mla_disable_rope_compression
fi

if [ "${USE_SYNTHETIC_DATA:-0}" = "1" ]; then
  set -- "$@" --use_synthetic_data --synthetic_seq_length "${SYNTHETIC_SEQ_LENGTH:-2048}"
fi

if [ "${FLOAT11_NEED_ADJUST:-0}" = "1" ]; then
  set -- "$@" --float11_need_adjust --float11_update_steps "${FLOAT11_UPDATE_STEPS:-64}"
fi

echo "Running: $*"
"$@"
