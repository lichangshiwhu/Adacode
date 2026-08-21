#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
KV_DIR="${KV_DIR:-$ROOT_DIR/kv_compression}"
export PYTHONPATH="$ROOT_DIR:$KV_DIR:${PYTHONPATH:-}"

MODEL_CONFIG_NAME="${1:-qwen3_4B_config mistral_7B_config qwen1p5_moe_config huihui_moe_config}"
shift || true
EXTRA_ARGS=("$@")
# 512 1024 2048 4096 8192 16384
INPUT_LENGTHS="${INPUT_LENGTHS:-20480}"
CACHE_IMPLS="${CACHE_IMPLS:-dynamic float11 cpu_offload nvcomp_ans}"
BLOCK_SIZE="${BLOCK_SIZE:-128}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_SEQUENCE="${NUM_SEQUENCE:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
MIN_NEW_TOKENS="${MIN_NEW_TOKENS:-$MAX_NEW_TOKENS}"
MAX_LENGTH="${MAX_LENGTH:-20480}"
KV_LAYOUT="${KV_LAYOUT:-segmented_tiled}"
FLOAT11_NEED_ADJUST="${FLOAT11_NEED_ADJUST:-1}"
FLOAT11_UPDATE_STEPS="${FLOAT11_UPDATE_STEPS:-128}"
CPU_OFFLOAD_RATIO="${CPU_OFFLOAD_RATIO:-0.1}"
CPU_OFFLOAD_MIN_TOKENS="${CPU_OFFLOAD_MIN_TOKENS:-1}"
PIN_MEMORY_DISABLED="${PIN_MEMORY_DISABLED:-0}"
NVCOMP_UNCOMP_CHUNK_SIZE="${NVCOMP_UNCOMP_CHUNK_SIZE:-65536}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/kv_longbench_sweep}"

mkdir -p "$OUT_DIR"

log() {
  printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"
}

run_case() {
  local cache_impl="$1"
  local modelconfig="$2"
  # local input_length="$2"
  local output_json="$OUT_DIR/${modelconfig}_${cache_impl}.json"
  local output_log="$OUT_DIR/${modelconfig}_${cache_impl}.log"
  local status_json="$OUT_DIR/${modelconfig}_${cache_impl}_status.json"

  local cmd=(
    python "$KV_DIR/main_lossless_comp.py"
    --model_config_name "$modelconfig"
    --kv_cache_impl "$cache_impl"
    --block_size "$BLOCK_SIZE"
    --batch_size "$BATCH_SIZE"
    --num_seqence "$NUM_SEQUENCE"
    --max_length "$MAX_LENGTH"
    --min_new_tokens "$MIN_NEW_TOKENS"
    --max_new_tokens "$MAX_NEW_TOKENS"
    --output_path "$output_json"
  )
    # --synthetic_seq_length "$input_length"
    # --use_synthetic_data

  case "$cache_impl" in
    dynamic)
      ;;
    float11)
      cmd+=(--float11_layout "$KV_LAYOUT")
      if [ "$FLOAT11_NEED_ADJUST" = "1" ]; then
        cmd+=(--float11_need_adjust --float11_update_steps "$FLOAT11_UPDATE_STEPS")
      fi
      ;;
    nvcomp_ans)
      cmd+=(--nvcomp_uncomp_chunk_size "$NVCOMP_UNCOMP_CHUNK_SIZE")
      ;;
    cpu_offload)
      cmd+=(--cpu_offload_ratio "$CPU_OFFLOAD_RATIO" --cpu_offload_min_tokens "$CPU_OFFLOAD_MIN_TOKENS")
      if [ "$PIN_MEMORY_DISABLED" = "1" ]; then
        cmd+=(--disable_pin_memory)
      fi
      ;;
    *)
      printf 'Unsupported cache_impl: %s\n' "$cache_impl" >&2
      return 1
      ;;
  esac

  if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
    cmd+=("${EXTRA_ARGS[@]}")
  fi

  log "Running cache_impl=$cache_impl modelname=$modelconfig"
  set +e
  "${cmd[@]}" 2>&1 | tee "$output_log"
  local status=$?
  set -e
  printf '{"model_config_name":"%s","cache_impl":"%s","exit_code":%d}\n' \
    "$modelconfig" "$cache_impl" "$status" > "$status_json"
}

# for input_length in $INPUT_LENGTHS

for modelconfig in $MODEL_CONFIG_NAME
do
  for cache_impl in $CACHE_IMPLS
  do
    run_case "$cache_impl" "$modelconfig"
  done
done
