#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
KV_DIR="${KV_DIR:-$ROOT_DIR/kv_compression}"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"


for model_config_name in qwen3_4B_config mistral_7B_config qwen1p5_moe_config huihui_moe_config
do
python "$KV_DIR/main_kv_long.py" --compress_kv_cache --kv_adjuster_enabled --kv_adjust_update_steps 128 \
    --model_config_name ${model_config_name} --max_new_tokens ${MAX_NEW_TOKENS} --min_new_tokens ${MAX_NEW_TOKENS} \
    --kv_layout segmented_tiled --batch_size 1 --kv_experiment_name kv_long
done

