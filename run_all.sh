#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
KV_DIR="${KV_DIR:-$ROOT_DIR/kv_compression}"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1025}"

# Compare pre-computed compression parameters against per-run best parameters.
# for model_config_name in qwen3_4B_config mistral_7B_config qwen1p5_moe_config huihui_moe_config
# do
# python "$KV_DIR/main_kv_cache.py" --compress_kv_cache --kv_adjust_update_steps 0 --model_config_name ${model_config_name} --max_new_tokens 2 --min_new_tokens 2 --kv_layout segmented_tiled --batch_size 1 --kv_experiment_name precomputed
# python "$KV_DIR/main_kv_cache.py" --compress_kv_cache --kv_adjuster_enabled --kv_adjust_update_steps 1 --model_config_name ${model_config_name} --max_new_tokens 2 --min_new_tokens 2 --kv_layout segmented_tiled --batch_size 1 --kv_experiment_name current_best
# done

# 128 256 512 1024 
# kv_adjust_update_steps impacts
# for model_config_name in qwen3_4B_config mistral_7B_config qwen1p5_moe_config huihui_moe_config
# do
#     for adjust_step in 32 64 128 256 512 1024
#         do
#             python "$KV_DIR/main_kv_cache.py" --compress_kv_cache --kv_adjuster_enabled --kv_adjust_update_steps ${adjust_step} --model_config_name ${model_config_name} --max_new_tokens 1024 --kv_layout segmented_tiled --batch_size 1
#         done
# done

for model_config_name in qwen3_4B_config mistral_7B_config qwen1p5_moe_config huihui_moe_config
do
python "$KV_DIR/main_kv_cache.py" --compress_kv_cache --kv_adjuster_enabled --kv_adjust_update_steps 128 \
    --model_config_name ${model_config_name} --max_new_tokens ${MAX_NEW_TOKENS} --min_new_tokens ${MAX_NEW_TOKENS} \
    --kv_layout segmented_tiled --batch_size 1 --kv_experiment_name kv_short
done

