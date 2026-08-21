#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
KV_DIR="${KV_DIR:-$ROOT_DIR/kv_compression}"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

OUT_DIR="${1:-$SCRIPT_DIR/validation_outputs}"
mkdir -p "$OUT_DIR"
export OUT_DIR
export ROOT_DIR

MODEL_CONFIG_NAME="${MODEL_CONFIG_NAME:-qwen2_14B_config}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
MIN_NEW_TOKENS="${MIN_NEW_TOKENS:-$MAX_NEW_TOKENS}"
BLOCK_SIZE="${BLOCK_SIZE:-128}"
HEAD_DIM="${HEAD_DIM:-16}"
NUM_KV_HEADS="${NUM_KV_HEADS:-2}"
NUM_ATTN_HEADS="${NUM_ATTN_HEADS:-2}"
ATOL="${ATOL:-1e-2}"
RTOL="${RTOL:-0}"
DEVICE="${DEVICE:-cuda}"
RUN_E2E="${RUN_E2E:-0}"
RUN_VALIDATOR="${RUN_VALIDATOR:-1}"
SKIP_MISSING_DEPS="${SKIP_MISSING_DEPS:-1}"
KV_VALIDATE_NBITS="${KV_VALIDATE_NBITS:-3}"
KV_VALIDATE_RIGHT="${KV_VALIDATE_RIGHT:-3}"
KV_ADJUSTER_ENABLED="${KV_ADJUSTER_ENABLED:-0}"
KV_ADJUST_UPDATE_STEPS="${KV_ADJUST_UPDATE_STEPS:-1000000000}"

export BLOCK_SIZE
export HEAD_DIM
export NUM_KV_HEADS
export NUM_ATTN_HEADS
export ATOL
export RTOL
export DEVICE
export KV_VALIDATE_NBITS
export KV_VALIDATE_RIGHT
export KV_ADJUSTER_ENABLED
export KV_ADJUST_UPDATE_STEPS

log() {
  printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"
}

log "KV_DIR=$KV_DIR"
log "ROOT_DIR=$ROOT_DIR"
log "Writing outputs to $OUT_DIR"

python - <<'PY' | tee "$OUT_DIR/dependency_check.json"
import importlib
import json
import os
import sys

mods = ['torch', 'transformers', 'datasets', 'triton']
result = {}
missing = []
for name in mods:
    try:
        mod = importlib.import_module(name)
        result[name] = getattr(mod, '__version__', 'imported')
    except Exception as e:
        result[name] = f'IMPORT_ERROR: {type(e).__name__}: {e}'
        missing.append(name)

skip_missing = os.environ.get('SKIP_MISSING_DEPS', '1') == '1'
summary = {
    'modules': result,
    'missing': missing,
    'skip_missing_deps': skip_missing,
    'can_run_main_kv_cache': not any(name in missing for name in ['torch', 'transformers', 'datasets']),
}
print(json.dumps(summary, indent=2, ensure_ascii=False))
if missing and not skip_missing:
    sys.exit(1)
PY

log "Running compileall"
python -m compileall \
  "$KV_DIR/compress_tiled_kv.py" \
  "$KV_DIR/float_cache_utils.py" \
  "$KV_DIR/ops/kv_attention.py" \
  "$KV_DIR/validate_kv_attention.py" \
  "$KV_DIR/main_kv_cache.py" \
  2>&1 | tee "$OUT_DIR/compileall.log"

HAS_TORCH="$(python - <<'PY'
import importlib.util
print('1' if importlib.util.find_spec('torch') else '0')
PY
)"
HAS_TRANSFORMERS="$(python - <<'PY'
import importlib.util
print('1' if importlib.util.find_spec('transformers') else '0')
PY
)"
HAS_DATASETS="$(python - <<'PY'
import importlib.util
print('1' if importlib.util.find_spec('datasets') else '0')
PY
)"

CAN_RUN_MAIN_KV_CACHE=0
if [[ "$HAS_TORCH" == "1" && "$HAS_TRANSFORMERS" == "1" && "$HAS_DATASETS" == "1" ]]; then
  CAN_RUN_MAIN_KV_CACHE=1
fi

printf '{"skipped":true,"reason":"custom triton attention removed from ours"}\n' > "$OUT_DIR/dense_triton_numeric.json"

if [[ "$RUN_VALIDATOR" == "1" ]]; then
  if [[ "$HAS_TORCH" != "1" ]]; then
    printf '{"skipped":true,"reason":"torch missing"}\n' > "$OUT_DIR/validate_materialize.json"
  else
    VALIDATOR_DEVICE="$DEVICE"
    if [[ "$VALIDATOR_DEVICE" == "cuda" && "$(python - <<'PY'
import torch
print('1' if torch.cuda.is_available() else '0')
PY
)" != "1" ]]; then
      VALIDATOR_DEVICE="cpu"
    fi

    log "Preparing validator fixtures"
    VALIDATOR_DEVICE="$VALIDATOR_DEVICE" python - <<'PY'
import os
import sys
import torch

sys.path.insert(0, os.environ['ROOT_DIR'])
from kv_compression.float_cache_utils import float11Layer

out_dir = os.environ['OUT_DIR']
head_dim = int(os.environ['HEAD_DIM'])
num_kv_heads = int(os.environ['NUM_KV_HEADS'])
num_attn_heads = int(os.environ['NUM_ATTN_HEADS'])
seq_len = int(os.environ['BLOCK_SIZE'])
nbits = int(os.environ.get('KV_VALIDATE_NBITS', '3'))
right = int(os.environ.get('KV_VALIDATE_RIGHT', str(nbits)))
need_adjust = os.environ.get('KV_ADJUSTER_ENABLED', '0') == '1'
device = torch.device(os.environ.get('VALIDATOR_DEVICE', os.environ.get('DEVICE', 'cuda')))

assert num_attn_heads == num_kv_heads, 'Default fixture expects num_attn_heads == num_kv_heads'

torch.manual_seed(2026)
q = torch.randn(1, num_attn_heads, 1, head_dim, dtype=torch.float32, device=device).to(torch.bfloat16)
k = torch.randn(1, num_kv_heads, seq_len, head_dim, dtype=torch.float32, device=device).to(torch.bfloat16)
v = torch.randn(1, num_kv_heads, seq_len, head_dim, dtype=torch.float32, device=device).to(torch.bfloat16)

layer = float11Layer(
    keys_right=right,
    values_right=right,
    keys_nbits=nbits,
    values_nbits=nbits,
    block_size=seq_len,
    layout='legacy',
)
layer.update(k, v, need_adjust=need_adjust)
cache_view = layer.get_cache_view()
cache_view = {k: v for k, v in cache_view.items() if k != '_float11_layer'}

torch.save(q, os.path.join(out_dir, 'query_states.pt'))
torch.save(cache_view, os.path.join(out_dir, 'cache_view.pt'))
torch.save(k, os.path.join(out_dir, 'key_states.pt'))
torch.save(v, os.path.join(out_dir, 'value_states.pt'))
PY

    log "Running materialize validator"
    python "$KV_DIR/validate_kv_attention.py" \
      --query_states "$OUT_DIR/query_states.pt" \
      --cache_view "$OUT_DIR/cache_view.pt" \
      --mode materialize \
      --atol "$ATOL" \
      --rtol "$RTOL" \
      --device "$VALIDATOR_DEVICE" \
      --json | tee "$OUT_DIR/validate_materialize.json"
  fi
fi

if [[ "$RUN_E2E" == "1" ]]; then
  if [[ "$CAN_RUN_MAIN_KV_CACHE" != "1" ]]; then
    printf '{"status":"skipped","reason":"missing torch/transformers/datasets"}\n' > "$OUT_DIR/run_kv_modes_status.json"
    printf '{"status":"skipped","reason":"missing torch/transformers/datasets"}\n' > "$OUT_DIR/e2e_status.json"
  else
    log "Running end-to-end main_kv_cache modes via run_kv_modes.sh"
    set +e
    bash "$KV_DIR/run_kv_modes.sh" "$MODEL_CONFIG_NAME" \
      --kv_adjust_update_steps "$KV_ADJUST_UPDATE_STEPS" \
      $( [[ "$KV_ADJUSTER_ENABLED" == "1" ]] && printf '%s' '--kv_adjuster_enabled' ) \
      2>&1 | tee "$OUT_DIR/run_kv_modes.log"
    run_modes_status=$?

    python "$KV_DIR/main_kv_cache.py" \
      --model_config_name "$MODEL_CONFIG_NAME" \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      --min_new_tokens "$MIN_NEW_TOKENS" \
      --kv_adjust_update_steps "$KV_ADJUST_UPDATE_STEPS" \
      $( [[ "$KV_ADJUSTER_ENABLED" == "1" ]] && printf '%s' '--kv_adjuster_enabled' ) \
      2>&1 | tee "$OUT_DIR/e2e_torch_dynamic_cache_native.log"
    s1=$?

    python "$KV_DIR/main_kv_cache.py" \
      --model_config_name "$MODEL_CONFIG_NAME" \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      --min_new_tokens "$MIN_NEW_TOKENS" \
      --compress_kv_cache \
      --kv_adjust_update_steps "$KV_ADJUST_UPDATE_STEPS" \
      $( [[ "$KV_ADJUSTER_ENABLED" == "1" ]] && printf '%s' '--kv_adjuster_enabled' ) \
      2>&1 | tee "$OUT_DIR/e2e_compressed_materialize_decode_native.log"
    s2=$?
    set -e

    printf '{"status":"%s","exit_code":%d}\n' "$( [[ $run_modes_status -eq 0 ]] && printf ok || printf failed )" "$run_modes_status" > "$OUT_DIR/run_kv_modes_status.json"
    python - <<PY > "$OUT_DIR/e2e_status.json"
import json
print(json.dumps({
  'torch_dynamic_cache_native': $s1,
  'compressed_materialize_decode_native': $s2,
}, indent=2))
PY
  fi
fi

log "Validation script finished"
