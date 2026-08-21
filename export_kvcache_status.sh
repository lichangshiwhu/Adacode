set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
KV_DIR="${KV_DIR:-$ROOT_DIR/kv_compression}"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"


# python $KV_DIR/export_kvcache_status.py --dtype fp16 --stats-dtype fp16 --cache-backend traditional_kv \
#     --models /opt/data/private/Qwen/Qwen3-4B /opt/data/private/mistralai/Mistral-7B-Instruct-v0.3 /opt/data/private/Qwen/Qwen1.5-MoE-A2.7B-Chat /opt/data/private/huihui-ai/Huihui-MoE-1.2B-A0.6B \
#     --datasets c4 --output-prefix-name kv_fp16 


# python $KV_DIR/export_kvcache_status.py --dtype fp16 --stats-dtype e4m3 --cache-backend traditional_kv \
#     --models /opt/data/private/Qwen/Qwen3-4B /opt/data/private/mistralai/Mistral-7B-Instruct-v0.3 /opt/data/private/Qwen/Qwen1.5-MoE-A2.7B-Chat /opt/data/private/huihui-ai/Huihui-MoE-1.2B-A0.6B \
#     --datasets c4 --output-prefix-name kv_e4m3 


# python $KV_DIR/export_kvcache_status.py --dtype fp16 --stats-dtype e5m2 --cache-backend traditional_kv \
#     --models /opt/data/private/Qwen/Qwen3-4B /opt/data/private/mistralai/Mistral-7B-Instruct-v0.3 /opt/data/private/Qwen/Qwen1.5-MoE-A2.7B-Chat /opt/data/private/huihui-ai/Huihui-MoE-1.2B-A0.6B \
#     --datasets c4 --output-prefix-name kv_e5m2


# python $KV_DIR/export_kvcache_status.py --dtype bf16 --stats-dtype bf16 --cache-backend traditional_kv \
#     --models /opt/data/private/Qwen/Qwen3-4B /opt/data/private/mistralai/Mistral-7B-Instruct-v0.3 /opt/data/private/Qwen/Qwen1.5-MoE-A2.7B-Chat /opt/data/private/huihui-ai/Huihui-MoE-1.2B-A0.6B \
#     --datasets 2wikimqa gov_report multi_news musique c4 \
#     --output-prefix-name kv_bf16_longbench \
#     --max-new-tokens 128
#     # --datasets simplemath mbpp aime25 livecodebench \

# /opt/data/private/deepseek-ai/deepseek-ai/DeepSeek-V2-Lite
python $KV_DIR/export_kvcache_status.py --dtype bf16 --stats-dtype bf16 --cache-backend mla \
    --models /opt/data/private/deepseek-ai/deepseek-ai/DeepSeek-V2-Lite \
    --datasets c4 \
    --output-prefix-name kv_test \
    --max-new-tokens 1024
