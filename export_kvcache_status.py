import os
import gc
import json
import argparse
from typing import Any

# import tpdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, StaticCache, Cache
from datasets import load_dataset

import triton
# from model_scripts import yicoderconfig, qwen2config, qwen1config
from compress_bitmap_merge import get_exp_kernel_constexpr
from utils import DatasetsLoader

def _build_dtype_configs() -> dict[str, dict[str, Any]]:
    configs = {
        'bf16': {
            'torch_dtype': torch.bfloat16,
            'storage_bits': 16,
            'exp_bits': 8,
            'mantissa_bits': 7,
            'view_dtype': torch.uint16,
        },
        'fp16': {
            'torch_dtype': torch.float16,
            'storage_bits': 16,
            'exp_bits': 5,
            'mantissa_bits': 10,
            'view_dtype': torch.uint16,
        },
    }
    if hasattr(torch, 'float8_e4m3fn'):
        configs['fp8_e4m3fn'] = {
            'torch_dtype': torch.float8_e4m3fn,
            'storage_bits': 8,
            'exp_bits': 4,
            'mantissa_bits': 3,
            'view_dtype': torch.uint8,
        }
    if hasattr(torch, 'float8_e5m2'):
        configs['fp8_e5m2'] = {
            'torch_dtype': torch.float8_e5m2,
            'storage_bits': 8,
            'exp_bits': 5,
            'mantissa_bits': 2,
            'view_dtype': torch.uint8,
        }
    return configs


DTYPE_CONFIGS = _build_dtype_configs()
DTYPE_ALIASES = {
    'bfloat16': 'bf16',
    'float16': 'fp16',
    'half': 'fp16',
    'e4m3': 'fp8_e4m3fn',
    'fp8': 'fp8_e4m3fn',
    'float8_e4m3fn': 'fp8_e4m3fn',
    'e5m2': 'fp8_e5m2',
    'float8_e5m2': 'fp8_e5m2',
}


def _resolve_text_config(config: Any) -> Any:
    if hasattr(config, 'get_text_config'):
        try:
            return config.get_text_config(decoder=True)
        except TypeError:
            return config.get_text_config()
    return config


def _is_mla_model_config(config: Any) -> bool:
    text_config = _resolve_text_config(config)
    model_type = str(getattr(text_config, 'model_type', '')).lower()
    architectures = [str(item).lower() for item in getattr(text_config, 'architectures', []) or []]
    if getattr(text_config, 'use_mla', False):
        return True
    if getattr(text_config, 'kv_lora_rank', None) is not None:
        return True
    if getattr(text_config, 'q_lora_rank', None) is not None:
        return True
    if 'deepseek_v2' in model_type or 'deepseek_v3' in model_type:
        return True
    return any('deepseekv2' in item or 'deepseekv3' in item for item in architectures)


def _normalize_dtype_name(dtype: Any) -> str:
    if isinstance(dtype, torch.dtype):
        for name, config in DTYPE_CONFIGS.items():
            if config['torch_dtype'] == dtype:
                return name
        raise ValueError(f'Unsupported torch dtype for KV stats: {dtype}')
    dtype_name = str(dtype).lower()
    dtype_name = DTYPE_ALIASES.get(dtype_name, dtype_name)
    if dtype_name not in DTYPE_CONFIGS:
        raise ValueError(f'Unsupported dtype for KV stats: {dtype}')
    return dtype_name


def _get_dtype_config(dtype: Any) -> dict[str, Any]:
    return DTYPE_CONFIGS[_normalize_dtype_name(dtype)]


def _view_raw_bits(t: torch.Tensor, dtype: Any | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
    config = _get_dtype_config(t.dtype if dtype is None else dtype)
    if not t.is_contiguous():
        t = t.contiguous()
    flatten_t = t.view(-1)
    raw_bits = flatten_t.view(config['view_dtype']).to(torch.int32)
    return raw_bits, config


def _quantized_raw_bits(t: torch.Tensor, dtype: Any) -> tuple[torch.Tensor, dict[str, Any]]:
    dtype_name = _normalize_dtype_name(dtype)
    config = _get_dtype_config(dtype_name)
    if dtype_name in {'fp8_e4m3fn', 'fp8_e5m2'}:
        quantized = t.detach().to('cpu', dtype=torch.float32).to(config['torch_dtype'])
    else:
        quantized = t.to(config['torch_dtype'])
    return _view_raw_bits(quantized, dtype_name)


def get_exp_freq_list(t, dtype='bf16'):
    dtype_name = _normalize_dtype_name(dtype)
    if t.numel() == 0:
        return [0] * (256 if dtype_name == 'bf16' else (1 << _get_dtype_config(dtype_name)['exp_bits']))
    if dtype_name == 'bf16':
        BLOCK_SIZE = 1024
        flatten_t = t.contiguous().view(-1)
        total_size = flatten_t.numel()
        device = flatten_t.device
        extracted_values = torch.zeros(total_size, dtype=torch.uint8, device=device)
        grid1 = lambda meta: (triton.cdiv(total_size, meta['BLOCK_SIZE']), )
        get_exp_kernel_constexpr[grid1](
            flatten_t.view(torch.uint16),
            extracted_values,
            total_size,
            BLOCK_SIZE=BLOCK_SIZE
        )
        t_freq_list = torch.bincount(extracted_values.to(torch.int64), minlength=2**8)
        return t_freq_list.cpu().tolist()

    raw_bits, config = _quantized_raw_bits(t, dtype_name)
    exponent_bits = (raw_bits >> config['mantissa_bits']) & ((1 << config['exp_bits']) - 1)
    t_freq_list = torch.bincount(exponent_bits.to(torch.int64), minlength=1 << config['exp_bits'])
    return t_freq_list.cpu().tolist()


def get_exp_freq_list_fp16(t, count_len=2**5):
    return get_exp_freq_list(t, dtype='fp16')


def get_exp_freq_list_E4M3(t, count_len=2**4):
    return get_exp_freq_list(t, dtype='fp8_e4m3fn')


def get_sign_freq_list(t, dtype='bf16'):
    if t.numel() == 0:
        return [0, 0]
    raw_bits, config = _quantized_raw_bits(t, dtype)
    sign_bits = (raw_bits >> (config['storage_bits'] - 1)) & 0x1
    sign_freq_list = torch.bincount(sign_bits.to(torch.int64), minlength=2)
    return sign_freq_list.cpu().tolist()


def get_mantissa_freq_list(t, dtype='bf16'):
    if t.numel() == 0:
        dtype_name = _normalize_dtype_name(dtype)
        return [0] * (1 << _get_dtype_config(dtype_name)['mantissa_bits'])
    raw_bits, config = _quantized_raw_bits(t, dtype)
    mantissa_mask = (1 << config['mantissa_bits']) - 1
    mantissa_bits = raw_bits & mantissa_mask
    t_freq_list = torch.bincount(mantissa_bits.to(torch.int64), minlength=1 << config['mantissa_bits'])
    return t_freq_list.cpu().tolist()


def get_all_bits_freq_list(t, dtype='bf16'):
    dtype_name = _normalize_dtype_name(dtype)
    config = _get_dtype_config(dtype_name)
    if t.numel() == 0:
        return [0] * (1 << config['storage_bits'])
    raw_bits, config = _quantized_raw_bits(t, dtype_name)
    t_freq_list = torch.bincount(raw_bits.to(torch.int64), minlength=1 << config['storage_bits'])
    return t_freq_list.cpu().tolist()


def detect_cache_backend(model, kv_cache=None) -> str:
    if _is_mla_model_config(getattr(model, 'config', None)):
        return 'mla'
    if hasattr(kv_cache, 'layers'):
        for layer in kv_cache.layers:
            if any(hasattr(layer, attr) for attr in ('compressed_kv', 'latent_cache', 'latent_states', 'kv_latents', 'k_pe', 'rope_cache')):
                return 'mla'
    return 'traditional_kv'


def _normalize_tensor_name(name: str) -> str:
    return name.lower().replace(' ', '_')


def _iter_tensor_attrs(layer) -> list[tuple[str, torch.Tensor]]:
    tensor_attrs: list[tuple[str, torch.Tensor]] = []
    for attr in dir(layer):
        if attr.startswith('_'):
            continue
        value = getattr(layer, attr, None)
        if isinstance(value, torch.Tensor) and value.ndim >= 2:
            tensor_attrs.append((attr, value))
    return tensor_attrs


def _slice_prefill_decode(tensor: torch.Tensor, prefill_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    if tensor.ndim < 2:
        raise ValueError(f'Cache tensor must have a sequence dimension, got shape={tuple(tensor.shape)}')
    seq_len = int(tensor.shape[-2])
    prefix_len = min(int(prefill_len), seq_len)
    return (
        tensor.narrow(-2, 0, prefix_len),
        tensor.narrow(-2, prefix_len, seq_len - prefix_len),
    )


def _get_layer_tensor_groups(kv_cache, layer_idx: int, backend: str) -> dict[str, torch.Tensor]:
    tensor_groups: dict[str, torch.Tensor] = {}
    layer = kv_cache.layers[layer_idx] if hasattr(kv_cache, 'layers') else None

    def _maybe_add(name: str, value: Any):
        if isinstance(value, torch.Tensor):
            tensor_groups[_normalize_tensor_name(name)] = value

    if layer is not None:
        _maybe_add('keys', getattr(layer, 'keys', None))
        _maybe_add('values', getattr(layer, 'values', None))
        _maybe_add('key_cache', getattr(layer, 'key_cache', None))
        _maybe_add('value_cache', getattr(layer, 'value_cache', None))
        if backend == 'mla':
            for attr, value in _iter_tensor_attrs(layer):
                normalized_attr = _normalize_tensor_name(attr)
                if normalized_attr not in tensor_groups:
                    _maybe_add(attr, value)

    if not tensor_groups:
        key_cache = getattr(kv_cache, 'key_cache', None)
        value_cache = getattr(kv_cache, 'value_cache', None)
        if isinstance(key_cache, (list, tuple)) and layer_idx < len(key_cache):
            _maybe_add('keys', key_cache[layer_idx])
        if isinstance(value_cache, (list, tuple)) and layer_idx < len(value_cache):
            _maybe_add('values', value_cache[layer_idx])

    if 'keys' not in tensor_groups and 'key_cache' in tensor_groups:
        tensor_groups['keys'] = tensor_groups.pop('key_cache')
    if 'values' not in tensor_groups and 'value_cache' in tensor_groups:
        tensor_groups['values'] = tensor_groups.pop('value_cache')

    if not tensor_groups:
        available = [] if layer is None else [
            attr for attr in dir(layer)
            if not attr.startswith('_') and isinstance(getattr(layer, attr, None), torch.Tensor)
        ]
        raise ValueError(
            f'Unable to locate cache tensors for layer {layer_idx} under backend={backend}. '
            f'Available tensor attrs: {available}'
        )
    return tensor_groups


def extract_exp_freq(kv_cache, freq_type, prefill_len=None, dtype='bf16', backend='traditional_kv'):
    if freq_type == 'sign':
        get_freq_list = lambda tensor: get_sign_freq_list(tensor, dtype=dtype)
    elif freq_type == 'exp':
        get_freq_list = lambda tensor: get_exp_freq_list(tensor, dtype=dtype)
    elif freq_type == 'mantissa':
        get_freq_list = lambda tensor: get_mantissa_freq_list(tensor, dtype=dtype)
    elif freq_type == 'all':
        get_freq_list = lambda tensor: get_all_bits_freq_list(tensor, dtype=dtype)
    else:
        raise ValueError(f'Unsupported freq_type: {freq_type}')

    exp_freq = {}
    for layer_idx in range(len(kv_cache.layers)):
        tensor_groups = _get_layer_tensor_groups(kv_cache, layer_idx, backend)
        kv_pair = {}
        if prefill_len is not None:
            for tensor_name, tensor in tensor_groups.items():
                prefill_tensor, decode_tensor = _slice_prefill_decode(tensor, prefill_len)
                kv_pair[f'prefill_{tensor_name}'] = get_freq_list(prefill_tensor)
                kv_pair[f'decode_{tensor_name}'] = get_freq_list(decode_tensor)
        else:
            for tensor_name, tensor in tensor_groups.items():
                kv_pair[tensor_name] = get_freq_list(tensor)
        exp_freq[layer_idx] = kv_pair
    return exp_freq

def merge_exp_freq(freq1, freq2):
    exp_freq = {}
    for k1 in freq1.keys():
        kv_pair = {}
        for k2 in freq1[k1].keys():
            list1 = freq1[k1][k2]
            list2 = freq2[k1][k2]
            kv_pair[k2] = [l1+l2 for l1, l2 in zip(list1, list2)]
        exp_freq[k1] = kv_pair
    return exp_freq


class InlinePromptDataset:
    def __init__(self, prompt: str):
        self.prompts = [prompt]

    def __getitem__(self, indices):
        if isinstance(indices, slice):
            return self.prompts[indices]
        return self.prompts[indices]

    def __len__(self):
        return len(self.prompts)

    def __repr__(self):
        preview = self.prompts[0].replace('\n', '\\n')
        if len(preview) > 80:
            preview = preview[:77] + '...'
        return f'InlinePromptDataset(prompt="{preview}")'


def _is_inline_prompt_dataset(dataset_name: str) -> bool:
    return dataset_name.startswith('prompt:') or dataset_name.startswith('prompt=')


def _parse_inline_prompt(dataset_name: str) -> str:
    if dataset_name.startswith('prompt:'):
        prompt = dataset_name[len('prompt:'):]
    elif dataset_name.startswith('prompt='):
        prompt = dataset_name[len('prompt='):]
    else:
        raise ValueError(f'Not an inline prompt dataset: {dataset_name}')
    prompt = prompt.strip()
    if not prompt:
        raise ValueError('Inline prompt dataset is empty. Use --datasets "prompt:your prompt text".')
    return prompt

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Export KV cache status statistics.')
    parser.add_argument('--dtype', default='bf16', help='Model dtype: bf16, fp16, fp8_e4m3fn, fp8_e5m2, fp8, e4m3, e5m2.')
    parser.add_argument('--stats-dtype', default=None, help='Statistics dtype; defaults to --dtype. Use fp8_* here for numerical simulation.')
    parser.add_argument('--cache-backend', default='auto', choices=['auto', 'traditional_kv', 'mla'], help='Cache backend selection; auto-detect from model config.')
    parser.add_argument('--models', nargs='*', default=[
        '/opt/data/private/Qwen/Qwen3-4B',
        '/opt/data/private/mistralai/Mistral-7B-Instruct-v0.3',
        '/opt/data/private/Qwen/Qwen1.5-MoE-A2.7B-Chat',
        '/opt/data/private/huihui-ai/Huihui-MoE-1.2B-A0.6B',
    ], help='Model IDs to export.')
    parser.add_argument('--datasets', nargs='*', default=['riddlebench', 'simplemath', 'c4', '2wikimqa'], help='Datasets to sample. Use "prompt:TEXT" or "prompt=TEXT" to run a direct inline test prompt.')
    parser.add_argument('--freq-types', nargs='*', default=['sign', 'exp', 'mantissa'], help='Frequency types to export: sign, exp, mantissa, all. all exports full raw bit-pattern histograms.')
    parser.add_argument('--output-prefix-name', default='kv_test', help='Output file prefix.')
    parser.add_argument('--max-new-tokens', type=int, default=1024, help='Max new tokens for generation.')
    parser.add_argument('--batch-size', type=int, default=32, help='Total samples per dataset.')
    parser.add_argument('--mini-batch-size', type=int, default=4, help='Samples per generation batch.')
    return parser


def main():
    gc.collect()
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    args = build_arg_parser().parse_args()
    model_dtype_name = _normalize_dtype_name(args.dtype)
    stats_dtype_name = _normalize_dtype_name(args.stats_dtype or model_dtype_name)
    dtype = DTYPE_CONFIGS[model_dtype_name]['torch_dtype']

    for model_id in args.models:
        for freq_type in args.freq_types:
            output_path = f"./model_configs/{args.output_prefix_name}_{model_id.split('/')[-1]}_{freq_type}.json"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            # , trust_remote_code=True
            model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map='auto')
            model.eval()
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            tokenizer.pad_token = tokenizer.eos_token

            dataset_kv_freq = {}
            inline_prompt_total = sum(1 for dn in args.datasets if _is_inline_prompt_dataset(dn))
            inline_prompt_seen = 0
            for dn in args.datasets:
                gc.collect()
                if _is_inline_prompt_dataset(dn):
                    inline_prompt_seen += 1
                    dataset_key = 'prompt' if inline_prompt_total == 1 else f'prompt_{inline_prompt_seen}'
                    dataset = InlinePromptDataset(_parse_inline_prompt(dn))
                    total_samples = len(dataset)
                else:
                    dataset_key = dn
                    dataset = DatasetsLoader(dn)
                    total_samples = args.batch_size
                print(dataset)
                for start_idx in range(0, total_samples, args.mini_batch_size):
                    end_idx = min(start_idx + args.mini_batch_size, total_samples)
                    prompts = dataset[start_idx:end_idx]
                    with torch.no_grad():
                        try:
                            prompt_cache = DynamicCache(config=model.config)
                            new_inputs = tokenizer(prompts, padding=True, padding_side='left', return_tensors='pt').to(model.device.type)
                            prefill_len = new_inputs.input_ids.shape[1]
                            _ = model.generate(
                                **new_inputs,
                                past_key_values=prompt_cache,
                                max_new_tokens=args.max_new_tokens,
                            )
                            cache_backend = args.cache_backend
                            if cache_backend == 'auto':
                                cache_backend = detect_cache_backend(model, prompt_cache)
                            print(f'cache_backend={cache_backend}')
                            if dataset_key not in dataset_kv_freq:
                                dataset_kv_freq[dataset_key] = extract_exp_freq(prompt_cache, freq_type, prefill_len=prefill_len, dtype=stats_dtype_name, backend=cache_backend)
                            else:
                                existing_freq = dataset_kv_freq[dataset_key]
                                new_freq = extract_exp_freq(prompt_cache, freq_type, prefill_len=prefill_len, dtype=stats_dtype_name, backend=cache_backend)
                                dataset_kv_freq[dataset_key] = merge_exp_freq(existing_freq, new_freq)
                            print(f'{dataset_kv_freq=}')
                            with open(output_path, 'w', encoding='utf-8') as f:
                                json.dump(dataset_kv_freq, f)
                        except Exception as e:
                            print(f'{type(e).__name__}: {e}')
                            print(f'skip {dataset_key} tasks')
                del prompt_cache
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()
            print(dataset_kv_freq)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(dataset_kv_freq, f)


if __name__ == "__main__":
    main()
