import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, GenerationConfig

from kv_compression.cpu_offload_cache_utils import CPUOffloadCache
from kv_compression.float_cache_utils import Float11Cache
from kv_compression.mla_float_cache_utils import MLAFloat11Cache
from kv_compression.nvcomp_cache_utils import NvcompANSCache
from utils import DatasetsLoader, reset_peak_memory
from manager import convert_model
from model_configs import model_config_map


LONGBENCH_DATASETS = ['2wikimqa', 'gov_report', 'multi_news', 'musique']
SHORT_DATASETS = ['riddlebench', 'simplemath', 'aime25', 'aime24', 'livecodebench', 'mbpp']
DATASET_GROUPS = {
    'longbench': LONGBENCH_DATASETS,
    'short': SHORT_DATASETS,
    'short_datasets': SHORT_DATASETS,
    'synthetic': ['synthetic'],
}


def _sanitize_for_json(value):
    if isinstance(value, dict):
        return {k: _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_for_json(v) for v in value]
    return value


def _extract_generated_output_head(tokenizer, sequences: torch.Tensor, prompt_width: int, preview_tokens: int = 32):
    generated_sequences = sequences[:, prompt_width:]
    previews = []
    pad_token_id = tokenizer.pad_token_id
    for row in generated_sequences:
        row_cpu = row.detach().to('cpu')
        if pad_token_id is not None:
            row_cpu = row_cpu[row_cpu != pad_token_id]
        previews.append([int(token_id) for token_id in row_cpu[:preview_tokens].tolist()])
    return previews


def _model_cuda_devices(model) -> list[str]:
    device_map = getattr(model, 'hf_device_map', None)
    if not isinstance(device_map, dict):
        return []
    devices = set()
    for placement in device_map.values():
        if isinstance(placement, int):
            devices.add(f'cuda:{placement}')
        elif isinstance(placement, str) and placement.startswith('cuda'):
            devices.add(placement)
    return sorted(devices)


def _model_has_non_cuda_placements(model) -> bool:
    device_map = getattr(model, 'hf_device_map', None)
    if not isinstance(device_map, dict):
        return False
    return any(isinstance(p, str) and p in {'cpu', 'disk'} for p in device_map.values())


def _visible_cuda_max_memory(reserve_gib: float = 1.5) -> Optional[dict[Any, str]]:
    if not torch.cuda.is_available():
        return None
    max_memory: dict[Any, str] = {}
    for device_idx in range(torch.cuda.device_count()):
        total_gib = torch.cuda.get_device_properties(device_idx).total_memory / (1024 ** 3)
        usable_gib = max(1, int(math.floor(total_gib - reserve_gib)))
        max_memory[device_idx] = f'{usable_gib}GiB'
    return max_memory


def _model_primary_device(model) -> torch.device:
    model_cuda_devices = _model_cuda_devices(model)
    if model_cuda_devices:
        return torch.device(model_cuda_devices[0])
    return model.device


def _resolve_kv_cache_impl(args) -> str:
    if args.kv_cache_impl != 'auto':
        return args.kv_cache_impl
    if args.enable_cpu_offload and args.compress_kv_cache:
        raise ValueError('Use --kv_cache_impl to choose one cache implementation; --enable_cpu_offload and --compress_kv_cache are both set.')
    if args.enable_cpu_offload:
        return 'cpu_offload'
    if args.compress_kv_cache:
        return 'float11'
    return 'dynamic'


def _build_kv_cache(args, model, model_config, kv_cache_impl: str):
    if kv_cache_impl == 'dynamic':
        return DynamicCache(config=model.config)
    if kv_cache_impl == 'cpu_offload':
        gpu_keep_ratio = max(1e-6, min(1.0, 1.0 - float(args.cpu_offload_ratio)))
        return CPUOffloadCache(
            config=model.config,
            gpu_keep_ratio=gpu_keep_ratio,
            pin_memory=not bool(args.disable_pin_memory),
            min_offload_tokens=args.cpu_offload_min_tokens,
        )
    if kv_cache_impl == 'float11':
        return Float11Cache(
            config=model.config,
            right=model_config.rights,
            exp_bits=model_config.nbits,
            need_adjust=bool(args.float11_need_adjust),
            update_steps=int(args.float11_update_steps),
            residual_length=0,
            block_size=int(args.block_size),
            layout=args.float11_layout,
            seq_tile_size=args.float11_seq_tile_size,
        )
    if kv_cache_impl == 'mla_float11':
        return MLAFloat11Cache(
            config=model.config,
            right=model_config.rights,
            exp_bits=model_config.nbits,
            latent_right=int(args.mla_latent_right),
            rope_right=int(args.mla_rope_right),
            latent_nbits=int(args.mla_latent_nbits),
            rope_nbits=int(args.mla_rope_nbits),
            need_adjust=bool(args.float11_need_adjust),
            update_steps=int(args.float11_update_steps),
            residual_length=0,
            block_size=int(args.block_size),
            layout=args.float11_layout,
            seq_tile_size=args.float11_seq_tile_size,
            mla_cache_mode=args.mla_cache_mode,
            validate_shapes=not bool(args.disable_mla_shape_check),
            compress_rope=not bool(args.mla_disable_rope_compression),
        )
    if kv_cache_impl == 'nvcomp_ans':
        return NvcompANSCache(
            config=model.config,
            block_size=int(args.block_size),
            uncomp_chunk_size=int(args.nvcomp_uncomp_chunk_size),
        )
    raise ValueError(f'Unsupported kv_cache_impl: {kv_cache_impl}')


def _synthetic_vocab_size(model, tokenizer) -> int:
    vocab_size = getattr(model.config, 'vocab_size', None)
    if vocab_size is None:
        vocab_size = getattr(tokenizer, 'vocab_size', None)
    if vocab_size is None:
        raise RuntimeError('Cannot infer vocab size for synthetic input_ids.')
    return int(vocab_size)


def _make_synthetic_inputs(args, model, tokenizer, batch_size: int, start_idx: int) -> dict[str, torch.Tensor]:
    seq_length = max(1, int(args.synthetic_seq_length))
    vocab_size = _synthetic_vocab_size(model, tokenizer)
    device = _model_primary_device(model)
    if args.synthetic_token_id is not None:
        token_id = int(args.synthetic_token_id)
        if token_id < 0 or token_id >= vocab_size:
            raise ValueError(f'--synthetic_token_id must be in [0, {vocab_size}), got {token_id}.')
        input_ids = torch.full((batch_size, seq_length), token_id, dtype=torch.long, device=device)
    else:
        low = 3 if vocab_size > 4 else 0
        generator = torch.Generator(device='cpu')
        generator.manual_seed(int(args.synthetic_seed) + int(start_idx))
        input_ids = torch.randint(
            low=low,
            high=vocab_size,
            size=(batch_size, seq_length),
            dtype=torch.long,
            generator=generator,
        ).to(device)
        for special_id in (tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.bos_token_id):
            if special_id is not None and 0 <= int(special_id) < vocab_size:
                input_ids.masked_fill_(input_ids == int(special_id), low)
    attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=device)
    return {'input_ids': input_ids, 'attention_mask': attention_mask}


def _resolve_dataset_names(args) -> list[str]:
    if args.datasets:
        requested = args.datasets
    elif args.use_synthetic_data:
        requested = ['synthetic']
    else:
        requested = ['longbench']

    known_datasets = set(LONGBENCH_DATASETS) | set(SHORT_DATASETS) | {'synthetic'}
    resolved = []
    for name in requested:
        if name in DATASET_GROUPS:
            resolved.extend(DATASET_GROUPS[name])
        elif name in known_datasets:
            resolved.append(name)
        else:
            valid = sorted(known_datasets | set(DATASET_GROUPS))
            raise ValueError(f'Unsupported dataset {name!r}. Expected one of: {", ".join(valid)}')

    deduped = []
    seen = set()
    for name in resolved:
        if name not in seen:
            deduped.append(name)
            seen.add(name)
    return deduped


def get_parse():
    parser = argparse.ArgumentParser(description='Run lossless KV cache CPU offloading.')
    parser.add_argument('--model_config_name', type=str, default='qwen2_14B_config')
    parser.add_argument('--compress_weights', action='store_true')
    parser.add_argument('--compress_kv_cache', action='store_true', help='Compatibility flag: use Float11Cache when kv_cache_impl=auto.')
    parser.add_argument(
        '--kv_cache_impl',
        type=str,
        default='auto',
        choices=('auto', 'dynamic', 'cpu_offload', 'float11', 'mla_float11', 'nvcomp_ans'),
        help='Which KV cache implementation to use.',
    )
    parser.add_argument('--enable_cpu_offload', action='store_true', help='Use lossless CPU offloaded KV cache.')
    parser.add_argument('--cpu_offload_ratio', type=float, default=0.25, help='Approximate fraction of KV cache kept on CPU.')
    parser.add_argument('--cpu_offload_min_tokens', type=int, default=1)
    parser.add_argument('--disable_pin_memory', action='store_true')
    parser.add_argument('--block_size', type=int, default=128, help='Block size used by compressed KV cache implementations.')
    parser.add_argument(
        '--float11_layout',
        '--kv_layout',
        type=str,
        default='segmented_tiled',
        choices=('legacy', 'tiled', 'segmented_tiled'),
        dest='float11_layout',
        help='Layout for Float11Cache.',
    )
    parser.add_argument(
        '--float11_need_adjust',
        '--kv_adjuster_enabled',
        action='store_true',
        dest='float11_need_adjust',
        help='Enable Float11Cache adjuster.',
    )
    parser.add_argument(
        '--float11_update_steps',
        '--kv_adjust_update_steps',
        type=int,
        default=64,
        dest='float11_update_steps',
        help='Float11Cache adjust interval in decode steps.',
    )
    parser.add_argument('--float11_seq_tile_size', type=int, default=None)
    parser.add_argument(
        '--mla_cache_mode',
        type=str,
        default='auto',
        choices=('auto', 'latent', 'expanded'),
        help='MLA cache tensor mode. latent means compressed_kv + k_pe; expanded means regular full key/value.',
    )
    parser.add_argument('--mla_latent_right', type=int, default=127, help='Float11 exponent right bound for MLA compressed_kv/latent cache.')
    parser.add_argument('--mla_rope_right', type=int, default=127, help='Float11 exponent right bound for MLA k_pe/RoPE cache.')
    parser.add_argument('--mla_latent_nbits', type=int, default=4, help='Exponent bits for MLA compressed_kv/latent cache.')
    parser.add_argument('--mla_rope_nbits', type=int, default=4, help='Exponent bits for MLA k_pe/RoPE cache.')
    parser.add_argument('--mla_disable_rope_compression', action='store_true', help='Keep MLA k_pe/RoPE cache dense and only compress latent cache.')
    parser.add_argument('--disable_mla_shape_check', action='store_true', help='Disable MLA cache tensor shape warnings.')
    parser.add_argument('--nvcomp_uncomp_chunk_size', type=int, default=65536, help='nvCOMP ANS uncompressed chunk size.')
    parser.add_argument('--max_new_tokens', type=int, default=None)
    parser.add_argument('--min_new_tokens', type=int, default=0)
    parser.add_argument('--do_sample', action='store_true')
    parser.add_argument('--num_seqence', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_length', type=int, default=20480)
    parser.add_argument(
        '--datasets',
        '--dataset',
        nargs='+',
        default=None,
        help='Datasets to run. Defaults to longbench. Accepts concrete dataset names or groups: longbench, short, synthetic.',
    )
    parser.add_argument('--use_synthetic_data', action='store_true', help='Use generated input_ids instead of loading real datasets.')
    parser.add_argument('--synthetic_seq_length', type=int, default=2048, help='Exact input sequence length for synthetic data.')
    parser.add_argument('--synthetic_token_id', type=int, default=None, help='Use one fixed token id for synthetic data; random tokens are used by default.')
    parser.add_argument('--synthetic_seed', type=int, default=0, help='Random seed for synthetic input_ids.')
    parser.add_argument('--output_path', type=str, default=None)
    return parser


if __name__ == '__main__':
    os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
    args = get_parse().parse_args()

    model_config = model_config_map[args.model_config_name]()
    model_id = model_config.model_id
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else model_config.max_new_tokens
    dtype = torch.bfloat16
    mini_batch_size = max(1, int(args.batch_size))
    total_seqs = int(args.num_seqence)
    kv_cache_impl = _resolve_kv_cache_impl(args)
    gpu_keep_ratio = max(1e-6, min(1.0, 1.0 - float(args.cpu_offload_ratio)))

    output_path = args.output_path
    if output_path is None:
        output_path = (
            f"kv_lossless_{kv_cache_impl}_{model_id.split('/')[-1]}_"
            f"weights_{bool(args.compress_weights)}_offload_{bool(args.enable_cpu_offload)}.json"
        )

    model_load_kwargs = {
        'dtype': dtype,
        'device_map': 'auto',
    }
    visible_cuda_max_memory = _visible_cuda_max_memory()
    if visible_cuda_max_memory is not None and len(visible_cuda_max_memory) > 1:
        model_load_kwargs['max_memory'] = visible_cuda_max_memory
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_load_kwargs)
    model.eval()
    if kv_cache_impl != 'dynamic' and _model_has_non_cuda_placements(model):
        raise RuntimeError(
            'Compressed or offloaded KV cache expects decoder layers to stay on CUDA; '
            f'found non-CUDA model placements: {getattr(model, "hf_device_map", None)}'
        )
    if args.compress_weights:
        model = convert_model(model, model_config.config_dict, is_tie_weight=model.config.tie_word_embeddings)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset_names = _resolve_dataset_names(args)
    all_info_lists = {}
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            all_info_lists = json.load(f)

    for dn in dataset_names:
        use_synthetic = dn == 'synthetic'
        dataset = None if use_synthetic else DatasetsLoader(dn)
        if use_synthetic:
            dataset_total = total_seqs
        else:
            dataset_total = min(len(dataset), total_seqs)
        all_info_lists[dn] = []
        for start_idx in range(0, dataset_total, mini_batch_size):
            end_idx = min(start_idx + mini_batch_size, dataset_total)
            prompts = None if use_synthetic else dataset[start_idx:end_idx]
            prompt_cache = None
            outputs = None
            with torch.no_grad():
                try:
                    info_dict = {
                        'dataset': dn,
                        'sample_start_idx': int(start_idx),
                        'batch_size': int(end_idx - start_idx),
                        'cache_mode': kv_cache_impl,
                        'cpu_offload_ratio_requested': float(args.cpu_offload_ratio),
                        'gpu_keep_ratio': float(gpu_keep_ratio),
                        'do_sample': bool(args.do_sample),
                        'max_new_tokens': int(max_new_tokens),
                        'min_new_tokens': int(args.min_new_tokens),
                        'block_size': int(args.block_size),
                        'input_mode': 'synthetic' if use_synthetic else 'dataset',
                    }
                    if use_synthetic:
                        info_dict['synthetic_seq_length'] = int(args.synthetic_seq_length)
                        info_dict['synthetic_token_id'] = args.synthetic_token_id
                        info_dict['synthetic_seed'] = int(args.synthetic_seed)
                    if kv_cache_impl == 'float11':
                        info_dict['float11_layout'] = args.float11_layout
                        info_dict['float11_need_adjust'] = bool(args.float11_need_adjust)
                        info_dict['float11_update_steps'] = int(args.float11_update_steps)
                    if kv_cache_impl == 'mla_float11':
                        info_dict['float11_layout'] = args.float11_layout
                        info_dict['float11_need_adjust'] = bool(args.float11_need_adjust)
                        info_dict['float11_update_steps'] = int(args.float11_update_steps)
                        info_dict['mla_cache_mode'] = args.mla_cache_mode
                        info_dict['mla_latent_right'] = int(args.mla_latent_right)
                        info_dict['mla_rope_right'] = int(args.mla_rope_right)
                        info_dict['mla_latent_nbits'] = int(args.mla_latent_nbits)
                        info_dict['mla_rope_nbits'] = int(args.mla_rope_nbits)
                        info_dict['mla_compress_rope'] = not bool(args.mla_disable_rope_compression)
                    if kv_cache_impl == 'nvcomp_ans':
                        info_dict['nvcomp_uncomp_chunk_size'] = int(args.nvcomp_uncomp_chunk_size)
                    prompt_cache = _build_kv_cache(args, model, model_config, kv_cache_impl)

                    if use_synthetic:
                        new_inputs = _make_synthetic_inputs(args, model, tokenizer, end_idx - start_idx, start_idx)
                    else:
                        new_inputs = tokenizer(
                            prompts,
                            padding=True,
                            padding_side='left',
                            return_tensors='pt',
                            max_length=args.max_length,
                            truncation=True,
                        ).to(_model_primary_device(model))
                    input_lengths = new_inputs['attention_mask'].sum(dim=1).float()
                    info_dict['input_shape'] = [int(v) for v in new_inputs['input_ids'].shape]
                    info_dict['attention_mask_shape'] = [int(v) for v in new_inputs['attention_mask'].shape]
                    info_dict['total_input_tokens'] = input_lengths.sum().item()
                    info_dict['input_mean_len'] = input_lengths.mean().item()

                    generation_config = GenerationConfig(
                        max_new_tokens=max_new_tokens,
                        min_new_tokens=args.min_new_tokens,
                        do_sample=bool(args.do_sample),
                        return_dict_in_generate=True,
                        output_scores=False,
                        output_attentions=False,
                        output_hidden_states=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                    if not args.do_sample:
                        generation_config.temperature = None
                        generation_config.top_p = None
                        generation_config.top_k = None

                    if torch.cuda.is_available():
                        reset_peak_memory()
                    begin_time = time.time()
                    outputs = model.generate(
                        **new_inputs,
                        past_key_values=prompt_cache,
                        generation_config=generation_config,
                    )
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    generate_time = time.time() - begin_time
                    info_dict['generate_time'] = generate_time

                    sequences = outputs.sequences if hasattr(outputs, 'sequences') else outputs
                    prompt_width = int(new_inputs['input_ids'].shape[1])
                    info_dict['generated_output_head_32'] = _extract_generated_output_head(
                        tokenizer,
                        sequences,
                        prompt_width=prompt_width,
                        preview_tokens=32,
                    )
                    output_lengths = torch.sum(sequences != tokenizer.pad_token_id, dim=1).float()
                    info_dict['total_output_tokens'] = output_lengths.sum().item()
                    info_dict['output_mean_len'] = output_lengths.mean().item()
                    info_dict['total_new_tokens'] = info_dict['total_output_tokens'] - info_dict['total_input_tokens']
                    info_dict['tokens_per_second'] = (
                        info_dict['total_new_tokens'] / generate_time if generate_time > 0 else 0.0
                    )
                    if hasattr(prompt_cache, 'get_memory'):
                        info_dict.update(prompt_cache.get_memory())
                    if torch.cuda.is_available():
                        info_dict['cuda_peak_allocated_gib'] = sum(
                            torch.cuda.max_memory_allocated(device=torch.device(f'cuda:{i}')) / (1024 ** 3)
                            for i in range(torch.cuda.device_count())
                        )
                    all_info_lists[dn].append(_sanitize_for_json(info_dict))
                    Path(output_path).write_text(json.dumps(all_info_lists, indent=2), encoding='utf-8')
                finally:
                    if hasattr(prompt_cache, 'synchronize_and_release'):
                        prompt_cache.synchronize_and_release()
                    if outputs is not None and hasattr(outputs, 'past_key_values'):
                        outputs.past_key_values = None
                    del prompt_cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
