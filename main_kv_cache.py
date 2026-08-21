import time
import math
import os
import gc
import json
import argparse
from contextlib import nullcontext
from pathlib import Path

# import tpdm

from typing import Any, Optional

import torch
from torch.profiler import profile, record_function, ProfilerActivity, schedule
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, StaticCache, Cache, GenerationConfig
from datasets import load_dataset

from kv_compression.float_cache_utils import Float11Cache
from utils import DatasetsLoader
from manager import convert_model
from model_configs import qwen2_14B_config, phi_4_config, ds_dis_llama_8b_config, ds_dis_qwen_7b_config, model_config_map
from utils import get_allocated_peak_memory, reset_peak_memory, get_device_map_accelerate


def _sanitize_for_json(value):
    if isinstance(value, dict):
        return {k: _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_for_json(v) for v in value]
    return value


def _extract_generated_output_head(
    tokenizer,
    sequences: torch.Tensor,
    prompt_width: int,
    preview_tokens: int = 32,
):
    generated_sequences = sequences[:, prompt_width:]
    previews = []
    pad_token_id = tokenizer.pad_token_id
    for row in generated_sequences:
        row_cpu = row.detach().to('cpu')
        if pad_token_id is not None:
            row_cpu = row_cpu[row_cpu != pad_token_id]
        head_tokens = row_cpu[:preview_tokens]
        previews.append([int(token_id) for token_id in head_tokens.tolist()])
    return previews


def _safe_file_stem(value: str) -> str:
    return ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in value)


def _model_cuda_devices(model) -> list[str]:
    device_map = getattr(model, 'hf_device_map', None)
    if not isinstance(device_map, dict):
        return []
    devices = set()
    for placement in device_map.values():
        if isinstance(placement, int):
            devices.add(f'cuda:{placement}')
            continue
        if isinstance(placement, str) and placement.startswith('cuda'):
            devices.add(placement)
    return sorted(devices)


def _model_has_non_cuda_placements(model) -> bool:
    device_map = getattr(model, 'hf_device_map', None)
    if not isinstance(device_map, dict):
        return False
    for placement in device_map.values():
        if isinstance(placement, str) and placement in {'cpu', 'disk'}:
            return True
    return False


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


def _build_profiler(args, sample_idx: int, cache_mode: str):
    if not args.profile_generate:
        return None, nullcontext(), None

    trace_dir = Path(args.profile_output_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    cache_mode_stem = _safe_file_stem(cache_mode)
    sample_stem = f"sample{sample_idx}_{cache_mode_stem}"
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    prof_schedule = None
    if args.profile_wait or args.profile_warmup or args.profile_active:
        prof_schedule = schedule(
            wait=max(0, int(args.profile_wait)),
            warmup=max(0, int(args.profile_warmup)),
            active=max(1, int(args.profile_active)),
            repeat=1,
        )

    prof = profile(
        activities=activities,
        schedule=prof_schedule,
        record_shapes=bool(args.profile_record_shapes),
        profile_memory=bool(args.profile_memory),
        with_stack=bool(args.profile_with_stack),
    )
    return prof, prof, trace_dir / sample_stem


def _finalize_profiler(prof, trace_base_path: Optional[Path], info_dict: dict):
    if prof is None:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    trace_path = trace_base_path.with_suffix('.chrome_trace.json') if trace_base_path is not None else None
    if trace_path is not None:
        prof.export_chrome_trace(str(trace_path))
        info_dict['profiler_trace_path'] = str(trace_path)
    self_cuda_sort = 'self_cuda_time_total' if torch.cuda.is_available() else 'self_cpu_time_total'
    total_cuda_sort = 'cuda_time_total' if torch.cuda.is_available() else 'cpu_time_total'
    self_table = prof.key_averages().table(sort_by=self_cuda_sort, row_limit=100)
    total_table = prof.key_averages().table(sort_by=total_cuda_sort, row_limit=100)
    info_dict['profiler_self_table_sort_by'] = self_cuda_sort
    info_dict['profiler_total_table_sort_by'] = total_cuda_sort
    info_dict['profiler_self_table'] = self_table
    info_dict['profiler_total_table'] = total_table
    if trace_base_path is not None:
        self_table_path = trace_base_path.with_suffix('.self_table.txt')
        total_table_path = trace_base_path.with_suffix('.total_table.txt')
        self_table_path.write_text(self_table, encoding='utf-8')
        total_table_path.write_text(total_table, encoding='utf-8')
        info_dict['profiler_self_table_path'] = str(self_table_path)
        info_dict['profiler_total_table_path'] = str(total_table_path)


def get_parse():
    parser = argparse.ArgumentParser(description='Script for kv cache compression model.')
    # For compression
    parser.add_argument('--model_config_name', type=str, default='qwen2_14B_config', help='Which model config can be used.')
    parser.add_argument('--compress_kv_cache', action='store_true', help='Which compress the kv cache.')
    parser.add_argument('--compress_weights', action='store_true', help='Which compress the model weights.')
    parser.add_argument('--block_size', type=int, default=128, help='Float11Cache compression flush block size.')
    parser.add_argument(
        '--kv_layout',
        type=str,
        default='auto',
        choices=('auto', 'legacy', 'tiled', 'segmented_tiled'),
        help='KV cache layout. auto uses legacy.',
    )
    parser.add_argument('--max_new_tokens', type=int, default=None, help='Override max_new_tokens from model config.')
    parser.add_argument('--min_new_tokens', type=int, default=0, help='Force a minimum number of generated tokens before EOS.')
    parser.add_argument('--do_sample', action='store_true', help='Enable sampling during generation.')
    parser.add_argument('--kv_adjuster_enabled', action='store_true', help='Enable adaptive KV compressor adjustment during generation.')
    parser.add_argument('--kv_adjust_update_steps', type=int, default=64, help='Decode-step interval for KV compressor adjustment; 0 disables updates.')
    parser.add_argument(
        '--kv_auto_init_compress_params',
        action='store_true',
        help='Compute initial KV compression parameters from runtime KV tensors before the first compression.',
    )
    parser.add_argument('--kv_experiment_name', type=str, default=None, help='Optional label included in output file names and JSON records.')
    parser.add_argument('--profile_generate', action='store_true', help='Profile a short generate() run with torch.profiler.')
    parser.add_argument('--profile_output_dir', type=str, default='profiler_outputs', help='Directory for profiler tables and Chrome traces.')
    parser.add_argument('--profile_record_shapes', action='store_true', help='Record operator input shapes in profiler output.')
    parser.add_argument('--profile_memory', action='store_true', help='Record memory usage in profiler output.')
    parser.add_argument('--profile_with_stack', action='store_true', help='Record Python stacks in profiler output.')
    parser.add_argument('--profile_wait', type=int, default=0, help='Profiler schedule wait steps before capture.')
    parser.add_argument('--profile_warmup', type=int, default=0, help='Profiler schedule warmup steps before active capture.')
    parser.add_argument('--profile_active', type=int, default=1, help='Profiler schedule active steps to capture.')
    parser.add_argument('--num_seqence', type=int, default=64, help='The total sequence.')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of sequences to process in parallel.')
    parser.add_argument('--max_length', type=int, default=20480, help='The max length of input sequence.')
    return parser



if __name__ == "__main__":
    os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
    re_start = True

    args = get_parse().parse_args()
    compress_weights = args.compress_weights
    compress_kv_cache = args.compress_kv_cache
    kv_adjuster_enabled = bool(args.kv_adjuster_enabled)
    kv_adjust_update_steps = int(args.kv_adjust_update_steps)
    effective_kv_adjust_update_steps = kv_adjust_update_steps if kv_adjuster_enabled else 0
    kv_layout = args.kv_layout
    if kv_layout == 'auto':
        kv_layout = 'legacy'
    model_config = model_config_map[args.model_config_name]()
    if args.kv_auto_init_compress_params:
        model_config.rights = None
        model_config.nbits = None
    # deepseekr1_distill_llama8b_config()
    # qwen3_4B_config()
    output_prefix_name = 'kv_test'
    # './datasets/jet-ai/longbench'
    model_id = model_config.model_id
    experiment_suffix = ''
    if args.kv_experiment_name:
        experiment_suffix = f"_{_safe_file_stem(args.kv_experiment_name)}"
    auto_params_suffix = '_auto_params' if args.kv_auto_init_compress_params else ''
    output_path = f"{output_prefix_name}_{model_id.split('/')[-1]}_{compress_weights}_{compress_kv_cache}_adjuster_{kv_adjuster_enabled}_steps{kv_adjust_update_steps}{auto_params_suffix}{experiment_suffix}_info.json"
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else model_config.max_new_tokens
    min_new_tokens = args.min_new_tokens
    # 4096
    dtype = torch.bfloat16
    mini_batch_size = max(1, int(args.batch_size))
    total_seqs = args.num_seqence

    model_load_kwargs = {
        'dtype': dtype,
        'device_map': 'auto',
    }
    visible_cuda_max_memory = _visible_cuda_max_memory()
    if visible_cuda_max_memory is not None and len(visible_cuda_max_memory) > 1:
        # Triton KV compression/decode assumes each layer lives entirely on CUDA.
        # Constrain Accelerate to shard only across visible GPUs so it does not
        # silently place decoder blocks on CPU in multi-GPU runs.
        model_load_kwargs['max_memory'] = visible_cuda_max_memory
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_load_kwargs)
    model.eval()
    model_cuda_devices = _model_cuda_devices(model)
    if compress_kv_cache and _model_has_non_cuda_placements(model):
        raise RuntimeError(
            'Compressed KV cache requires all decoder layers to stay on CUDA, but '
            f'Accelerate placed part of the model on non-CUDA devices: {getattr(model, "hf_device_map", None)}'
        )
    if compress_weights:
        model = convert_model(model, model_config.config_dict, is_tie_weight=model.config.tie_word_embeddings)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
# '2wikimqa', 'gov_report', 'multi_news', 'musique', 'multifieldqa_en',
# 'passage_count', 'passage_retrieval_en', 'qasper', 'qmsum', 'hotpotqa', 'lcc', \
                    # 'repobench-p', 'samsum','trec', 'triviaqa'

    # for qwen_1p5_moe_config:
    # 'narrativeqa', \
    #                  'repobench-p', 'triviaqa'
    # 'riddlebench', 'simplemath', 'c4', 

    longbench_dataset = ['2wikimqa', 'gov_report', 'multi_news', 'musique', 'multifieldqa_en', 'narrativeqa', 'passage_count', 'passage_retrieval_en', 'qasper', 'qmsum', 'hotpotqa', 'lcc', 'repobench-p', 'samsum','trec', 'triviaqa']
    short_datasets = ['riddlebench', 'simplemath', 'aime25', 'aime24', 'livecodebench', 'mbpp']

    dataset_name = short_datasets
    # ['2wikimqa', 'gov_report', 'multi_news', 'musique']
    # [ 'livecodebench', 'mbpp', 'aime24', 'aime25']
    all_info_lists = {}
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as file:
            all_info_lists = json.load(file)

    for dn in dataset_name:
        dataset = DatasetsLoader(dn)
        total_seqs = min(len(dataset), total_seqs)
        prompt_cache = None

        if re_start is False:
            if dn in all_info_lists.keys() and len(all_info_lists[dn]) != 0:
                print(f"find {dn}, skip")
                continue

        all_info_lists[dn] = []
        for start_idx in range(0, total_seqs, mini_batch_size):
            end_idx = min(start_idx + mini_batch_size, total_seqs)
            prompts = dataset[start_idx:end_idx]
            prompt_cache = None
            outputs = None
            with torch.no_grad():
                try:
                    info_dict = {}
                    info_dict['attention_backend_requested'] = 'native'
                    info_dict['attention_backend'] = 'native'
                    info_dict['prefill_time'] = 0.0
                    info_dict['decode_time'] = 0.0
                    info_dict['prefill_peak_memory'] = 0.0
                    info_dict['decode_peak_memory'] = 0.0
                    info_dict['decode_tokens'] = 0
                    info_dict['tokens_per_second_decode'] = 0.0
                    info_dict['stopped_early'] = False
                    info_dict['eos_reached'] = False
                    info_dict['stop_reason'] = 'unknown'
                    info_dict['do_sample'] = bool(args.do_sample)
                    info_dict['max_new_tokens'] = int(max_new_tokens)
                    info_dict['min_new_tokens'] = int(min_new_tokens)
                    info_dict['profiler_enabled'] = bool(args.profile_generate)
                    info_dict['kv_layout'] = kv_layout
                    if args.kv_auto_init_compress_params:
                        info_dict['kv_auto_init_compress_params'] = True
                    info_dict['kv_experiment_name'] = args.kv_experiment_name
                    if compress_kv_cache:
                        prompt_cache = Float11Cache(
                            config=model.config,
                            right=model_config.rights,
                            exp_bits=model_config.nbits,
                            need_adjust=kv_adjuster_enabled,
                            update_steps=effective_kv_adjust_update_steps,
                            residual_length=0,
                            block_size=args.block_size,
                            layout=kv_layout,
                            auto_init_compress_params=bool(args.kv_auto_init_compress_params),
                        )
                        info_dict['kv_adjuster_enabled'] = bool(prompt_cache.need_adjust)
                        info_dict['kv_adjust_update_steps'] = int(getattr(prompt_cache, 'update_steps', 0))
                        info_dict['cache_mode'] = 'compressed_materialize_decode'
                        info_dict['cache_decode_impl'] = 'materialize_dense'
                    else:
                        prompt_cache = DynamicCache(config=model.config)
                        info_dict['cache_mode'] = 'torch_dynamic_cache'
                        info_dict['cache_decode_impl'] = 'native'
 
                    new_inputs = tokenizer(prompts, padding=True, padding_side='left', return_tensors="pt", max_length=args.max_length, truncation=True)
                    new_inputs = new_inputs.to(_model_primary_device(model))

                    input_lengths = new_inputs['attention_mask'].sum(dim=1).float()
                    info_dict['batch_size'] = int(new_inputs['input_ids'].shape[0])
                    info_dict['input_shape'] = [int(v) for v in new_inputs['input_ids'].shape]
                    info_dict['attention_mask_shape'] = [int(v) for v in new_inputs['attention_mask'].shape]
                    info_dict['total_input_tokens'] = input_lengths.sum().item()
                    info_dict['input_mean_len'] = input_lengths.mean().item()
                    generation_config = GenerationConfig(
                        max_new_tokens=max_new_tokens,
                        min_new_tokens=min_new_tokens,
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

                    prof, prof_context, profiler_base_path = _build_profiler(
                        args,
                        sample_idx=start_idx,
                        cache_mode=info_dict['cache_mode'],
                    )
                    if torch.cuda.is_available():
                        reset_peak_memory()
                    prefill_begin_time = time.time()
                    with prof_context:
                        with record_function('float11.model_generate'):
                            outputs = model.generate(
                                **new_inputs,
                                past_key_values=prompt_cache,
                                generation_config=generation_config,
                            )
                        if prof is not None:
                            prof.step()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        total_generate_peak = sum(
                            torch.cuda.max_memory_allocated(device=torch.device(f'cuda:{i}')) / (1024 ** 3)
                            for i in range(torch.cuda.device_count())
                        )
                        # generate() internally包含prefill+decode，无法在不改生成流程的情况下严格拆分峰值。
                        info_dict['decode_peak_memory'] = total_generate_peak
                        info_dict['prefill_peak_memory'] = total_generate_peak
                    prefill_end_time = time.time()

                    info_dict['prefill_time'] = 0.0
                    info_dict['decode_time'] = prefill_end_time - prefill_begin_time
                    info_dict['generate_time'] = info_dict['decode_time']

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
                    info_dict['decode_tokens'] = max(0, int(info_dict['total_new_tokens']))
                    info_dict['tokens_per_second_decode'] = (
                        info_dict['decode_tokens'] / info_dict['decode_time'] if info_dict['decode_time'] > 0 else 0.0
                    )
                    info_dict['stopped_early'] = bool(info_dict['decode_tokens'] < max_new_tokens)
                    eos_token_id = tokenizer.eos_token_id
                    if eos_token_id is not None and sequences.shape[-1] > 0:
                        info_dict['eos_reached'] = bool((sequences[:, -1] == eos_token_id).any().item())
                    info_dict['stop_reason'] = 'eos' if info_dict['eos_reached'] else ('max_new_tokens' if not info_dict['stopped_early'] else 'other')

                    # info_dict['prefill_throughput'] = info_dict['total_input_tokens'] / info_dict['generate_time']
                    # info_dict['decode_throughput'] = info_dict['total_new_tokens'] / info_dict['generate_time']
                    info_dict['e2e_throughput'] = info_dict['total_output_tokens'] / info_dict['generate_time']

                    allocated, reserved, peak_memory = get_allocated_peak_memory()
                    info_dict['allocated_memory'] = allocated
                    info_dict['reserved_memory'] = reserved
                    info_dict['peak_memory'] = peak_memory

                    func = getattr(prompt_cache, 'get_memory', None)
                    if callable(func):
                        mem_dict = prompt_cache.get_memory()
                        info_dict.update(mem_dict)
                        if isinstance(prompt_cache, Float11Cache):
                            info_dict['compressed_block_flushes'] = int(mem_dict.get('compressed_block_flushes', 0))
                            info_dict['compressed_tokens_flushed'] = int(mem_dict.get('compressed_tokens_flushed', 0))
                            info_dict['residual_tokens_kept'] = int(mem_dict.get('residual_tokens_kept', 0))
                            info_dict['stored_compressed_block_count'] = int(mem_dict.get('stored_compressed_block_count', 0))
                            info_dict['stored_residual_seq_len'] = int(mem_dict.get('stored_residual_seq_len', 0))
                            info_dict['cache_view_returned_block_count'] = int(mem_dict.get('cache_view_returned_block_count', 0))
                            if args.kv_auto_init_compress_params:
                                get_params = getattr(prompt_cache, 'get_compress_params', None)
                                if callable(get_params):
                                    info_dict.update(get_params())
                    _finalize_profiler(prof, profiler_base_path, info_dict)
                    print(_sanitize_for_json(info_dict))
                    all_info_lists[dn].append(_sanitize_for_json(info_dict))
                    json.dump(all_info_lists, open(output_path, 'w'))
                except Exception as e:
                    import traceback
                    # print(f"{type(e).__name__}: {e}")
                    traceback.print_exc()
                    print(f"skip {dn} tasks")
                finally:
                    if isinstance(prompt_cache, Float11Cache):
                        try:
                            prompt_cache.synchronize_and_release()
                        except Exception as release_exc:
                            print(f"Float11Cache release warning: {type(release_exc).__name__}: {release_exc}")
                    if outputs is not None and hasattr(outputs, 'past_key_values'):
                        outputs.past_key_values = None
                    del outputs
                    del prompt_cache
                    gc.collect()
                    if torch.cuda.is_available():
                        try:
                            torch.cuda.empty_cache()
                        except Exception as empty_cache_exc:
                            print(f"torch.cuda.empty_cache warning: {type(empty_cache_exc).__name__}: {empty_cache_exc}")
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception as empty_cache_exc:
                print(f"torch.cuda.empty_cache warning: {type(empty_cache_exc).__name__}: {empty_cache_exc}")
