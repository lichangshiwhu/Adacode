import os
import gc
import json
from collections import Counter

# import tpdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, StaticCache, Cache
from datasets import load_dataset

import triton
# from model_scripts import yicoderconfig, qwen2config, qwen1config
from compress import get_exp_kernel_constexpr
from utils import DatasetsLoader

def bf16_exponent_counts(t: torch.Tensor) -> Counter:
    uint16_view = t.view(torch.uint16).cpu().numpy().ravel()
    # 0x7F80 = 0111, 1111, 1000, 0000
    exponent = (uint16_view & 0x7F80) >> 7
    return Counter(exponent)

def get_freq_list_old(t, count_len = 2**8):
    t_counter = Counter()
    t_counter.update(bf16_exponent_counts(t.to(torch.bfloat16)))
    t_freq_list = [0 for _ in range(count_len)]
    for e in range(0, count_len):
        t_freq_list[e] = t_counter.get(e, 0)
    return t_freq_list

def get_exp_freq_list(t, count_len = 2**8):
    BLOCK_SIZE = 1024
    if not t.is_contiguous():
        t = t.contiguous()
    flatten_t = t.view(-1)
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
    t_freq_list = torch.bincount(extracted_values, minlength=count_len)
    return t_freq_list.cpu().tolist()

def get_exp_freq_list_fp16(t, count_len = 2**5):
    if not t.is_contiguous():
        t = t.contiguous()
    flatten_t = t.view(-1)
    total_size = flatten_t.numel()
    device = flatten_t.device

    uint16_view = flatten_t.view(torch.uint16)    
    exponent_bits = (uint16_view >> 10) & 0x1F
    t_freq_list = torch.bincount(exponent_bits, minlength=count_len)
    
    return t_freq_list.cpu().tolist()


def get_exp_freq_list_E4M3(t, count_len = 2**8):
    BLOCK_SIZE = 1024
    if not t.is_contiguous():
        t = t.contiguous()
    flatten_t = t.view(-1)
    total_size = flatten_t.numel()
    device = flatten_t.device
    floor(log2(flatten_t.abs()))
    extracted_values = torch.zeros(total_size, dtype=torch.uint8, device=device)
    grid1 = lambda meta: (triton.cdiv(total_size, meta['BLOCK_SIZE']), )
    get_exp_kernel_constexpr[grid1](
        flatten_t.view(torch.uint16), 
        extracted_values, 
        total_size, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    t_freq_list = torch.bincount(extracted_values, minlength=count_len)
    return t_freq_list.cpu().tolist()

def get_sign_freq_list(t):
    return [(t >= 0).sum().item(), (t <= 0).sum().item()]

def get_mantissa_freq_list(t, count_len = 2**7):
    t_uint16 = t.view(torch.uint16).to(torch.int32).flatten()
    mask = torch.tensor(0x007F, dtype=torch.int32)
    mask_t_uint16 = t_uint16 & mask
    t_freq_list = torch.bincount(mask_t_uint16, minlength=count_len)
    return t_freq_list.cpu().tolist()

def extract_exp_freq(kv_cache, freq_type, prefill_len=None, dtype='bf16'):
    if freq_type == 'sign':
        get_freq_list = get_sign_freq_list
    elif freq_type == 'exp':
        if dtype == 'bf16':
            get_freq_list = get_exp_freq_list
        elif dtype == 'fp16':
            get_freq_list = get_exp_freq_list_fp16
    elif freq_type == 'mantissa':
        get_freq_list = get_mantissa_freq_list

    exp_freq = {}
    count_len = 2**8
    for layer_idx in range(len(kv_cache.layers)):
        keys = kv_cache.layers[layer_idx].keys
        values = kv_cache.layers[layer_idx].values
        if prefill_len is not None:
            prefill_keys = keys[:, :, :prefill_len, :]
            prefill_values = values[:, :, :prefill_len, :]
            decode_keys = keys[:, :, prefill_len:, :]
            decode_values = values[:, :, prefill_len:, :]
            kv_pair = {'prefill_keys': get_freq_list(prefill_keys),
                       'prefill_values': get_freq_list(prefill_values),
                       'decode_keys': get_freq_list(decode_keys),
                       'decode_values': get_freq_list(decode_values)}
        else:
            key_freq_list = get_freq_list(keys)
            value_freq_list = get_freq_list(values)
            kv_pair['keys'] = key_freq_list
            kv_pair['values'] = value_freq_list
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

if __name__ == "__main__":
    gc.collect()
    os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
    # 01ai/Yi-Coder-1.5B-Chat
    # Qwen/Qwen2.5-1.5B
    # target_config = qwen2config
    LL_model_ids = ['/opt/data/private/Qwen/Qwen3-4B', '/opt/data/private/mistralai/Mistral-7B-Instruct-v0.3', \
                    '/opt/data/private/Qwen/Qwen1.5-MoE-A2.7B-Chat', '/opt/data/private/huihui-ai/Huihui-MoE-1.2B-A0.6B']
    # ['/opt/data/private/LLM-Research/Phi-4-reasoning-plus', '/opt/data/private/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
                # '/opt/data/private/deepseek-ai/DeepSeek-R1-Distill-Llama-8B', '/opt/data/private/Qwen/Qwen2.5-14B-Instruct']

    str2dtype={
        'bf16':torch.bfloat16,
    }
    output_prefix_name = 'kv_test'
    # model_id = '/opt/data/private/Qwen/Qwen3-4B'
    for model_id in LL_model_ids:
        dataset_name = ['mbpp', 'simplemath', 'aime25', 'livecodebench',]
        # ['2wikimqa', 'gov_report', 'multi_news', 'musique', 'multifieldqa_en', 'narrativeqa', \
                        # 'passage_count', 'passage_retrieval_en', 'qasper', 'qmsum', 'hotpotqa', 'lcc', \
                        # 'repobench-p', 'samsum','trec', 'triviaqa']
        # ,'mantissa', 'sign'
        for freq_type in ['sign', 'exp', 'mantissa']:
            max_new_tokens = 1024
            # torch.bfloat16
            str_dtype='bf16'
            dtype = str2dtype[str_dtype]
            batch_size = 32
            mini_batch_size = 4
            output_path = f"./model_configs/{output_prefix_name}_{model_id.split('/')[-1]}_{freq_type}.json"

            model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, trust_remote_code=True, device_map='auto')
            model.eval()
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            tokenizer.pad_token = tokenizer.eos_token

            dataset_kv_freq = {}
            for dn in dataset_name:
                gc.collect()
                dataset = DatasetsLoader(dn)
                print(dataset)
                for start_idx in range(0, batch_size, mini_batch_size):
                    end_idx = min(start_idx + mini_batch_size, batch_size)
                    prompts = dataset[start_idx:end_idx]
                    with torch.no_grad():
                        try:
                            prompt_cache = DynamicCache(config=model.config)
                            new_inputs = tokenizer(prompts, padding=True, padding_side='left', return_tensors="pt").to(model.device.type)
                            prefill_len = new_inputs.input_ids.shape[1]
                            outputs = model.generate(**new_inputs, past_key_values=prompt_cache, max_new_tokens=max_new_tokens)
                            if dn not in dataset_kv_freq:
                                dataset_kv_freq[dn] = extract_exp_freq(prompt_cache, freq_type, prefill_len=prefill_len, dtype=str_dtype)
                            else:
                                existing_freq = dataset_kv_freq[dn]
                                new_freq = extract_exp_freq(prompt_cache, freq_type, prefill_len=prefill_len, dtype=str_dtype)
                                dataset_kv_freq[dn] = merge_exp_freq(existing_freq, new_freq)
                            print(f"{dataset_kv_freq=}")
                            json.dump(dataset_kv_freq, open(output_path, 'w'))
                        except Exception as e:
                            print(f"{type(e).__name__}: {e}")
                            print(f"skip {dn} tasks")
                del prompt_cache
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()
            print(dataset_kv_freq)
            json.dump(dataset_kv_freq, open(output_path, 'w'))

