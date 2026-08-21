import json
import numpy as np
# import matplotlib.pyplot as plt
import math

def topk_with_index(arr, k):
    indexed = sorted(enumerate(arr), key=lambda x: x[1], reverse=True)
    topk_indexed = indexed[:k]
    indices = [item[0] for item in topk_indexed]
    values = [item[1] for item in topk_indexed]
    
    return values, indices

def report_current_bits(data_lists, titles, left=115, right=130, nbit=4):
    # print(f"Report the compress ratio for indices from {left} to {right}, where nbit is {nbit}")
    for i, data_list in enumerate(data_lists):
        total_numbers = sum(data_list)
        original_memory = total_numbers * 8

        indices = [indice for indice in range(left, right+1)]
        values = [data_list[indice] for indice in range(left, right + 1)]
        total_norm_values = sum(values)
        outlier_values = total_numbers - total_norm_values
        compressed_memory = total_norm_values * nbit + outlier_values * (8+32)
        compress_ratio = (total_numbers * 8 + compressed_memory) / (total_numbers * 8 + original_memory)
        # print(f"{titles[i]=}, {nbit=}, {indices=}, compressed_memory={compressed_memory/8/1024/1024} MB, {compress_ratio=}")

def analysis_best_bits_coo(data_lists, titles, store_index=False):
    nbits = [1, 2, 3, 4, 5, 6, 7, 8]
    res_rights = []
    res_nbits = []
    for i, data_list in enumerate(data_lists):
        total_numbers = sum(data_list)
        original_memory = total_numbers * 8
        best_compressed_memory = original_memory
        best_indices = []
        best_nbit = 0
        best_outlier_values = 0
        for nbit in nbits:
            number_space = 2**nbit
            values, indices = topk_with_index(data_list, number_space)
            total_norm_values = sum(values)
            outlier_values = total_numbers - total_norm_values
            compressed_memory = total_numbers * nbit + outlier_values * 8
            if store_index is True:
                compressed_memory += outlier_values * 32
            if compressed_memory < best_compressed_memory:
                best_compressed_memory = compressed_memory
                best_indices = indices
                best_nbit = nbit
                best_outlier_values = outlier_values
        # if total_numbers * 8 + original_memory == 0:
        compress_ratio = (total_numbers * 8 + best_compressed_memory) / (total_numbers * 8 + original_memory)
        best_indices = sorted(best_indices)
        best_right = best_indices[-1]
        res_rights.append(best_right)
        res_nbits.append(best_nbit)
        # print(f"{titles[i]=}, {best_outlier_values=}, {best_outlier_values/total_numbers=}, {best_nbit=}, {best_indices=}, best_compressed_memory={best_compressed_memory/8/1024/1024} MB, {compress_ratio=}")
    return res_rights, res_nbits

def analysis_best_bits(data_lists, titles, is_coo=False):
    nbits = [1, 2, 3, 4, 5, 6, 7, 8]
    res_rights = []
    res_nbits = []
    for i, data_list in enumerate(data_lists):
        total_numbers = sum(data_list)
        original_memory = total_numbers * 8
        best_compressed_memory = original_memory
        best_indices = []
        best_nbit = 0
        best_outlier_values = 0
        for nbit in nbits:
            if is_coo:
                number_space = 2**nbit
            else:
                number_space = 2**nbit - 1

            values, indices = topk_with_index(data_list, number_space)
            total_norm_values = sum(values)
            outlier_values = total_numbers - total_norm_values
            compressed_memory = total_numbers * nbit + outlier_values * 8

            if is_coo is True:
                # store index
                compressed_memory += outlier_values * 32

            if compressed_memory < best_compressed_memory:
                best_compressed_memory = compressed_memory
                best_indices = indices
                best_nbit = nbit
                best_outlier_values = outlier_values
        # if total_numbers * 8 + original_memory == 0:
        compress_ratio = (total_numbers * 8 + best_compressed_memory) / (total_numbers * 8 + original_memory)
        best_indices = sorted(best_indices)
        best_right = best_indices[-1]
        res_rights.append(best_right)
        res_nbits.append(best_nbit)
        # print(f"{titles[i]=}, {best_outlier_values=}, {best_outlier_values/total_numbers=}, {best_nbit=}, {best_indices=}, best_compressed_memory={best_compressed_memory/8/1024/1024} MB, {compress_ratio=}")
    return res_rights, res_nbits

def _resolve_component_names(freq_dict):
    if 'prefill_keys' in freq_dict and 'prefill_values' in freq_dict:
        return 'keys', 'values'
    if 'prefill_latent' in freq_dict and 'prefill_rope' in freq_dict:
        return 'latent', 'rope'
    available_keys = sorted(freq_dict.keys())
    raise KeyError(
        'Cannot resolve cache component names from frequency dict. '
        f'Expected keys/values or latent/rope fields, got {available_keys}'
    )

def get_freq_by_keys(freq_dict, prefix_key):
    first_name, second_name = _resolve_component_names(freq_dict)
    if prefix_key == 'merge':
        keys_freq = [p+d for p, d in zip(freq_dict[f'prefill_{first_name}'], freq_dict[f'decode_{first_name}'])]
        values_freq = [p+d for p, d in zip(freq_dict[f'prefill_{second_name}'], freq_dict[f'decode_{second_name}'])]
    elif prefix_key == 'prefill':
        keys_freq = freq_dict[f'prefill_{first_name}']
        values_freq = freq_dict[f'prefill_{second_name}']
    elif prefix_key == 'decode':
        keys_freq = freq_dict[f'decode_{first_name}']
        values_freq = freq_dict[f'decode_{second_name}']
    return keys_freq, values_freq

def merge_layer_key_value(layer_freq, infer_stage):
    merged_layer_freq = []
    for layer_idx, kv in layer_freq.items():
        first_name, second_name = _resolve_component_names(kv)
        if infer_stage == 'merge':
            prefill_keys_freq = kv[f'prefill_{first_name}']
            prefill_values_freq = kv[f'prefill_{second_name}']
            decode_keys_freq = kv[f'decode_{first_name}']
            decode_values_freq = kv[f'decode_{second_name}']
            keys_freq = [p+d for p, d in zip(prefill_keys_freq, decode_keys_freq)]
            values_freq = [p+d for p, d in zip(prefill_values_freq, decode_values_freq)]
        else:
            keys_freq = kv[f'{infer_stage}_{first_name}']
            values_freq = kv[f'{infer_stage}_{second_name}']
        if len(merged_layer_freq) == 0:
            merged_layer_freq = [keys_freq[i] + values_freq[i] for i in range(len(keys_freq))]
        else:
            for i in range(len(merged_layer_freq)):
                merged_layer_freq[i] = merged_layer_freq[i] + keys_freq[i] + values_freq[i]
    return merged_layer_freq


def analysis_exp_freq_datasets(json_file_path, infer_stage, is_coo=False):
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    dataset_names = []
    for ds in list(data.keys()):
        dataset_names.append(ds)

    data_lists = [merge_layer_key_value(data[dn], infer_stage) for dn in dataset_names]

    # Activation Frequency of Experts in the 
    titles = [f'{dataset_names[i]}' for i in range(len(dataset_names))]
    analysis_best_bits(data_lists, titles, is_coo)


def analysis_exp_freq_keys(json_file_path, infer_stage, dataset_index=0):
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    dataset_name = list(data.keys())[dataset_index]
    # print(f"{dataset_name=}")

    layers = list(data[dataset_name].keys())
    # [3*i + 1 for i in range(1, 9)]
    data_lists = [get_freq_by_keys(data[dataset_name][str(layer_idx)], infer_stage)[0] for layer_idx in layers]
    titles = [f'{i}-th layers' for i in layers]

    # ����ֱ��ͼ
    analysis_best_bits(data_lists, titles)

def analysis_exp_freq_values(json_file_path, infer_stage, dataset_index=0):
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    dataset_name = list(data.keys())[dataset_index]
    # print(f"{dataset_name=}")

    layers = list(data[dataset_name].keys())
    # layers = [3*i + 1 for i in range(1, 9)]
    data_lists = [get_freq_by_keys(data[dataset_name][str(layer_idx)], infer_stage)[1] for layer_idx in layers]
    titles = [f'{i}-th layers' for i in layers]
    analysis_best_bits(data_lists, titles)


def export_initial_compress_parameters(json_file_path, dataset_name='c4', infer_stage='merge'):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    layers = list(data[dataset_name].keys())
    keys_lists = [get_freq_by_keys(data[dataset_name][str(layer_idx)], infer_stage)[0] for layer_idx in layers]
    values_lists = [get_freq_by_keys(data[dataset_name][str(layer_idx)], infer_stage)[1] for layer_idx in layers]
    titles = [f'{i}-th layers' for i in layers]
    keys_rights, keys_nbits = analysis_best_bits_coo(keys_lists, titles)
    values_rights, values_nbits = analysis_best_bits_coo(values_lists, titles)
    rights = [(kr, vr) for kr, vr in zip(keys_rights, values_rights)]
    nbits = [(kn, vn) for kn, vn in zip(keys_nbits, values_nbits)]
    # print(f"{rights=}")
    # print(f"{nbits=}")
    return rights, nbits

if __name__ == '__main__':
    json_file_path='./best_bit_analysis/c4_Qwen3-30B-A3B-Thinking-2507_exp.json'
    export_initial_compress_parameters(json_file_path)
