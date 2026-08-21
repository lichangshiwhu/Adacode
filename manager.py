import os
import enum
from typing import Optional

import torch
import transformers
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity


from compress_bitmap_merge import CompressorBitMap
# from compress_bitmap_where import CompressorBitMap

def get_mem_pool_info(model_config, model: torch.nn.Module):
    max_numel = 0
    max_weight_name = None
    for weight_name, nbit, right in model_config:

        parts = weight_name.split('.')
        if parts[-1] in ['weight', 'bias']:
            parts = parts[:-1]
        
        attr_name = parts[-1]
        if 'lm_head' in weight_name:
            parent_module = model
        elif len(parts) > 1:
            parent_path = '.'.join(parts[:-1])
            parent_module = model.get_submodule(parent_path)
        else:
            parent_module = model
        
        target_module = getattr(parent_module, attr_name)

        target_numel = target_module.weight.numel()
        if max_numel < target_numel:
            max_numel = max(max_numel, target_numel)
            max_weight_name = weight_name

    weight_info = {
        'max_numel':max_numel,
        'max_weight_name':max_weight_name
    }

    return weight_info

def get_decode_hook(block_id):
    def decode_hook(module, _):
        device = module.compressor.device
        # only prefill compression
        if device is None:
            return
        n_elements = module.compressor.n_elements
        reconstructed = torch.empty(n_elements, dtype=torch.bfloat16, device=device)
        # if block_id == 0:
        #     with profile(
        #         activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        #         record_shapes=True,
        #         with_flops=True,
        #     ) as prof:
        #         module.compressor.uncompress_with_weights(reconstructed.view(torch.uint16))
        #     print(prof.key_averages().table(
        #         sort_by="cuda_time_total",
        #         row_limit=20
        #     ))
        #     prof.export_chrome_trace("trace.json")
        module.compressor.uncompress_with_weights(reconstructed.view(torch.uint16))        
        weights_list = torch.tensor_split(reconstructed, module.split_positions)
        for child, model_shape, module_part, weight in zip(module.childen, module.model_shapes, module.module_part, weights_list):
            child.__dict__[module_part] = weight.view(model_shape)
        # only prefill compression
        module.compressor.reset_memory()
    return decode_hook


def get_release_hook(block_id):
    def release_hook(module, input, output):
        for child, module_part in zip(module.childen, module.module_part):
            child.__dict__.pop(module_part, None)
    return release_hook

def convert_model(model, model_config, is_tie_weight=True, release_hook=True):
    weights_list = {}
    modules_list = {}
    module_parts_list = {}
    for block_id in model_config.keys():
        # if 'lm_head' in model_config[block_id]['decode_hook_name'] \
        # or 'embed_tokens' in model_config[block_id]['decode_hook_name']:
        #     continue
        weights = []
        modules = []
        module_parts = []
        for weight_name in model_config[block_id]['model_names']:
            parts = weight_name.split('.')
            parent_path = '.'.join(parts[:-1])
            parent_module = model.get_submodule(parent_path)
            modules.append(parent_module)
            weights.append(getattr(parent_module, parts[-1]).data.clone().flatten())
            if parts[-1] == 'weight' or parts[-1] == 'bias':
                delattr(parent_module, parts[-1])
            else:
                assert False, f'must be weight or bias'
            module_parts.append(parts[-1])
            # module.weight = None
        weights_list[block_id]=torch.cat(weights)
        modules_list[block_id]=modules
        module_parts_list[block_id]=module_parts

    def get_split_positions(model_n_elements):
        split_positions = [model_n_elements[i] for i in range(len(model_n_elements) - 1)]
        for i  in range(1, len(split_positions)):
            split_positions[i] += split_positions[i - 1]
        return split_positions

    first_compressor = None
    for block_id in model_config.keys():
        # if 'lm_head' in model_config[block_id]['decode_hook_name'] \
        # or 'embed_tokens' in model_config[block_id]['decode_hook_name']:
        #     continue
        module_name = model_config[block_id]['decode_hook_name']
        module = model.get_submodule(module_name)
        module.register_forward_pre_hook(get_decode_hook(block_id))
        setattr(module, 'model_shapes', model_config[block_id]['model_shapes'])
        model_n_elements = model_config[block_id]['model_n_elements']
        assert len(model_config[block_id]['model_shapes']) == len(model_n_elements), f"{block_id=}"
        split_positions = get_split_positions(model_n_elements)
        setattr(module, 'split_positions', split_positions)
        compressor = CompressorBitMap(model_config[block_id]['best_indices'][-1], model_config[block_id]['nbits'])
        compressor.compress(weights_list[int(block_id)])
        setattr(module, 'compressor', compressor)
        setattr(module, 'childen', modules_list[int(block_id)])
        setattr(module, 'module_part', module_parts_list[int(block_id)])
        # release hook
        if release_hook:
            release_module_name = model_config[block_id]['release_hook_name']
            release_module = model.get_submodule(release_module_name)
            setattr(release_module, 'childen', modules_list[int(block_id)])
            setattr(release_module, 'module_part', module_parts_list[int(block_id)])
            release_module.register_forward_hook(get_release_hook(block_id))

        if block_id == 0:
            first_compressor = compressor

    # assert first_compressor is not None
    if is_tie_weight:
        module = model.lm_head
        delattr(module, 'weight')
        module.register_forward_pre_hook(get_decode_hook(0))
        setattr(module, 'model_shapes', model_config[0]['model_shapes'])
        model_n_elements = model_config[0]['model_n_elements']
        split_positions = get_split_positions(model_n_elements)
        setattr(module, 'split_positions', split_positions)
        setattr(module, 'compressor', first_compressor)
        setattr(module, 'childen', [module])
        setattr(module, 'module_part', ['weight'])
        # release hook
        if release_hook:
            module.register_forward_hook(get_release_hook(0))

    return model

def get_device_map(model, cpu_ratio=0.28, gpu_device="cuda:0"):
    total_params = sum(p.numel() for p in model.parameters())
    target_cpu = cpu_ratio * total_params
    cpu_params = 0
    device_map = {}
    named_modules = list(model.named_modules())
    named_modules.sort(key=lambda x: x[0])
    print(f"{named_modules=}")
    for name, module in named_modules:
        module_params = sum(p.numel() for p in module.parameters())
        if module_params == 0:
            continue
        if cpu_params < target_cpu:
            device_map[name] = "cpu"
            cpu_params += module_params
        else:
            device_map[name] = gpu_device

    actual_cpu_ratio = cpu_params / total_params
    print(f"the ratio of model parameters on cpu: {actual_cpu_ratio:.2%}")
    return device_map


