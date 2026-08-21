import torch
from datasets import load_dataset

import triton
import triton.language as tl
import gc

from accelerate import infer_auto_device_map, dispatch_model

dataset_root_dir='/opt/data/private/datasets'
DATASETS = {
    'c4': lambda: load_dataset('json', data_files='/opt/data/private/datasets/allenai_c4/*.json', trust_remote_code=True)['train'],
    'livecodebench': lambda: load_dataset('json', data_files='/opt/data/private/datasets/livecodebench/test5.jsonl', trust_remote_code=True)['train'],
    'mbpp': lambda: load_dataset('json', data_files='/opt/data/private/datasets/mbpp.jsonl', trust_remote_code=True)['train'],
    'aime24': lambda: load_dataset('/opt/data/private/datasets/aime24', trust_remote_code=True)['test'],
    'aime25': lambda: load_dataset('json', data_files='/opt/data/private/datasets/aime2025.jsonl', trust_remote_code=True)['train'],
    'riddlebench': lambda: load_dataset('/opt/data/private/datasets/ai4bharat/RiddleBench/data', trust_remote_code=True)['train'],
    'simplemath': lambda: load_dataset("csv", data_files='/opt/data/private/datasets/ProCreations/SimpleMath/simplemath_100k_balanced.csv', trust_remote_code=True)['train'],
}

def add_datasets_for_longbench():
    dataset_name = ['2wikimqa', 'gov_report', 'multi_news', 'musique', 'multifieldqa_en', 'narrativeqa', \
                    'passage_count', 'passage_retrieval_en', 'qasper', 'qmsum', 'hotpotqa', 'lcc', \
                    'repobench-p', 'samsum','trec', 'triviaqa']
    for dn in dataset_name:
       DATASETS[dn] = lambda dn=dn: load_dataset('/opt/data/private/jet-ai/longbench', dn)['test']

#  
def get_dataset_key(dataset_name):
    if dataset_name in ['c4', 'mbpp']:
        return 'text'
    elif dataset_name == 'livecodebench':
        return 'question_content'
    elif dataset_name in ['aime25','aime24', 'riddlebench']:
        return 'question'
    elif dataset_name in ['simplemath']:
        return 'problem'
    return 'context'

class DatasetsLoader():
    def __init__(self, dataset_name):
        add_datasets_for_longbench()
        self.dataset_name = dataset_name
        self.dataset = DATASETS[dataset_name]()

    def __getitem__(self, indices):
        if isinstance(indices, slice):
            start_idx = indices.start or 0
            end_idx = indices.stop or len(self.dataset)
        elif isinstance(indices, tuple):
            # 如果传入 (start_idx, end_idx) 元组
            start_idx, end_idx = indices
        else:
            # 如果传入单个索引
            return self._get_single_item(indices)
        dataset_key = get_dataset_key(self.dataset_name)
        return self.dataset[dataset_key][start_idx:end_idx]

    def _get_single_item(self, idx):
        """获取单个项目"""
        dataset_key = get_dataset_key(self.dataset_name)
        return self.dataset[dataset_key][idx]
    
    def __len__(self):
        """返回数据集长度"""
        return len(self.dataset)


@triton.jit
def where_kernel(
    bitmaps_ptr,        # int8/bool [N]
    out_idx_ptr,        # int32  [N]  (max size = N)
    global_counter_ptr, # int32  [1]
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    in_range = offs < N

    bits = tl.load(bitmaps_ptr + offs, mask=in_range, other=0)
    is_true = bits != 0

    # 1. block 内前缀和：把 True/False 转成 1/0 再 cumsum
    ones = is_true.to(tl.int32)
    # 如果你本地有 tl.cumsum，可以直接：
    local_scan = tl.cumsum(ones, axis=0)

    # 本 block 中 True 的总数 = 最后一个位置的 scan 值
    num_true_block = tl.max(local_scan, axis=0)

    # 2. 通过全局原子加拿到当前 block 的写入起点
    base = tl.atomic_add(global_counter_ptr, num_true_block)

    # 3. 对每个 True 元素，把它的全局位置算出来并写入
    # local_scan 是 [1, 2, 3, ...]（仅在 is_true 为 True 的位置有意义）
    # 全局位置 = base + local_scan - 1
    write_pos = base + local_scan - 1

    tl.store(
        out_idx_ptr + write_pos,
        offs,
        mask=is_true & in_range,
    )

@triton.jit
def where_and_fill_invalid_data_fused(
    original_data_ptr: tl.tensor,
    bitmaps_ptr: tl.tensor,
    invalid_exp_ptr: tl.tensor,
    global_counter_ptr: tl.tensor,
    n_elements: tl.int32,
    n_invalid: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    """
    融合的 kernel：同时完成 where(bitmaps) 和 fill_invalid_data 操作
    基于 where_kernel 的逻辑，使用 atomic_add 维护全局计数器
    确保 invalid_exp 的顺序对应关系（按位置索引顺序）
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    in_range = offs < n_elements
    
    # 加载 bitmaps
    bits = tl.load(bitmaps_ptr + offs, mask=in_range, other=0)
    is_true = bits != 0
    
    # 将 True/False 转换为 1/0 (int32)
    ones = tl.where(is_true, 1, 0).to(tl.int32)
    
    # Block 内前缀和：使用 cumsum 向量化计算
    # local_scan[i] 表示位置 i 之前（包含 i）有多少个 True
    # 使用 cumsum 替代循环，实现向量化的前缀和计算
    # 注意：如果 Triton 不支持 cumsum，请使用下方注释中的静态展开循环
    local_scan = tl.cumsum(ones)
    
    # 备选方案（如果 cumsum 不支持，取消注释下面的代码，注释掉上面的 cumsum）：
    # local_scan = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    # for i in range(BLOCK_SIZE):
    #     if i == 0:
    #         local_scan[i] = ones[i]
    #     else:
    #         local_scan[i] = local_scan[i-1] + ones[i]
    
    # 本 block 中 True 的总数 = 直接使用 sum 计算（向量化，无需循环）
    num_true_block = tl.sum(ones)
    
    # 通过全局原子加拿到当前 block 的写入起点（在 invalid_exp 中的起始索引）
    base = tl.atomic_add(global_counter_ptr, num_true_block)
    
    # 向量化计算所有位置的 invalid_exp_idx
    # local_scan - 1 是当前 block 内的排名（从 0 开始）
    # 只对 True 位置计算有效索引，非 True 位置设为 -1（无效）
    invalid_exp_idx = tl.where(is_true, base + local_scan - 1, -1)
    
    # 创建 mask：位置有效、是 True、且 invalid_exp_idx 有效
    valid_mask = in_range & is_true & (invalid_exp_idx >= 0) & (invalid_exp_idx < n_invalid)
    
    # 向量化加载对应的 invalid_exp 值（只对有效位置）
    invalid_exp = tl.load(invalid_exp_ptr + invalid_exp_idx, mask=valid_mask, other=0)
    invalid_exp_cast = tl.cast(invalid_exp, tl.uint16) << 7
    
    # 向量化加载原始数据
    original_data = tl.load(original_data_ptr + offs, mask=valid_mask, other=0)
    
    # 向量化计算填充后的数据：保留 sign 和 mantissa，替换 exponent
    filled_data = (original_data & 0x807F) | invalid_exp_cast
    
    # 向量化写回（只写回有效位置）
    tl.store(original_data_ptr + offs, filled_data, mask=valid_mask)

def get_allocated_peak_memory():
    allocated = 0
    reserved = 0
    peak_memory = 0
    for i in range(torch.cuda.device_count()):
        device = torch.device(f'cuda:{i}')
        # device_name = torch.cuda.get_device_name(device)
        allocated += torch.cuda.memory_allocated(device=device)/(1024 ** 3)
        reserved += torch.cuda.max_memory_reserved(device=device)/(1024 ** 3)
        peak_memory += torch.cuda.max_memory_allocated(device=device)/(1024 ** 3)
    return allocated, reserved, peak_memory

def reset_peak_memory():
    for i in range(torch.cuda.device_count()):
        device = torch.device(f'cuda:{i}')
        torch.cuda.memory.reset_peak_memory_stats(device)

def get_device_map_accelerate(model, cpu_ratio=0.28, gpu_device=0):
    total_params = sum(p.numel() for p in model.parameters())
    bytes_per_param = 2

    total_mem = total_params * bytes_per_param
    cpu_mem = total_mem * cpu_ratio / 1e9
    gpu_mem = total_mem / 1e9 - cpu_mem + 0.1

    max_memory = {
        gpu_device: f"{gpu_mem:.2f}GiB",
        "cpu": f"{cpu_mem:.2f}GiB"
    }

    # target_cpu_bytes = int(total_params * cpu_ratio * bytes_per_param)
    # gpu_memory = f"{int(total_params * (1 - cpu_ratio) * bytes_per_param / 1e9)}GiB"

    # max_memory = {
    #     gpu_device: gpu_memory,
    #     "cpu": f"{target_cpu_bytes / 1e9}GiB"
    # }

    device_map = infer_auto_device_map(model, max_memory=max_memory)
    return device_map


def print_trans_mem_from_pipe(pipe):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    mem_allocated = torch.cuda.memory_allocated("cuda") / (1024**3)
    print(f"{mem_allocated=}")
    del pipe.transformer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    memory_trans = mem_allocated - torch.cuda.memory_allocated("cuda")  / (1024 ** 3)
    print(f"{memory_trans=}")
