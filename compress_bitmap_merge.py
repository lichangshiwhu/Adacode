import time
from contextlib import nullcontext

import torch
import torch.profiler as profiler

import triton
import triton.language as tl


def _device_context(device):
    torch_device = torch.device(device)
    if torch_device.type == 'cuda':
        return torch.cuda.device(torch_device)
    return nullcontext()

@triton.jit
def count_invalid_blocks_from_bitmap(
    bitmaps_ptr,          # bool tensor (flatten, length N)
    block_counts_ptr,     # int32 tensor (length n_blocks)
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = offs < N

    b = tl.load(bitmaps_ptr + offs, mask=m, other=0).to(tl.int32)  # True->1 False->0
    cnt = tl.sum(b, axis=0)
    tl.store(block_counts_ptr + pid, cnt)


@triton.jit
def fill_invalid_from_bitmap_two_pass(
    x_ptr: tl.tensor,                # uint16 tensor (flatten) to be modified in-place
    invalid_exp_ptr: tl.tensor,      # uint8 tensor, length = total_invalid
    block_base_ptr: tl.tensor,       # int32 tensor, length n_blocks (exclusive prefix sum)
    invalid_sign: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    base = tl.load(block_base_ptr + pid - 1, mask=pid>0, other=0)

    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = offs < N

    x = tl.load(x_ptr + offs, mask=m, other=0)
    is_invalid = ((x & 0x7F80) == invalid_sign)
    # is_invalid = tl.load(bitmaps_ptr + offs, mask=m, other=0).to(tl.int1)

    # stable local rank within the block: 0..k-1 for invalid positions
    c = tl.cumsum(is_invalid.to(tl.int32), axis=0) - 1
    idx = base + c

    # bounds + mask
    valid = is_invalid & (idx >= 0)
    # & (idx < total_invalid)

    inv_exp_u8 = tl.load(invalid_exp_ptr + idx, mask=valid, other=0).to(tl.uint16)
    inv_exp_bits = inv_exp_u8 << 7

    # replace exponent bits, keep sign+mantissa
    filled = (x & 0x807F) | inv_exp_bits
    # out = tl.where(valid, filled, x)

    tl.store(x_ptr + offs, filled, mask=m&valid)

@triton.jit
def get_exp_kernel_constexpr(
    original_data_ptr,
    invalid_exp_ptr,
    total_size,
    BLOCK_SIZE: tl.constexpr,
):
    # left=122
    # right=129
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_size
    original_data = tl.load(original_data_ptr + offsets, mask=mask)
    
    # Extract bits 7-14
    # & 0x7F80
    extracted_values = ((original_data) >> 7) & 0xFF
    extracted_values = tl.cast(extracted_values, tl.uint8)
    # extracted_values = (extracted_values - left) & 0xF8
    # invalid_condition = (extracted_values < left) | (extracted_values > right)
    # extracted_values = tl.where(invalid_condition, extracted_values, left)
    
    tl.store(invalid_exp_ptr + offsets, extracted_values, mask=mask)

@triton.jit
def get_exp_mask_kernel_constexpr(
    original_data_ptr,
    invalid_exp_ptr,
    mask_ptr,
    left,
    right,
    total_size,
    BLOCK_SIZE: tl.constexpr,
):
    # left=122
    # right=129
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_size
    original_data = tl.load(original_data_ptr + offsets, mask=mask)
    
    # Extract bits 7-14
    # & 0x7F80
    extracted_values = ((original_data) >> 7) & 0xFF
    extracted_values = tl.cast(extracted_values, tl.int32)
    tl.store(invalid_exp_ptr + offsets, extracted_values.to(tl.uint8), mask=mask)
    meta_mask = ((extracted_values <= left) | (extracted_values > right))
    tl.store(mask_ptr + offsets, meta_mask, mask=mask)
 
@triton.jit
def get_invalid_mask_kernel_constexpr(
    original_data_ptr,
    invalid_mask_ptr,
    total_size,
    BLOCK_SIZE: tl.constexpr,
):    
    left=122
    right=129
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < total_size
    
    original_data = tl.load(original_data_ptr + offsets, mask=mask)
    
    # Extract bits 7-14
    extracted_values = ((original_data & 0x7F80) >> 7).to(tl.int32)

    invalid_mask = ((extracted_values < left) | (extracted_values > right)).to(tl.uint8)
    
    tl.store(invalid_mask_ptr + offsets, invalid_mask, mask=mask)

@triton.jit
def get_invalid_exp(invalid_data_ptr, invalid_exp_ptr, invalid_size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < invalid_size
    invalid_data = tl.load(invalid_data_ptr+offsets, mask=mask)
    extracted_values = (invalid_data & 0x7F80) >> 7
    extracted_values = tl.cast(extracted_values, tl.uint8)
    tl.store(invalid_exp_ptr + offsets, extracted_values, mask=mask)

# @triton.jit
# def get_exp_mask_count_fused_kernel(
#     data_ptr,             # uint16 input
#     extracted_ptr,        # uint8  output: 8-bit exponent
#     block_counts_ptr,     # int32  output: invalid count per decomp-block
#     left:  tl.constexpr,
#     right: tl.constexpr,
#     n_elements: tl.int32,
#     BLOCK_SIZE: tl.constexpr,   # 与解压 kernel 的 BLOCK_SIZE * 8 对齐
# ):
#     pid = tl.program_id(axis=0)
#     offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
#     mask = offsets < n_elements

#     data    = tl.load(data_ptr + offsets, mask=mask, other=0).to(tl.uint16)
#     exp     = ((data >> 7) & 0xFF).to(tl.uint8)                  # 8-bit 指数
#     is_inv  = mask & ((exp <= left) | (exp > right))               # 范围外 = 无效

#     tl.store(extracted_ptr   + offsets, exp,    mask=mask)

#     # 当前 decomp-block 内的无效计数，直接写入 block_counts
#     count = tl.sum(is_inv.to(tl.int32))
#     tl.store(block_counts_ptr + pid, count)

@triton.jit
def get_exp_mask_count_fused_kernel(
    data_ptr,             # uint16 input
    extracted_ptr,        # uint8  output: 8-bit exponent
    block_counts_ptr,     # int32  output: invalid count per decomp-block
    hist_ptr,             # int32  output: 256-bin histogram (only written when need_adjust)
    left:  tl.constexpr,
    right: tl.constexpr,
    n_elements: tl.int32,
    need_adjust: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,   # 与解压 kernel 的 BLOCK_SIZE * 8 对齐
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    data    = tl.load(data_ptr + offsets, mask=mask, other=0).to(tl.uint16)
    exp     = ((data >> 7) & 0xFF).to(tl.int32)                  # 8-bit 指数
    is_inv  = mask & ((exp <= left) | (exp > right))               # 范围外 = 无效

    tl.store(extracted_ptr   + offsets, exp.to(tl.uint8),    mask=mask)

    # 当前 decomp-block 内的无效计数，直接写入 block_counts
    count = tl.sum(is_inv.to(tl.int32))
    tl.store(block_counts_ptr + pid, count)

    # 将指数分布统计写入 256-bin histogram（原子累加）
    if need_adjust:
        exp_int = exp.to(tl.int32)
        tl.atomic_add(hist_ptr + exp_int, 1, mask=mask)

def get_invalid_index_values(
    original_data_bfloat16,
    left: int = 122,
    right: int = 129,
    decomp_block_size: int = 128,   # 对应解压 kernel 的 BLOCK_SIZE
    need_adjust: bool = False,
):
    """
    一次 kernel 同时产出压缩/解压所需的全部参数。

    返回：
        invalid_exp   : Tensor[uint8]  — 越界元素的原始指数（紧凑存储）
        invalid_mask  : Tensor[bool]   — 全量无效标记（压缩时 scatter 使用）
        block_base    : Tensor[int32]  — 每个 decomp-block 的无效偏移前缀和
        extracted_values (可选)        — need_adjust=True 时返回完整指数张量
    """
    flat   = original_data_bfloat16.view(-1)
    device = flat.device
    N      = flat.numel()

    # 每个 decomp-block 覆盖 decomp_block_size * 8 个 bf16 元素
    n_blocks   = triton.cdiv(N, decomp_block_size)

    extracted_values = torch.empty(N,        dtype=torch.uint8,  device=device)
    block_counts     = torch.empty(n_blocks, dtype=torch.int32,  device=device)
    hist = torch.zeros(256, dtype=torch.int32, device=device) if need_adjust else None
    hist_ptr = hist if need_adjust else extracted_values  # 占位，kernel 内 constexpr 分支不会访问

    with _device_context(device):
        get_exp_mask_count_fused_kernel[(n_blocks,)](
            flat.view(torch.uint16),
            extracted_values,
            block_counts,
            hist_ptr,
            left, right, N,
            need_adjust=need_adjust,
            BLOCK_SIZE=decomp_block_size,
        )

    invalid_exp = extracted_values[(extracted_values <= left) | (extracted_values > right)]
    block_base  = torch.cumsum(block_counts, dim=0)                    # 解压索引基址

    if need_adjust:
        return invalid_exp, block_base, hist
    return invalid_exp, block_base, None

@triton.jit
def compress3bit_vector(
    original_data: tl.tensor,
    exp_data: tl.tensor,
    sign_mass_data: tl.tensor,
    n_elements: tl.int32,
    n_exp_data: tl.int32,
    left: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    # ── 一次性加载 8×BLOCK_SIZE 个元素（原来是 8 次单独 load）──
    start_idx = pid * BLOCK_SIZE * 8
    flat_offs = start_idx + tl.arange(0, BLOCK_SIZE * 8)   # shape: (BLOCK_SIZE*8,)
    mask_valid = flat_offs < n_elements

    elements = tl.load(
        original_data + flat_offs, mask=mask_valid, other=0
    ).to(tl.uint32)   # uint32 保证移位不溢出

    # ── sign_mass：一次性向量化提取并存储（原来是 8 次）──
    # bf16 layout: [sign(15) | exp(14:7) | mant(6:0)]
    # sign_mass:   [sign(7)  | mant(6:0)]
    sign_mass_values = tl.cast(
        ((elements >> 8) & 0x80) | (elements & 0x7F),
        tl.uint8
    )
    tl.store(sign_mass_data + flat_offs, sign_mass_values, mask=mask_valid)

    # ── 提取 3-bit 压缩指数，越界时置 0（无效哨兵）──
    exp_bits = ((elements >> 7) & 0xFF).to(tl.int32)
    compressed_bits = exp_bits - tl.full(exp_bits.shape, left, tl.int32)
    compressed_bits = tl.where(
        (compressed_bits & 0xFFF8) == 0,
        compressed_bits & 0x07,
        0
    ).to(tl.uint32)

    # ── 向量化打包：对应解压的 byte_idx = 21 - (lane%8)*3 ──
    # 每个元素在 24-bit packed 中的 shift 量：21,18,15,12,9,6,3,0
    lane  = tl.arange(0, BLOCK_SIZE * 8) % 8          # shape: (BLOCK_SIZE*8,)
    shift = 21 - lane * 3                              # 与解压 byte_idx 完全对称
    shifted = compressed_bits << shift                 # 各就各位

    # ── 每 8 个元素 OR 合并 → packed24，shape: (BLOCK_SIZE,) ──
    # tl.sum 在 uint32 上等价于 bitwise-OR（各位不重叠）
    packed24 = tl.sum(
        tl.reshape(shifted, (BLOCK_SIZE, 8)), axis=1
    )                                                  # shape: (BLOCK_SIZE,)

    # ── 拆成 3 字节存储 ──
    b0 = tl.cast((packed24 >> 16) & 0xFF, tl.uint8)
    b1 = tl.cast((packed24 >>  8) & 0xFF, tl.uint8)
    b2 = tl.cast( packed24        & 0xFF, tl.uint8)

    exp_offsets = pid * BLOCK_SIZE * 3 + tl.arange(0, BLOCK_SIZE) * 3
    tl.store(exp_data + exp_offsets,     b0, mask=exp_offsets     < n_exp_data)
    tl.store(exp_data + exp_offsets + 1, b1, mask=exp_offsets + 1 < n_exp_data)
    tl.store(exp_data + exp_offsets + 2, b2, mask=exp_offsets + 2 < n_exp_data)

@triton.jit
def compress1bit_vector(
    original_data: tl.tensor,
    exp_data: tl.tensor,
    sign_mass_data: tl.tensor,
    n_elements: tl.int32,
    n_exp_data: tl.int32,
    left: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    start_idx = pid * BLOCK_SIZE * 8
    flat_offs = start_idx + tl.arange(0, BLOCK_SIZE * 8)
    mask_valid = flat_offs < n_elements

    elements = tl.load(original_data + flat_offs, mask=mask_valid, other=0).to(tl.uint32)
    sign_mass_values = tl.cast(((elements >> 8) & 0x80) | (elements & 0x7F), tl.uint8)
    tl.store(sign_mass_data + flat_offs, sign_mass_values, mask=mask_valid)

    exp_bits = ((elements >> 7) & 0xFF).to(tl.int32)
    compressed_bits = exp_bits - tl.full(exp_bits.shape, left, tl.int32)
    compressed_bits = tl.where(
        (compressed_bits & 0xFFFE) == 0,
        compressed_bits & 0x01,
        0,
    ).to(tl.uint32)

    lane = tl.arange(0, BLOCK_SIZE * 8) % 8
    shift = 7 - lane
    shifted = compressed_bits << shift
    all_byte0 = tl.sum(tl.reshape(shifted, (BLOCK_SIZE, 8)), axis=1)

    exp_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(exp_data + exp_offsets, tl.cast(all_byte0, tl.uint8), mask=exp_offsets<n_exp_data)

@triton.jit
def compress2bit_vector(
    original_data: tl.tensor,
    exp_data: tl.tensor,
    sign_mass_data: tl.tensor,
    n_elements: tl.int32,
    n_exp_data: tl.int32,
    left: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    start_idx = pid * BLOCK_SIZE * 4
    flat_offs = start_idx + tl.arange(0, BLOCK_SIZE * 4)
    mask_valid = flat_offs < n_elements

    elements = tl.load(original_data + flat_offs, mask=mask_valid, other=0).to(tl.uint32)
    sign_mass_values = tl.cast(((elements >> 8) & 0x80) | (elements & 0x7F), tl.uint8)
    tl.store(sign_mass_data + flat_offs, sign_mass_values, mask=mask_valid)

    exp_bits = ((elements >> 7) & 0xFF).to(tl.int32)
    compressed_bits = exp_bits - tl.full(exp_bits.shape, left, tl.int32)
    compressed_bits = tl.where(
        (compressed_bits & 0xFFFC) == 0,
        compressed_bits & 0x03,
        0,
    ).to(tl.uint32)

    lane = tl.arange(0, BLOCK_SIZE * 4) % 4
    shift = (3 - lane) * 2
    shifted = compressed_bits << shift
    all_byte0 = tl.sum(tl.reshape(shifted, (BLOCK_SIZE, 4)), axis=1)

    exp_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(exp_data + exp_offsets, tl.cast(all_byte0, tl.uint8), mask=exp_offsets<n_exp_data)

@triton.jit
def compress4bit_vector(
    original_data: tl.tensor,
    exp_data: tl.tensor,
    sign_mass_data: tl.tensor,
    n_elements: tl.int32,
    n_exp_data: tl.int32,
    left: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    start_idx = pid * BLOCK_SIZE * 2
    flat_offs = start_idx + tl.arange(0, BLOCK_SIZE * 2)
    mask_valid = flat_offs < n_elements

    elements = tl.load(original_data + flat_offs, mask=mask_valid, other=0).to(tl.uint32)
    sign_mass_values = tl.cast(((elements >> 8) & 0x80) | (elements & 0x7F), tl.uint8)
    tl.store(sign_mass_data + flat_offs, sign_mass_values, mask=mask_valid)

    exp_bits = ((elements >> 7) & 0xFF).to(tl.int32)
    compressed_bits = exp_bits - tl.full(exp_bits.shape, left, tl.int32)
    compressed_bits = tl.where(
        (compressed_bits & 0xFFF0) == 0,
        compressed_bits & 0x0F,
        0,
    ).to(tl.uint32)

    lane = tl.arange(0, BLOCK_SIZE * 2) % 2
    shift = (1 - lane) * 4
    shifted = compressed_bits << shift
    all_byte0 = tl.sum(tl.reshape(shifted, (BLOCK_SIZE, 2)), axis=1)

    exp_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(exp_data + exp_offsets, tl.cast(all_byte0, tl.uint8), mask=exp_offsets<n_exp_data)

# A:1010 B:1011 C:1100 D:1101 E:1110 F:1111
@triton.jit
def uncompress1bit_vector_merge(
    original_data: tl.tensor,  #  uint16
    invalid_exp_ptr: tl.tensor,
    block_base_ptr: tl.tensor,
    exp_data: tl.tensor,       #  uint8
    sign_mass_data: tl.tensor, #  uint8
    n_elements: tl.int32,
    left: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    start_idx = pid * BLOCK_SIZE * 8

    exp_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    exp_values = tl.load(exp_data + exp_offsets)

    out_idx = tl.arange(0, BLOCK_SIZE * 8)
    lane_idx = out_idx // 8
    byte_idx = 7 - (out_idx % 8)
    e1 = ((tl.gather(exp_values, lane_idx, axis=0) >> byte_idx) & 0x01).cast(tl.uint8)

    is_invalid = (e1 == 0)
    c = tl.cumsum(is_invalid.to(tl.int32), axis=0) - 1
    base = tl.load(block_base_ptr + pid - 1, mask=pid > 0, other=0)
    idx = base + c

    flat_offs = start_idx + out_idx
    mask_valid = flat_offs < n_elements
    invalid_flag = is_invalid & (idx >= 0) & mask_valid
    inv_exp_u8 = tl.load(invalid_exp_ptr + idx, mask=invalid_flag, other=0)
    sign_mass = tl.load(sign_mass_data + flat_offs, mask=mask_valid, other=0).to(tl.uint16)
    sign_mass_bf16 = ((sign_mass & tl.constexpr(0x0080)) << 8) | (sign_mass & tl.constexpr(0x007F))
    org_exp_i32 = tl.where(invalid_flag, inv_exp_u8.to(tl.int32), e1.to(tl.int32) + left)
    out_val = sign_mass_bf16 | (org_exp_i32.to(tl.uint16) << 7)
    tl.store(original_data + flat_offs, out_val, mask=mask_valid)

# A:1010 B:1011 C:1100 D:1101 E:1110 F:1111
# 206
@triton.jit
def uncompress3bit_vector_merge(
    original_data: tl.tensor,  #  uint16
    invalid_exp_ptr: tl.tensor, 
    block_base_ptr: tl.tensor,
    exp_data: tl.tensor,       #  uint8
    sign_mass_data: tl.tensor, #  uint8
    n_elements: tl.int32,      #
    left: tl.constexpr,      #
    BLOCK_SIZE: tl.constexpr,  #
):
    pid = tl.program_id(axis=0)
    # packed the 3 Bytes
    exp_start_idx = pid * BLOCK_SIZE * 3
    exp_offsets = exp_start_idx + tl.arange(0, BLOCK_SIZE) * 3
    b0 = tl.load(exp_data + exp_offsets).to(tl.uint32)  # 
    b1 = tl.load(exp_data + exp_offsets + 1).to(tl.uint32)  # 
    b2 = tl.load(exp_data + exp_offsets + 2).to(tl.uint32)  # 
    packed24 = (b0 << 16) | (b1 << 8) | b2

    # right blocks
    out_idx = tl.arange(0, BLOCK_SIZE * 8)
    lane_idx = out_idx // 8
    byte_idx = 21 - (out_idx % 8) * 3
    e8 = ((tl.gather(packed24, lane_idx, axis=0) >> byte_idx) & 0x07).cast(tl.uint8)

    is_invalid = (e8 == 0)
    c = tl.cumsum(is_invalid.to(tl.int32), axis=0) - 1
    base = tl.load(block_base_ptr + pid - 1, mask=pid>0, other=0)
    idx = base + c

    start_idx  = pid * BLOCK_SIZE * 8
    flat_offs  = start_idx + out_idx
    mask_valid = flat_offs < n_elements
    invalid_flag = is_invalid & (idx >= 0) & mask_valid

    inv_exp_u8 = tl.load(invalid_exp_ptr + idx, mask=invalid_flag, other=0).to(tl.uint16)

    sign_mass = tl.load(
        sign_mass_data + flat_offs,
        mask=mask_valid,
        other=0
    ).to(tl.uint16)

    sign_mass_bf16 = ((sign_mass & tl.constexpr(0x0080)) << 8) | (sign_mass & tl.constexpr(0x007F))
    e8_i32 = e8.to(tl.int32)
    org_exp_i32 = tl.where(invalid_flag, inv_exp_u8.to(tl.int32), e8_i32 + left)
    org_exp = org_exp_i32.to(tl.uint16)
    out_val = sign_mass_bf16 | (org_exp << 7)
    tl.store(original_data + flat_offs, out_val, mask=mask_valid)

@triton.jit
def uncompress2bit_vector_merge(
    original_data: tl.tensor,  #  uint16
    invalid_exp_ptr: tl.tensor, 
    block_base_ptr: tl.tensor,
    exp_data: tl.tensor,       #  uint8
    sign_mass_data: tl.tensor, #  uint8
    n_elements: tl.int32,      # 
    left: tl.constexpr,      # 
    BLOCK_SIZE: tl.constexpr,  # 
):
    pid = tl.program_id(axis=0)
    start_idx = pid * BLOCK_SIZE * 4

    exp_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    exp_values = tl.load(exp_data + exp_offsets)

    out_idx = tl.arange(0, BLOCK_SIZE * 4)
    lane_idx = out_idx // 4
    byte_idx = (3 - (out_idx % 4)) * 2
    # 1111
    e2 = ((tl.gather(exp_values, lane_idx, axis=0) >> byte_idx) & 0x03).cast(tl.uint8)

    is_invalid = (e2 == 0)
    c = tl.cumsum(is_invalid.to(tl.int32), axis=0) - 1
    base = tl.load(block_base_ptr + pid - 1, mask=pid>0, other=0)
    idx = base + c

    flat_offs = start_idx + out_idx
    mask_valid = flat_offs < n_elements
    invalid_flag = is_invalid & (idx >= 0) & mask_valid
    inv_exp_u8 = tl.load(invalid_exp_ptr + idx, mask=invalid_flag, other=0)

    sign_mass = tl.load(
        sign_mass_data + flat_offs,
        mask=mask_valid,
        other=0
    ).to(tl.uint16)
    sign_mass_bf16 = ((sign_mass & tl.constexpr(0x0080)) << 8) | (sign_mass & tl.constexpr(0x007F))
    org_exp_i32 = tl.where(invalid_flag, inv_exp_u8.to(tl.int32), e2.to(tl.int32) + left)
    out_val = sign_mass_bf16 | (org_exp_i32.to(tl.uint16) << 7)
    tl.store(original_data + flat_offs, out_val, mask=mask_valid)

@triton.jit
def uncompress4bit_vector_merge(
    original_data: tl.tensor,  #  uint16
    invalid_exp_ptr: tl.tensor, 
    block_base_ptr: tl.tensor,
    exp_data: tl.tensor,       #  uint8
    sign_mass_data: tl.tensor, #  uint8
    n_elements: tl.int32,      # 
    left: tl.constexpr,      # 
    BLOCK_SIZE: tl.constexpr,  # 
):
    pid = tl.program_id(axis=0)
    start_idx = pid * BLOCK_SIZE * 2

    exp_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    exp_values = tl.load(exp_data + exp_offsets)

    out_idx = tl.arange(0, BLOCK_SIZE * 2)
    lane_idx = out_idx // 2
    byte_idx = (1 - (out_idx % 2)) * 4
    # 1111
    e2 = ((tl.gather(exp_values, lane_idx, axis=0) >> byte_idx) & 0x0F).cast(tl.uint8)

    is_invalid = (e2 == 0)
    c = tl.cumsum(is_invalid.to(tl.int32), axis=0) - 1
    base = tl.load(block_base_ptr + pid - 1, mask=pid>0, other=0)
    idx = base + c

    flat_offs = start_idx + out_idx
    mask_valid = flat_offs < n_elements
    invalid_flag = is_invalid & (idx >= 0) & mask_valid
    inv_exp_u8 = tl.load(invalid_exp_ptr + idx, mask=invalid_flag, other=0)

    sign_mass = tl.load(
        sign_mass_data + flat_offs,
        mask=mask_valid,
        other=0
    ).to(tl.uint16)
    sign_mass_bf16 = ((sign_mass & tl.constexpr(0x0080)) << 8) | (sign_mass & tl.constexpr(0x007F))
    org_exp_i32 = tl.where(invalid_flag, inv_exp_u8.to(tl.int32), e2.to(tl.int32) + left)
    out_val = sign_mass_bf16 | (org_exp_i32.to(tl.uint16) << 7)
    tl.store(original_data + flat_offs, out_val, mask=mask_valid)

@triton.jit
def fill_invalid_data(original_data_ptr: tl.tensor, 
                      n_elements: tl.int32, 
                      invalid_indices_ptr: tl.tensor, 
                      invalid_exp_ptr: tl.tensor, 
                      n_invalid_size: tl.int32, 
                      BLOCK_SIZE:tl.constexpr):
    pid = tl.program_id(axis=0)
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_invalid_size
    invalid_indices = tl.load(invalid_indices_ptr + offsets, mask=mask)
    invalid_exp = tl.load(invalid_exp_ptr + offsets, mask=mask)
    invalid_exp_cast = tl.cast(invalid_exp, tl.uint16) << 7

    valid_mask = mask & (invalid_indices < n_elements)
    original_data = tl.load(original_data_ptr+invalid_indices, mask=valid_mask)
    filled_data = (original_data&0x807F) | invalid_exp_cast

    tl.store(original_data_ptr+invalid_indices, filled_data, mask=valid_mask)

@triton.jit
def get_bitmaps(x_ptr: tl.tensor, 
            bitmaps: tl.tensor,
            left_nbits: tl.constexpr,
            N: tl.constexpr,
            BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    bitmap = tl.where(x&0x7F80 == left_nbits, 1, 0)
    tl.store(bitmaps + offs, bitmap, mask=mask)

class CompressorBitMap:
    def __init__(self, right=129, nbits=3):
        self.device = None
        self.original_shape = None
        self.n_elements = None
        self.n_exp_data = None
        self.sign_mass_data = None
        self.exp_data = None
        self.invalid_exp = None
        self.blocksize = 128
        # 64 -> 202
        # 128 -> 203
        # (left, right] is compressed
        # 0,000 0000 0,000 0000
        # 0x7F80
        self.left = right - 2**nbits + 1
        self.right = right
        self.nbits = nbits
        self._configure_runtime()
        self.block_base = None

    def _configure_runtime(self):
        uncompress_lists = [self.uncompress1bit, self.uncompress2bit, self.uncompress3bit, self.uncompress4bit]
        block_gaps = [8, 4, 8, 2]
        self.invalid_block_size = self.blocksize * block_gaps[self.nbits - 1]
        self.target_uncompressor = uncompress_lists[self.nbits - 1]
        self.left_nbits = self.left << 7

    def reset_memory(self):
        self.device = None
        self.n_elements = None
        self.n_exp_data = None
        
        self.original_shape = None
        self.sign_mass_data = None
        self.exp_data = None
        self.invalid_exp = None
        self.block_base = None

    def _update_compression_params(self, right, nbits):
        self.left = right - 2**nbits + 1
        self.right = right
        self.nbits = nbits
        self._configure_runtime()
        self.block_base = None

    def get_memory(self):
        if self.sign_mass_data is None:
            return {'compressed_memory': 0, 'original_memory': 0}
        m1 = self.sign_mass_data.numel() * self.sign_mass_data.element_size()
        m2 = self.exp_data.numel() * self.exp_data.element_size()
        m3 = 0 if self.invalid_exp is None else self.invalid_exp.numel() * self.invalid_exp.element_size()
        m4 = 0 if self.block_base is None else self.block_base.numel() * self.block_base.element_size()
        compressed_memory = m1 + m2 + m3 + m4
        original_m = 0 if self.n_elements is None else self.n_elements * 2
        ratio = 0 if original_m == 0 else compressed_memory / original_m
        return {'compressed_memory': compressed_memory, 'original_memory': original_m, 'ratio': ratio}

    def export_state(self):
        return {
            'right': self.right,
            'nbits': self.nbits,
            'left': self.left,
            'left_nbits': self.left_nbits,
            'blocksize': self.blocksize,
            'invalid_block_size': self.invalid_block_size,
            'device': self.device,
            'original_shape': self.original_shape,
            'n_elements': self.n_elements,
            'n_exp_data': self.n_exp_data,
            'sign_mass_data': self.sign_mass_data,
            'exp_data': self.exp_data,
            'invalid_exp': self.invalid_exp,
            'block_base': self.block_base,
        }

    def export_decode_view(self):
        return {
            'sign_mass_data': self.sign_mass_data,
            'exp_data': self.exp_data,
            'invalid_exp': self.invalid_exp,
            'block_base': self.block_base,
            'left': self.left,
            'right': self.right,
            'left_nbits': self.left_nbits,
            'nbits': self.nbits,
            'blocksize': self.blocksize,
            'invalid_block_size': self.invalid_block_size,
            'original_shape': self.original_shape,
            'n_elements': self.n_elements,
            'n_exp_data': self.n_exp_data,
            'device': self.device,
        }

    @staticmethod
    def export_decode_view_from_state(state):
        return {
            'sign_mass_data': state['sign_mass_data'],
            'exp_data': state['exp_data'],
            'invalid_exp': state['invalid_exp'],
            'block_base': state['block_base'],
            'left': state['left'],
            'right': state['right'],
            'left_nbits': state['left_nbits'],
            'nbits': state['nbits'],
            'blocksize': state['blocksize'],
            'invalid_block_size': state['invalid_block_size'],
            'original_shape': state['original_shape'],
            'n_elements': state['n_elements'],
            'n_exp_data': state['n_exp_data'],
            'device': state['device'],
        }

    @classmethod
    def from_state(cls, state):
        compressor = cls(right=state['right'], nbits=state['nbits'])
        compressor.device = state['device']
        compressor.original_shape = state['original_shape']
        compressor.n_elements = state['n_elements']
        compressor.n_exp_data = state['n_exp_data']
        compressor.sign_mass_data = state['sign_mass_data']
        compressor.exp_data = state['exp_data']
        compressor.invalid_exp = state['invalid_exp']
        compressor.block_base = state['block_base']
        return compressor

    @classmethod
    def compress_block(cls, original_data_bfloat16, right, nbits, need_adjust=False):
        compressor = cls(right=right, nbits=nbits)
        extracted_hist = compressor.compress(original_data_bfloat16, need_adjust=need_adjust)
        return compressor, extracted_hist

    def materialize(self):
        return self.uncompress()

    def compress(self, original_data_bfloat16, need_adjust=False):
        # torch.cuda.synchronize()
        
        self.device = original_data_bfloat16.device
        self.original_shape = original_data_bfloat16.shape
        self.n_elements = original_data_bfloat16.numel()

        self.invalid_exp, self.block_base, extracted_hist = \
            get_invalid_index_values(
                original_data_bfloat16,
                left=self.left,
                right=self.right,
                decomp_block_size=self.invalid_block_size,
                need_adjust=need_adjust,
            )

        if self.nbits == 3:
            assert self.n_elements % 8 == 0
            n_threads1 = self.n_elements // 8
            self.n_exp_data = n_threads1 * 3
            self.sign_mass_data = torch.zeros(self.n_elements, dtype=torch.uint8, device=self.device)
            self.exp_data = torch.zeros(self.n_exp_data, dtype=torch.uint8, device=self.device)
            grid1 = lambda meta: (triton.cdiv(n_threads1, meta['BLOCK_SIZE']), )
            
            with _device_context(self.device):
                compress3bit_vector[grid1](original_data_bfloat16.view(torch.uint16), self.exp_data, \
                                                self.sign_mass_data, self.n_elements, self.n_exp_data, self.left, self.blocksize)
        elif self.nbits == 1:
            n_threads1 = self.n_elements // 8
            self.n_exp_data = n_threads1
            self.sign_mass_data = torch.zeros(self.n_elements, dtype=torch.uint8, device=self.device)
            self.exp_data = torch.zeros(self.n_exp_data, dtype=torch.uint8, device=self.device)
            grid1 = lambda meta: (triton.cdiv(n_threads1, meta['BLOCK_SIZE']), )
            with _device_context(self.device):
                compress1bit_vector[grid1](original_data_bfloat16.view(torch.uint16), self.exp_data, \
                                                self.sign_mass_data, self.n_elements, self.n_exp_data, self.left, self.blocksize)
        elif self.nbits == 2:
            n_threads1 = self.n_elements // 4
            self.n_exp_data = n_threads1
            self.sign_mass_data = torch.zeros(self.n_elements, dtype=torch.uint8, device=self.device)
            self.exp_data = torch.zeros(self.n_exp_data, dtype=torch.uint8, device=self.device)
            grid1 = lambda meta: (triton.cdiv(n_threads1, meta['BLOCK_SIZE']), )
            with _device_context(self.device):
                compress2bit_vector[grid1](original_data_bfloat16.view(torch.uint16), self.exp_data, \
                                                self.sign_mass_data, self.n_elements, self.n_exp_data, self.left, self.blocksize)
        elif self.nbits == 4:
            n_threads1 = self.n_elements // 2
            self.n_exp_data = n_threads1
            self.sign_mass_data = torch.zeros(self.n_elements, dtype=torch.uint8, device=self.device)
            self.exp_data = torch.zeros(self.n_exp_data, dtype=torch.uint8, device=self.device)
            grid1 = lambda meta: (triton.cdiv(n_threads1, meta['BLOCK_SIZE']), )
            with _device_context(self.device):
                compress4bit_vector[grid1](original_data_bfloat16.view(torch.uint16), self.exp_data, \
                                                self.sign_mass_data, self.n_elements, self.n_exp_data, self.left, self.blocksize)
        return extracted_hist

    def uncompress1bit(self, uncompressed_odb):
        grid1 = lambda meta: (triton.cdiv((self.n_elements // 8), meta['BLOCK_SIZE']), )
        with _device_context(self.device):
            uncompress1bit_vector_merge[grid1](uncompressed_odb, self.invalid_exp, self.block_base, self.exp_data, \
                                                self.sign_mass_data, self.n_elements, self.left, self.blocksize)

    def uncompress2bit(self, uncompressed_odb):
        grid1 = lambda meta: (triton.cdiv((self.n_elements // 4), meta['BLOCK_SIZE']), )
        with _device_context(self.device):
            uncompress2bit_vector_merge[grid1](uncompressed_odb, self.invalid_exp, self.block_base, self.exp_data, \
                                                self.sign_mass_data, self.n_elements, self.left, self.blocksize)

    def uncompress3bit(self, uncompressed_odb):
        grid1 = lambda meta: (triton.cdiv(self.n_elements, meta['BLOCK_SIZE']*8), )
        with _device_context(self.device):
            uncompress3bit_vector_merge[grid1](uncompressed_odb, self.invalid_exp, self.block_base, self.exp_data, \
                                                self.sign_mass_data, self.n_elements, self.left, self.blocksize)

    def uncompress4bit(self, uncompressed_odb):
        grid1 = lambda meta: (triton.cdiv((self.n_elements // 2), meta['BLOCK_SIZE']), )
        with _device_context(self.device):
            uncompress4bit_vector_merge[grid1](uncompressed_odb, self.invalid_exp, self.block_base, self.exp_data, \
                                                self.sign_mass_data, self.n_elements, self.left, self.blocksize)
    def update_compress(self, right, nbits):
        uncompressed_odb = torch.empty(self.n_elements, dtype=torch.bfloat16, device=self.device)
        original_shape = self.original_shape
        self.target_uncompressor(uncompressed_odb.view(torch.uint16))

        self.device = None
        self.n_elements = None
        self.n_exp_data = None

        self.sign_mass_data = None
        self.exp_data = None
        self.invalid_exp = None
        self.block_base = None
        self.original_shape = None

        self._update_compression_params(right, nbits)

        return uncompressed_odb.view(original_shape)

    def uncompress(self):
        uncompressed_odb = torch.empty(self.n_elements, dtype=torch.bfloat16, device=self.device)
        self.target_uncompressor(uncompressed_odb.view(torch.uint16))

        return uncompressed_odb.view(self.original_shape)

    def uncompress_with_weights(self, uncompressed_odb):
        self.target_uncompressor(uncompressed_odb)

if __name__ == "__main__":
    import os, random
    import numpy as np
    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  #
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    B,H,S,D=8,8,8,8
    set_seed(0)
    blocksize=1024
    # 1000
    original_data_bfloat16 = torch.randn((B,H,S,D), dtype=torch.bfloat16).to('cuda:0') - 0.5
    # compress
    cp = CompressorBitMap(right=127, nbits=2)
    cp.compress(original_data_bfloat16)
    # uncompressed_odb=torch.zeros_like(original_data_bfloat16)
    uncompressed_odb = cp.uncompress()

    sub_res = original_data_bfloat16 - uncompressed_odb
    print("uncompress:", torch.sum(uncompressed_odb-original_data_bfloat16))
 
