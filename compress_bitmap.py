import time

import torch
import torch.profiler as profiler

import triton
import triton.language as tl

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
    x_ptr,                # uint16 tensor (flatten) to be modified in-place
    bitmaps_ptr,          # bool tensor (flatten)
    invalid_exp_ptr,      # uint8 tensor, length = total_invalid
    block_base_ptr,       # int32 tensor, length n_blocks (exclusive prefix sum)
    # total_invalid: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    base = tl.load(block_base_ptr + pid - 1, mask=pid>0, other=0).to(tl.int32)

    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = offs < N

    x = tl.load(x_ptr + offs, mask=m, other=0).to(tl.uint16)
    is_invalid = tl.load(bitmaps_ptr + offs, mask=m, other=0).to(tl.int1)

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
    out = tl.where(valid, filled, x)

    tl.store(x_ptr + offs, out, mask=m)

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
    extracted_values = tl.cast(extracted_values, tl.uint8)
    tl.store(invalid_exp_ptr + offsets, extracted_values, mask=mask)
    meta_mask = ((extracted_values <= left )| (extracted_values > right))
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
    extracted_values = (original_data & 0x7F80) >> 7
    
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

# (left, right] is valid
def get_invalid_index_values(original_data_bfloat16, left=122, right=129, need_adjust=False):
    # torch.cuda.synchronize()
    # get_exp_kernel_time = time.time()
    BLOCK_SIZE=1024
    flatten_original_data_bfloat16 = original_data_bfloat16.view(-1)
    device = flatten_original_data_bfloat16.device
    total_size = flatten_original_data_bfloat16.numel()

    extracted_values = torch.zeros(total_size, dtype=torch.uint8, device=device)
    invalid_mask = torch.zeros(total_size, dtype=torch.bool, device=device)
    grid1 = lambda meta: (triton.cdiv(total_size, meta['BLOCK_SIZE']), )
    get_exp_mask_kernel_constexpr[grid1](flatten_original_data_bfloat16.view(torch.uint16), extracted_values, \
                                            invalid_mask, left, right, total_size, BLOCK_SIZE=BLOCK_SIZE)
    invalid_exp = torch.masked_select(extracted_values, invalid_mask)
    if need_adjust:
        return invalid_exp, extracted_values
    # get_exp_kernel_constexpr[grid1](flatten_original_data_bfloat16.view(torch.uint16), extracted_values, total_size, BLOCK_SIZE=BLOCK_SIZE)
    return invalid_exp, None

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
    start_idx = pid * BLOCK_SIZE * 8
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)*8
    # mask = offsets < n_elements
    org_group_ptr = original_data + offsets
    sign_mass_group_ptr = sign_mass_data + offsets
    #####
    elements = tl.load(org_group_ptr, mask=offsets<n_elements)
    sign_mass_values = tl.cast(
        ((elements >> 8) & 0x80) | (elements & 0x7F), 
        tl.uint8
    )
    tl.store(sign_mass_group_ptr, sign_mass_values, mask=offsets<n_elements)
    compressed_bits = (elements >> 7) - left
    # 0,000 0000 0,000 0000
    # 0000 000,0, 0000 0000
    # 0111 1100
    compressed_bits = tl.where(compressed_bits&0x00F8==0, (compressed_bits&0x07)<<5, 0)
    all_byte0 = compressed_bits
    # all_byte0 = (compressed_bits& 0x07) << 5

    elements = tl.load(original_data + offsets + 1, mask=offsets + 1<n_elements)
    sign_mass_values = tl.cast(
        ((elements >> 8) & 0x80) | (elements & 0x7F), 
        tl.uint8
    )
    tl.store(sign_mass_group_ptr + 1, sign_mass_values, mask=offsets+1<n_elements)
    compressed_bits = (elements >> 7) - left
    # compressed_bits &= 0x00FF
    compressed_bits = tl.where(compressed_bits&0x00F8==0, (compressed_bits& 0x07) << 2, 0)
    all_byte0 |= compressed_bits
    # all_byte0 |= (compressed_bits& 0x07) << 2

    elements = tl.load(original_data + offsets + 2, mask=offsets+2<n_elements)
    sign_mass_values = tl.cast(
        ((elements >> 8) & 0x80) | (elements & 0x7F), 
        tl.uint8
    )
    tl.store(sign_mass_group_ptr + 2, sign_mass_values, mask=offsets+2<n_elements)
    compressed_bits = (elements >> 7) - left
    compressed_bits = tl.where(compressed_bits&0x00F8==0, compressed_bits , 0)
    all_byte0 |= (compressed_bits>> 1) & 0x03
    # all_byte0 |= (compressed_bits >> 1) & 0x03
    exp_offsets = pid * BLOCK_SIZE * 3 + tl.arange(0, BLOCK_SIZE)*3
    tl.store(exp_data + exp_offsets, tl.cast(all_byte0, tl.uint8), mask=exp_offsets<n_exp_data)
    all_byte1 = (compressed_bits & 0x01) << 7

    #####
    elements = tl.load(org_group_ptr + 3, mask=offsets+3<n_elements)
    sign_mass_values = tl.cast(
        ((elements >> 8) & 0x80) | (elements & 0x7F), 
        tl.uint8
    )
    tl.store(sign_mass_group_ptr + 3, sign_mass_values, mask=offsets+3<n_elements)
    compressed_bits = (elements >> 7) - left
    compressed_bits = tl.where(compressed_bits&0x00F8==0, (compressed_bits& 0x07) << 4, 0)
    all_byte1 |= compressed_bits

    elements = tl.load(org_group_ptr + 4, mask=offsets+4<n_elements)
    sign_mass_values = tl.cast(
        ((elements >> 8) & 0x80) | (elements & 0x7F), 
        tl.uint8
    )
    tl.store(sign_mass_group_ptr + 4, sign_mass_values, mask=offsets+4<n_elements)
    compressed_bits = (elements >> 7) - left
    compressed_bits = tl.where(compressed_bits&0x00F8==0, (compressed_bits& 0x07) <<1, 0)
    all_byte1 |= compressed_bits

    elements = tl.load(org_group_ptr + 5, mask=offsets+5<n_elements)
    sign_mass_values = tl.cast(
        ((elements >> 8) & 0x80) | (elements & 0x7F), 
        tl.uint8
    )
    tl.store(sign_mass_group_ptr + 5, sign_mass_values, mask=offsets+5<n_elements)
    compressed_bits = (elements >> 7) - left
    compressed_bits = tl.where(compressed_bits&0x00F8==0, compressed_bits, 0)
    all_byte1 |= (compressed_bits& 0x07)>>2
    tl.store(exp_data + exp_offsets + 1, tl.cast(all_byte1, tl.uint8), mask=exp_offsets + 1<n_exp_data)
    all_byte2 = (compressed_bits & 0x03) << 6

    #####
    elements = tl.load(org_group_ptr + 6, mask=offsets+6<n_elements)
    sign_mass_values = tl.cast(
        ((elements >> 8) & 0x80) | (elements & 0x7F), 
        tl.uint8
    )
    tl.store(sign_mass_group_ptr + 6, sign_mass_values, mask=offsets+6<n_elements)
    compressed_bits = (elements >> 7) - left
    compressed_bits = tl.where(compressed_bits&0x00F8==0, (compressed_bits&0x07)<<3, 0)
    all_byte2 |= compressed_bits

    elements = tl.load(org_group_ptr + 7, mask=offsets+7<n_elements)
    sign_mass_values = tl.cast(
        ((elements >> 8) & 0x80) | (elements & 0x7F), 
        tl.uint8
    )
    tl.store(sign_mass_group_ptr + 7, sign_mass_values, mask=offsets+7<n_elements)
    compressed_bits = (elements >> 7) - left
    compressed_bits = tl.where(compressed_bits&0x00F8==0, compressed_bits& 0x07, 0)
    all_byte2 |= compressed_bits
    tl.store(exp_data + exp_offsets + 2, tl.cast(all_byte2, tl.uint8), mask=exp_offsets+2<n_exp_data)


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
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)*8
    # mask = offsets < n_elements
    org_group_ptr = original_data + offsets
    sign_mass_group_ptr = sign_mass_data + offsets

    elements = tl.load(org_group_ptr, mask=offsets<n_elements)
    sign_mass_values = tl.cast(
        ((elements >> 8) & 0x80) | (elements & 0x7F), 
        tl.uint8
    )
    tl.store(sign_mass_group_ptr, sign_mass_values, mask=offsets<n_elements)
    # 0,000 0000 0,000 0000
    # 0000 000,0, 0000 0000
    compressed_bits = (elements >> 7) - left
    compressed_bits = tl.where(compressed_bits&0x00FE==0, compressed_bits&0x01, 0)
    all_byte0 = ((compressed_bits) << 7)

    elements = tl.load(org_group_ptr + 1, mask=offsets + 1<n_elements)
    sign_mass_values = tl.cast(
        ((elements >> 8) & 0x80) | (elements & 0x7F), 
        tl.uint8
    )
    tl.store(sign_mass_group_ptr + 1, sign_mass_values, mask=offsets + 1<n_elements)
    compressed_bits = (elements >> 7) - left
    compressed_bits = tl.where(compressed_bits&0x00FE==0, compressed_bits&0x01, 0)
    all_byte0 |= ((compressed_bits) << 6)


    elements = tl.load(org_group_ptr + 2, mask=offsets + 2<n_elements)
    sign_mass_values = tl.cast(
        ((elements >> 8) & 0x80) | (elements & 0x7F), 
        tl.uint8
    )
    tl.store(sign_mass_group_ptr + 2, sign_mass_values, mask=offsets + 2<n_elements)
    compressed_bits = (elements >> 7) - left
    compressed_bits = tl.where(compressed_bits&0x00FE==0, compressed_bits&0x01, 0)
    all_byte0 |= ((compressed_bits) << 5)


    elements = tl.load(org_group_ptr + 3, mask=offsets + 3<n_elements)
    sign_mass_values = tl.cast(
        ((elements >> 8) & 0x80) | (elements & 0x7F), 
        tl.uint8
    )
    tl.store(sign_mass_group_ptr + 3, sign_mass_values, mask=offsets + 3<n_elements)
    compressed_bits = (elements >> 7) - left
    compressed_bits = tl.where(compressed_bits&0x00FE==0, compressed_bits&0x01, 0)
    all_byte0 |= ((compressed_bits) << 4)


    elements = tl.load(org_group_ptr + 4, mask=offsets + 4<n_elements)
    sign_mass_values = tl.cast(
        ((elements >> 8) & 0x80) | (elements & 0x7F), 
        tl.uint8
    )
    tl.store(sign_mass_group_ptr + 4, sign_mass_values, mask=offsets + 4<n_elements)
    compressed_bits = (elements >> 7) - left
    compressed_bits = tl.where(compressed_bits&0x00FE==0, compressed_bits&0x01, 0)
    all_byte0 |= ((compressed_bits) << 3)


    elements = tl.load(org_group_ptr + 5, mask=offsets + 5<n_elements)
    sign_mass_values = tl.cast(
        ((elements >> 8) & 0x80) | (elements & 0x7F), 
        tl.uint8
    )
    tl.store(sign_mass_group_ptr + 5, sign_mass_values, mask=offsets + 5<n_elements)
    compressed_bits = (elements >> 7) - left
    compressed_bits = tl.where(compressed_bits&0x00FE==0, compressed_bits&0x01, 0)
    all_byte0 |= ((compressed_bits) << 2)


    elements = tl.load(org_group_ptr + 6, mask=offsets + 6<n_elements)
    sign_mass_values = tl.cast(
        ((elements >> 8) & 0x80) | (elements & 0x7F), 
        tl.uint8
    )
    tl.store(sign_mass_group_ptr + 6, sign_mass_values, mask=offsets + 6<n_elements)
    compressed_bits = (elements >> 7) - left
    compressed_bits = tl.where(compressed_bits&0x00FE==0, compressed_bits&0x01, 0)
    all_byte0 |= ((compressed_bits) << 1)


    elements = tl.load(org_group_ptr + 7, mask=offsets + 7<n_elements)
    sign_mass_values = tl.cast(
        ((elements >> 8) & 0x80) | (elements & 0x7F), 
        tl.uint8
    )
    tl.store(sign_mass_group_ptr + 7, sign_mass_values, mask=offsets + 7<n_elements)
    compressed_bits = (elements >> 7) - left
    compressed_bits = tl.where(compressed_bits&0x00FE==0, compressed_bits&0x01, 0)
    all_byte0 |= ((compressed_bits))

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
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)*4
    # mask = offsets < n_elements
    org_group_ptr = original_data + offsets
    sign_mass_group_ptr = sign_mass_data + offsets

    elements = tl.load(org_group_ptr, mask=offsets<n_elements)
    sign_mass_values = tl.cast(
        ((elements >> 8) & 0x80) | (elements & 0x7F), 
        tl.uint8
    )
    tl.store(sign_mass_group_ptr, sign_mass_values, mask=offsets<n_elements)
    # 16246
    # 0,000 0000 0,000 0000
    # 0000 000,0, 0000 0000
    compressed_bits = (elements >> 7) - left
    compressed_bits = tl.where(compressed_bits&0x00FC==0, compressed_bits&0x03, 0)
    # 4
    all_byte0 = ((compressed_bits) << 6)
    # 0
    elements = tl.load(org_group_ptr + 1, mask=offsets + 1<n_elements)
    sign_mass_values = tl.cast(
        ((elements >> 8) & 0x80) | (elements & 0x7F), 
        tl.uint8
    )
    tl.store(sign_mass_group_ptr + 1, sign_mass_values, mask=offsets + 1<n_elements)
    compressed_bits = (elements >> 7) - left
    compressed_bits = tl.where(compressed_bits&0x00FC==0, compressed_bits&0x03, 0)
    all_byte0 |= ((compressed_bits) << 4)

    elements = tl.load(org_group_ptr + 2, mask=offsets + 2<n_elements)
    sign_mass_values = tl.cast(
        ((elements >> 8) & 0x80) | (elements & 0x7F), 
        tl.uint8
    )
    tl.store(sign_mass_group_ptr + 2, sign_mass_values, mask=offsets + 2<n_elements)
    compressed_bits = (elements >> 7) - left
    compressed_bits = tl.where(compressed_bits&0x00FC==0, compressed_bits&0x03, 0)
    all_byte0 |= ((compressed_bits) << 2)

    elements = tl.load(org_group_ptr + 3, mask=offsets + 3<n_elements)
    sign_mass_values = tl.cast(
        ((elements >> 8) & 0x80) | (elements & 0x7F), 
        tl.uint8
    )
    tl.store(sign_mass_group_ptr + 3, sign_mass_values, mask=offsets + 3<n_elements)
    compressed_bits = (elements >> 7) - left
    compressed_bits = tl.where(compressed_bits&0x00FC==0, compressed_bits&0x03, 0)
    all_byte0 |= (compressed_bits)

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
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)*2
    # mask = offsets < n_elements
    org_group_ptr = original_data + offsets
    sign_mass_group_ptr = sign_mass_data + offsets

    elements = tl.load(org_group_ptr, mask=offsets<n_elements)
    sign_mass_values = tl.cast(
        ((elements >> 8) & 0x80) | (elements & 0x7F), 
        tl.uint8
    )
    tl.store(sign_mass_group_ptr, sign_mass_values, mask=offsets<n_elements)
    compressed_bits = (elements >> 7) - left
    # 0,000 0000 0,000 0000
    # 0000 000,0, 0000 0000
    compressed_bits = tl.where(compressed_bits&0x00F0==0, compressed_bits&0x0F, 0)
    all_byte0 = (compressed_bits) << 4

    elements = tl.load(org_group_ptr + 1, mask=offsets + 1<n_elements)
    sign_mass_values = tl.cast(
        ((elements >> 8) & 0x80) | (elements & 0x7F), 
        tl.uint8
    )
    tl.store(sign_mass_group_ptr + 1, sign_mass_values, mask=offsets + 1<n_elements)
    compressed_bits = (elements >> 7) - left
    compressed_bits = tl.where(compressed_bits&0x00F0==0, compressed_bits&0x0F, 0)
    all_byte0 |= (compressed_bits)

    exp_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(exp_data + exp_offsets, tl.cast(all_byte0, tl.uint8), mask=exp_offsets<n_exp_data)

# A:1010 B:1011 C:1100 D:1101 E:1110 F:1111
@triton.jit
def uncompress3bit_vector(
    original_data: tl.tensor,  #  uint16
    exp_data: tl.tensor,       #  uint8
    sign_mass_data: tl.tensor, #  uint8
    n_elements: tl.int32,      # 
    n_exp_data: tl.int32,      # 
    left: tl.constexpr,      # 
    BLOCK_SIZE: tl.constexpr,  # 
):
    pid = tl.program_id(axis=0)
    start_idx = pid * BLOCK_SIZE * 8
    offsets = start_idx + tl.arange(0, BLOCK_SIZE) * 8
    # 0
    sign_mass_values = tl.load(sign_mass_data + offsets, mask=offsets<n_elements)
    mass = tl.cast(sign_mass_values, tl.uint16)
    # & 0x007F
    sign = (mass & 0x0080) << 8
    origianl_values = sign | (mass& 0x007F)
    exp_start_idx = pid * BLOCK_SIZE * 3
    exp_offsets = exp_start_idx + tl.arange(0, BLOCK_SIZE) * 3
    exp_values = tl.load(exp_data + exp_offsets, mask=exp_offsets < n_exp_data)
    exp_values = tl.cast(exp_values, tl.uint16)
    exps = exp_values>>5
    exps &= 0x7
    exps = ((exps+left) << 7)
    origianl_values |= exps
    tl.store(original_data + offsets, origianl_values, mask=offsets<n_elements)
    # 1
    sign_mass_values = tl.load(sign_mass_data + offsets + 1, mask=offsets + 1<n_elements)
    mass = tl.cast(sign_mass_values, tl.uint16)
    # & 0x007F
    sign = (mass & 0x0080) << 8
    origianl_values = sign | (mass& 0x007F)
    exps = exp_values >> 2
    exps &= 0x7
    exps = ((exps+left) << 7)
    origianl_values |= exps
    tl.store(original_data + offsets + 1, origianl_values, mask=offsets + 1<n_elements)
    # 2
    sign_mass_values = tl.load(sign_mass_data + offsets + 2, mask=offsets + 2<n_elements)
    mass = tl.cast(sign_mass_values, tl.uint16)
    # & 0x007F
    sign = (mass & 0x0080) << 8
    origianl_values = sign | (mass& 0x007F)
    exps = (exp_values << 1)
    exp_values = tl.load(exp_data + exp_offsets + 1, mask=exp_offsets + 1 < n_exp_data)
    exp_values = tl.cast(exp_values, tl.uint16)
    exps |= (exp_values>>7)
    exps &= 0x7
    exps = ((exps+left) << 7)
    origianl_values |= exps
    tl.store(original_data + offsets + 2, origianl_values, mask=offsets + 2<n_elements)
    # 3
    sign_mass_values = tl.load(sign_mass_data + offsets + 3, mask=offsets + 3<n_elements)
    mass = tl.cast(sign_mass_values, tl.uint16)
    # & 0x007F
    sign = (mass & 0x0080) << 8
    origianl_values = sign | (mass& 0x007F)
    exps = exp_values >> 4
    exps &= 0x7
    exps = ((exps+left) << 7)
    origianl_values |= exps
    tl.store(original_data + offsets + 3, origianl_values, mask=offsets + 3<n_elements)
    # 4
    sign_mass_values = tl.load(sign_mass_data + offsets + 4, mask=offsets + 4<n_elements)
    mass = tl.cast(sign_mass_values, tl.uint16)
    # & 0x007F
    sign = (mass & 0x0080) << 8
    origianl_values = sign | (mass& 0x007F)
    exps = exp_values >> 1
    exps &= 0x7
    exps = ((exps+left) << 7)
    origianl_values |= exps
    tl.store(original_data + offsets + 4, origianl_values, mask=offsets + 4<n_elements)
    # 5
    sign_mass_values = tl.load(sign_mass_data + offsets + 5, mask=offsets + 5<n_elements)
    mass = tl.cast(sign_mass_values, tl.uint16)
    # & 0x007F
    sign = (mass & 0x0080) << 8
    origianl_values = sign | (mass& 0x007F)
    exps = exp_values << 2
    exp_values = tl.load(exp_data + exp_offsets + 2, mask=exp_offsets + 2 < n_exp_data)
    exp_values = tl.cast(exp_values, tl.uint16)
    exps |= (exp_values>>6)
    exps &= 0x7
    exps = ((exps+left) << 7)
    origianl_values |= exps
    tl.store(original_data + offsets + 5, origianl_values, mask=offsets + 5<n_elements)
    # 6
    sign_mass_values = tl.load(sign_mass_data + offsets + 6, mask=offsets + 6<n_elements)
    mass = tl.cast(sign_mass_values, tl.uint16)
    # & 0x007F
    sign = (mass & 0x0080) << 8
    origianl_values = sign | (mass& 0x007F)
    exps = (exp_values >> 3)
    exps &= 0x7
    exps = ((exps+left) << 7)
    origianl_values |= exps
    tl.store(original_data + offsets + 6, origianl_values, mask=offsets + 6<n_elements)
    # 7
    sign_mass_values = tl.load(sign_mass_data + offsets + 7, mask=offsets + 7<n_elements)
    mass = tl.cast(sign_mass_values, tl.uint16)
    # & 0x007F
    sign = (mass & 0x0080) << 8
    origianl_values = sign | (mass& 0x007F)
    exps = exp_values
    exps &= 0x7
    exps = ((exps+left) << 7)
    origianl_values |= exps
    tl.store(original_data + offsets + 7, origianl_values, mask=offsets + 7<n_elements)

# A:1010 B:1011 C:1100 D:1101 E:1110 F:1111
@triton.jit
def uncompress1bit_vector(
    original_data: tl.tensor,  #  uint16
    exp_data: tl.tensor,       #  uint8
    sign_mass_data: tl.tensor, #  uint8
    n_elements: tl.int32,      # 
    n_exp_data: tl.int32,      # 
    left: tl.constexpr,      # 
    BLOCK_SIZE: tl.constexpr,  # 
):
    pid = tl.program_id(axis=0)
    start_idx = pid * BLOCK_SIZE * 8
    offsets = start_idx + tl.arange(0, BLOCK_SIZE) * 8

    exp_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    exp_values = tl.load(exp_data + exp_offsets, mask=exp_offsets < n_exp_data)
    exp_values = tl.cast(exp_values, tl.uint16)
    for step in range(0, 8):
        sign_mass_values = tl.load(sign_mass_data + offsets + step, mask=offsets + step<n_elements)
        mass = tl.cast(sign_mass_values, tl.uint16)
        # & 0x007F
        sign = (mass & 0x0080) << 8
        origianl_values = sign | (mass& 0x007F)
        exps = exp_values>> (7 - step)
        exps &= 0x01
        exps = ((exps+left) << 7)
        origianl_values |= exps
        tl.store(original_data + offsets + step, origianl_values, mask=offsets + step<n_elements)

# A:1010 B:1011 C:1100 D:1101 E:1110 F:1111
@triton.jit
def uncompress2bit_vector(
    original_data: tl.tensor,  #  uint16
    exp_data: tl.tensor,       #  uint8
    sign_mass_data: tl.tensor, #  uint8
    n_elements: tl.int32,      # 
    n_exp_data: tl.int32,      # 
    left: tl.constexpr,      # 
    BLOCK_SIZE: tl.constexpr,  # 
):
    pid = tl.program_id(axis=0)
    start_idx = pid * BLOCK_SIZE * 4
    offsets = start_idx + tl.arange(0, BLOCK_SIZE) * 4

    exp_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    exp_values = tl.load(exp_data + exp_offsets, mask=exp_offsets < n_exp_data)
    exp_values = tl.cast(exp_values, tl.uint16)
    for step in range(0, 4):
        sign_mass_values = tl.load(sign_mass_data + offsets + step, mask=offsets + step<n_elements)
        mass = tl.cast(sign_mass_values, tl.uint16)
        # & 0x007F
        sign = (mass & 0x0080) << 8
        origianl_values = sign | (mass& 0x007F)
        exps = exp_values>> (6 - 2*step)
        exps &= 0x03
        exps = ((exps+left) << 7)
        origianl_values |= exps
        tl.store(original_data + offsets + step, origianl_values, mask=offsets + step<n_elements)

# A:1010 B:1011 C:1100 D:1101 E:1110 F:1111
@triton.jit
def uncompress4bit_vector(
    original_data: tl.tensor,  #  uint16
    exp_data: tl.tensor,       #  uint8
    sign_mass_data: tl.tensor, #  uint8
    n_elements: tl.int32,      # 
    n_exp_data: tl.int32,      # 
    left: tl.constexpr,      # 
    BLOCK_SIZE: tl.constexpr,  # 
):
    pid = tl.program_id(axis=0)
    start_idx = pid * BLOCK_SIZE * 2
    offsets = start_idx + tl.arange(0, BLOCK_SIZE) * 2

    exp_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    exp_values = tl.load(exp_data + exp_offsets, mask=exp_offsets < n_exp_data)
    exp_values = tl.cast(exp_values, tl.uint16)
    for step in range(0, 2):
        sign_mass_values = tl.load(sign_mass_data + offsets + step, mask=offsets + step<n_elements)
        mass = tl.cast(sign_mass_values, tl.uint16)
        # & 0x007F
        sign = (mass & 0x0080) << 8
        origianl_values = sign | (mass& 0x007F)
        exps = exp_values>> (4 - 4*step)
        exps &= 0xF
        exps = ((exps+left) << 7)
        origianl_values |= exps
        tl.store(original_data + offsets + step, origianl_values, mask=offsets + step<n_elements)

# 206
@triton.jit
def uncompress3bit_bitmaps_vector_merge(
    original_data: tl.tensor,  #  uint16
    bitmaps: tl.tensor,
    exp_data: tl.tensor,       #  uint8
    sign_mass_data: tl.tensor, #  uint8
    n_elements: tl.int32,      #
    left: tl.constexpr,      #
    BLOCK_SIZE: tl.constexpr,  #
):
    # tl.assume(n_elements*3==n_exp_data*8)
    pid = tl.program_id(axis=0)

    # 1) ���� coalesced ���� sign_mass [start_idx : start_idx+8*BLOCK_SIZE]
    # mask_flat = flat_offs < n_elements
    # , mask=mask_flat, other=0

    # 2) ���� 3 �� exp �ֽڣ�ÿ 8 ��Ԫ��һ�� 3 �ֽڣ�
    exp_start_idx = pid * BLOCK_SIZE * 3
    exp_offsets = exp_start_idx + tl.arange(0, BLOCK_SIZE) * 3
    b0 = tl.load(exp_data + exp_offsets) # uint8
    b1 = tl.load(exp_data + exp_offsets + 1) # uint8
    b2 = tl.load(exp_data + exp_offsets + 2) # uint8

    # ÿ lane �� b0,b1,b2 ��ȡ 8 �� 3-bit ָ�� e0~e7
    e0 = (b0 >> 5)
    e1 = (b0 >> 2)
    e2 = ((b0 << 1) | (b1 >> 7))
    e3 = (b1 >> 4)
    e4 = (b1 >> 1)
    e5 = ((b1 << 2) | (b2 >> 6))
    e6 = (b2 >> 3)
    e7 = b2
    # ��Ч���Ȱ� lane ����� uint64��ÿ�ֽ�һ�� 3-bit�����ٰ��ֽ�չ������ tl.where
    e_packed = (
        e0.cast(tl.uint64)
        | (e1.cast(tl.uint64) << 8)
        | (e2.cast(tl.uint64) << 16)
        | (e3.cast(tl.uint64) << 24)
        | (e4.cast(tl.uint64) << 32)
        | (e5.cast(tl.uint64) << 40)
        | (e6.cast(tl.uint64) << 48)
        | (e7.cast(tl.uint64) << 56)
    )
    out_idx = tl.arange(0, BLOCK_SIZE * 8)
    lane_idx = out_idx // 8
    byte_idx = out_idx % 8
    e8 = ((tl.gather(e_packed, lane_idx, axis=0) >> (byte_idx * 8)) & 0x07).cast(tl.uint8)

    start_idx = pid * BLOCK_SIZE * 8
    flat_offs = start_idx + tl.arange(0, BLOCK_SIZE * 8)

    bitmap_exp = (e8 == 0)
    tl.store(bitmaps + flat_offs, bitmap_exp, mask=flat_offs < n_elements)

    sign_mass_flat = tl.load(sign_mass_data + flat_offs)
    mass = tl.cast(sign_mass_flat, tl.uint16)
    sign = (mass & 0x0080) << 8
    base_val_flat = sign | (mass & 0x007F)

    exps_shifted = (e8 + left) 
    origianl_values = base_val_flat | (exps_shifted.cast(tl.uint16)<<7)
    tl.store(original_data + flat_offs, origianl_values, mask=flat_offs < n_elements)

@triton.jit
def uncompress4bit_bitmaps_vector_merge(
    original_data: tl.tensor,  #  uint16
    bitmaps: tl.tensor,
    exp_data: tl.tensor,       #  uint8
    sign_mass_data: tl.tensor, #  uint8
    n_elements: tl.int32,      # 
    # n_exp_data: tl.int32,      # 
    left: tl.constexpr,      # 
    BLOCK_SIZE: tl.constexpr,  # 
):
    pid = tl.program_id(axis=0)
    start_idx = pid * BLOCK_SIZE * 2
    offsets = start_idx + tl.arange(0, BLOCK_SIZE * 2)

    exp_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    exp_values = tl.load(exp_data + exp_offsets)

    out_idx = tl.arange(0, BLOCK_SIZE * 2)
    lane_idx = out_idx // 2
    byte_idx = out_idx % 2
    # 1111
    e2 = ((tl.gather(exp_values, lane_idx, axis=0) >> ((1-byte_idx) * 4)) & 0x0F).cast(tl.uint8)

    bitmap_exp = (e2 == 0)
    tl.store(bitmaps + offsets, bitmap_exp, mask=offsets < n_elements)

    sign_mass_values = tl.load(sign_mass_data + offsets, mask=offsets<n_elements)
    mass = tl.cast(sign_mass_values, tl.uint16)
    sign = (mass & 0x0080) << 8
    base_val_flat = sign | (mass& 0x007F)
    # exps = ((e2+left) << 7)
    # origianl_values |= exps
    exps_shifted = (e2 + left) 
    origianl_values = base_val_flat | (exps_shifted.cast(tl.uint16)<<7)
    tl.store(original_data + offsets, origianl_values, mask=offsets < n_elements)

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
    def __init__(self, right=129, nbits=3, mem_pool=None):
        self.device = None
        self.original_shape = None
        self.n_elements = None
        self.n_exp_data = None
        self.sign_mass_data = None
        self.exp_data = None
        self.invalid_exp = None, None
        self.blocksize = 128
        # 64 -> 202
        # 128 -> 203
        # (left, right] is compressed
        # 0,000 0000 0,000 0000
        # 0x7F80
        self.left = right - 2**nbits + 1
        self.right = right
        self.nbits = nbits
        self.mem_pool = mem_pool
        uncompress_lists = [self.uncompress1bit, self.uncompress2bit, self.uncompress3bit_bitmaps, self.uncompress4bit_bitmaps]
        self.target_uncompressor = uncompress_lists[self.nbits-1]
        self.left_nbits = self.left << 7

    def set_mem_pool(self, max_numel):
        self.mem_pool = torch.empty(max_numel, device=self.device, dtype=torch.uint16)

    def get_memory(self):
        if self.sign_mass_data is None:
            return {'compressed_memory': 0, 'original_memory': 0}
        m1 = self.sign_mass_data.numel() * self.sign_mass_data.element_size()
        m2 = self.exp_data.numel() * self.exp_data.element_size()
        m3 = self.invalid_exp.numel() * self.invalid_exp.element_size()
        compressed_memory = m1+m2+m3
        original_m = self.n_elements * 2
        return {'compressed_memory':compressed_memory, 'original_memory': original_m, 'ratio': compressed_memory/original_m}


    def compress(self, original_data_bfloat16, need_adjust=False):
        # torch.cuda.synchronize()
        
        self.device = original_data_bfloat16.device
        self.original_shape = original_data_bfloat16.shape
        self.invalid_exp, extracted_values = get_invalid_index_values(original_data_bfloat16, self.left, self.right, need_adjust=need_adjust)
        
        self.n_elements = original_data_bfloat16.numel()
        if self.nbits == 3:
            assert self.n_elements % 8 == 0
            n_threads1 = self.n_elements // 8
            self.n_exp_data = n_threads1 * 3
            self.sign_mass_data = torch.zeros(self.n_elements, dtype=torch.uint8, device=self.device)
            self.exp_data = torch.zeros(self.n_exp_data, dtype=torch.uint8, device=self.device)
            grid1 = lambda meta: (triton.cdiv(n_threads1, meta['BLOCK_SIZE']), )
            compress3bit_vector[grid1](original_data_bfloat16.view(torch.uint16), self.exp_data, \
                                            self.sign_mass_data, self.n_elements, self.n_exp_data, self.left, self.blocksize)
        elif self.nbits == 1:
            n_threads1 = self.n_elements // 8
            self.n_exp_data = n_threads1
            self.sign_mass_data = torch.zeros(self.n_elements, dtype=torch.uint8, device=self.device)
            self.exp_data = torch.zeros(self.n_exp_data, dtype=torch.uint8, device=self.device)
            grid1 = lambda meta: (triton.cdiv(n_threads1, meta['BLOCK_SIZE']), )
            compress1bit_vector[grid1](original_data_bfloat16.view(torch.uint16), self.exp_data, \
                                            self.sign_mass_data, self.n_elements, self.n_exp_data, self.left, self.blocksize)
        elif self.nbits == 2:
            n_threads1 = self.n_elements // 4
            self.n_exp_data = n_threads1
            self.sign_mass_data = torch.zeros(self.n_elements, dtype=torch.uint8, device=self.device)
            self.exp_data = torch.zeros(self.n_exp_data, dtype=torch.uint8, device=self.device)
            grid1 = lambda meta: (triton.cdiv(n_threads1, meta['BLOCK_SIZE']), )
            compress2bit_vector[grid1](original_data_bfloat16.view(torch.uint16), self.exp_data, \
                                            self.sign_mass_data, self.n_elements, self.n_exp_data, self.left, self.blocksize)
        elif self.nbits == 4:
            n_threads1 = self.n_elements // 2
            self.n_exp_data = n_threads1
            self.sign_mass_data = torch.zeros(self.n_elements, dtype=torch.uint8, device=self.device)
            self.exp_data = torch.zeros(self.n_exp_data, dtype=torch.uint8, device=self.device)
            grid1 = lambda meta: (triton.cdiv(n_threads1, meta['BLOCK_SIZE']), )
            compress4bit_vector[grid1](original_data_bfloat16.view(torch.uint16), self.exp_data, \
                                            self.sign_mass_data, self.n_elements, self.n_exp_data, self.left, self.blocksize)
        return extracted_values

    def uncompress1bit(self, uncompressed_odb):
        grid1 = lambda meta: (triton.cdiv((self.n_elements // 8), meta['BLOCK_SIZE']), )
        uncompress1bit_vector[grid1](uncompressed_odb, self.exp_data, \
                                            self.sign_mass_data, self.n_elements, self.n_exp_data, self.left, self.blocksize)

    def uncompress2bit(self, uncompressed_odb):
        grid1 = lambda meta: (triton.cdiv((self.n_elements // 4), meta['BLOCK_SIZE']), )
        uncompress2bit_vector[grid1](uncompressed_odb, self.exp_data, \
                                            self.sign_mass_data, self.n_elements, self.n_exp_data, self.left, self.blocksize)

    def uncompress3bit(self, uncompressed_odb):
        grid1 = lambda meta: (triton.cdiv((self.n_elements // 8), meta['BLOCK_SIZE']), )
        uncompress3bit_vector[grid1](uncompressed_odb, self.exp_data, \
                                            self.sign_mass_data, self.n_elements, self.n_exp_data, self.left, self.blocksize)

    def uncompress3bit_bitmaps(self, uncompressed_odb, bitmaps):
        grid1 = lambda meta: (triton.cdiv(self.n_elements, meta['BLOCK_SIZE']*8), )
        uncompress3bit_bitmaps_vector_merge[grid1](uncompressed_odb, bitmaps, self.exp_data, \
                                            self.sign_mass_data, self.n_elements, self.left, self.blocksize)

    def uncompress4bit(self, uncompressed_odb):
        grid1 = lambda meta: (triton.cdiv((self.n_elements // 2), meta['BLOCK_SIZE']), )
        uncompress4bit_vector[grid1](uncompressed_odb, self.exp_data, \
                                            self.sign_mass_data, self.n_elements, self.n_exp_data, self.left, self.blocksize)

    def uncompress4bit_bitmaps(self, uncompressed_odb, bitmaps):
        grid1 = lambda meta: (triton.cdiv((self.n_elements // 2), meta['BLOCK_SIZE']), )
        uncompress4bit_bitmaps_vector_merge[grid1](uncompressed_odb, bitmaps, self.exp_data, \
                                            self.sign_mass_data, self.n_elements, self.left, self.blocksize)

    def uncompress(self):
        uncompressed_odb = torch.empty(self.n_elements, device=self.device, dtype=torch.uint16)
        self.target_uncompressor(uncompressed_odb)

        bitmaps = torch.zeros(self.n_elements, dtype=torch.bool, device=self.device)
        grid3 = lambda meta: (triton.cdiv(bitmaps.numel(), meta['BLOCK_SIZE']),)
        get_bitmaps[grid3](uncompressed_odb, bitmaps, self.left_nbits, bitmaps.numel(), self.blocksize)
        invalid_index = torch.where(bitmaps)[0]

        grid4 = lambda meta: (triton.cdiv(invalid_index.numel(), meta['BLOCK_SIZE']), )
        fill_invalid_data[grid4](uncompressed_odb, self.n_elements, invalid_index, \
                                 self.invalid_exp, invalid_index.numel(), self.blocksize)

        return uncompressed_odb.view(self.original_shape).view(torch.bfloat16)
    
    
    def get_block_base(self, bitmaps, BLOCK_SIZE, n_blocks):
        block_counts = torch.empty((n_blocks,), device=self.device, dtype=torch.int32)
        grid = (n_blocks,)
        count_invalid_blocks_from_bitmap[grid](
            bitmaps,
            block_counts,
            N=self.n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
            # num_warps=4,
        )

        # GPU prefix-sum: exclusive base
        prefix = torch.cumsum(block_counts, dim=0)  # inclusive
        return prefix

    # def uncompress_with_weights(self, uncompressed_odb):
    #     bitmaps = torch.empty(self.n_elements, dtype=torch.bool, device=self.device)
    #     self.uncompress3bit_bitmaps(uncompressed_odb, bitmaps)

    #     invalid_index = torch.where(bitmaps)[0]
    #     grid4 = lambda meta: (triton.cdiv(invalid_index.numel(), meta['BLOCK_SIZE']), )
    #     fill_invalid_data[grid4](uncompressed_odb, self.n_elements, invalid_index, \
    #                              self.invalid_exp, invalid_index.numel(), self.blocksize)

    def uncompress_with_weights(self, uncompressed_odb):
        bitmaps = torch.empty(self.n_elements, dtype=torch.bool, device=self.device)
        self.target_uncompressor(uncompressed_odb, bitmaps)

        BLOCK_SIZE = max(256, int(self.blocksize) * 8)
        n_blocks = triton.cdiv(self.n_elements, BLOCK_SIZE)
        block_base = self.get_block_base(bitmaps, BLOCK_SIZE, n_blocks)

        grid = (n_blocks,)
        fill_invalid_from_bitmap_two_pass[grid](
            uncompressed_odb,
            bitmaps,
            self.invalid_exp,
            block_base,
            # total_invalid=total_invalid,
            N=self.n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )

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
    cp = CompressorBitMap(right=127, nbits=3)
    cp.compress(original_data_bfloat16)
    uncompressed_odb=torch.zeros_like(original_data_bfloat16)
    cp.uncompress_with_weights(uncompressed_odb.view(torch.uint16))

    sub_res = original_data_bfloat16 - uncompressed_odb
    print("uncompress:", torch.sum(uncompressed_odb-original_data_bfloat16))
 
 