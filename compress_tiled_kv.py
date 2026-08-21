from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Iterable
from contextlib import nullcontext

import torch

from kv_compression.compress_bitmap_merge import CompressorBitMap

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    TRITON_AVAILABLE = False


def _device_context(device):
    torch_device = torch.device(device)
    if torch_device.type == 'cuda':
        return torch.cuda.device(torch_device)
    return nullcontext()


_DEFAULT_TILE_NBITS = 3
_SUPPORTED_TILE_NBITS = (1, 2, 3, 4)


if TRITON_AVAILABLE:
    @triton.jit
    def _decode_tiled_bf16_kernel(
        sign_mass_ptr,
        exp_data_ptr,
        invalid_exp_ptr,
        tile_invalid_base_ptr,
        out_ptr,
        seq_len,
        seq_tile_size,
        head_dim,
        tile_num_lanes,
        num_sequence_tiles,
        num_kv_heads,
        left,
        nbits: tl.constexpr,
        BLOCK_SEQ: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        elems_per_tile = seq_tile_size * head_dim
        bytes_per_tile = seq_tile_size * tile_num_lanes * nbits
        offs = tl.arange(0, BLOCK_SEQ * BLOCK_D)
        tile_seq = offs // BLOCK_D
        tile_d = offs % BLOCK_D
        valid = (tile_seq < seq_tile_size) & (tile_d < head_dim)

        seq_tile_idx = pid % num_sequence_tiles
        kv_head_idx = (pid // num_sequence_tiles) % num_kv_heads
        batch_idx = pid // (num_kv_heads * num_sequence_tiles)
        seq_global = seq_tile_idx * seq_tile_size + tile_seq
        valid = valid & (seq_global < seq_len)

        sm_base = pid * elems_per_tile
        exp_base = pid * bytes_per_tile
        out_base = (batch_idx * num_kv_heads + kv_head_idx) * seq_len * head_dim

        lane_idx = tile_d // 8
        inner_idx = tile_d % 8
        packed_offsets = exp_base + (tile_seq * tile_num_lanes + lane_idx) * nbits
        packed = tl.zeros([BLOCK_SEQ * BLOCK_D], dtype=tl.uint32)
        for byte_idx in range(4):
            if byte_idx < nbits:
                byte_value = tl.load(exp_data_ptr + packed_offsets + byte_idx, mask=valid, other=0).to(tl.uint32)
                packed = packed | (byte_value << (8 * (nbits - 1 - byte_idx)))
        shifts = (7 - inner_idx) * nbits
        exp_mask = (1 << nbits) - 1
        e_n = ((packed >> shifts) & exp_mask).to(tl.uint8)

        is_invalid = (e_n == 0) & valid
        c = tl.cumsum(is_invalid.to(tl.int32), axis=0) - 1
        tile_base_prev = tl.load(tile_invalid_base_ptr + pid - 1, mask=pid > 0, other=0)
        invalid_idx = tile_base_prev + c
        inv_exp = tl.load(invalid_exp_ptr + invalid_idx, mask=is_invalid, other=0).to(tl.uint16)

        sm_offsets = sm_base + tile_seq * head_dim + tile_d
        sign_mass = tl.load(sign_mass_ptr + sm_offsets, mask=valid, other=0).to(tl.uint16)
        sign_mass_bf16 = ((sign_mass & 0x0080) << 8) | (sign_mass & 0x007F)
        org_exp = tl.where(is_invalid, inv_exp, (e_n + left).to(tl.uint16))
        out_u16 = sign_mass_bf16 | (org_exp << 7)
        out_offsets = out_base + seq_global * head_dim + tile_d
        tl.store(out_ptr + out_offsets, out_u16, mask=valid)
    @triton.jit
    def _decode_open_rows_bf16_kernel(
        sign_mass_ptr,       # [T*H, head_dim] uint8, token-major row order
        exp_code_ptr,        # [T*H, head_dim] uint8, token-major, 0 means outlier
        invalid_exp_ptr,     # [total_invalids] uint8, packed in token-major row order
        row_offsets_ptr,     # [T*H+1] int32, CSR offsets into invalid_exp per row
        out_ptr,             # [H, T, head_dim] uint16 output
        num_kv_heads,
        token_count,
        head_dim,
        left,
        BLOCK_D: tl.constexpr,
    ):
        # pid_h = head index, pid_t = token index
        pid_h = tl.program_id(axis=0)
        pid_t = tl.program_id(axis=1)

        # row index in token-major storage: row = token_idx * H + head_idx
        row_idx = pid_t * num_kv_heads + pid_h

        d_offs = tl.arange(0, BLOCK_D)
        valid = d_offs < head_dim

        # Load sign_mass and exp_code for this (token, head) row
        row_base = row_idx * head_dim
        sign_mass = tl.load(sign_mass_ptr + row_base + d_offs, mask=valid, other=0).to(tl.uint16)
        e_code = tl.load(exp_code_ptr + row_base + d_offs, mask=valid, other=0).to(tl.uint16)

        is_invalid = (e_code == 0) & valid

        # Locate this row's outliers in invalid_exp via CSR offsets
        inv_row_start = tl.load(row_offsets_ptr + row_idx).to(tl.int32)
        # Within-row local index of each invalid element
        c = tl.cumsum(is_invalid.to(tl.int32), axis=0) - 1
        inv_exp = tl.load(invalid_exp_ptr + inv_row_start + c, mask=is_invalid, other=0).to(tl.uint16)

        # Reconstruct bfloat16 bits
        sign_mass_bf16 = ((sign_mass & 0x0080) << 8) | (sign_mass & 0x007F)
        org_exp = tl.where(is_invalid, inv_exp, (e_code + left).to(tl.uint16))
        out_u16 = sign_mass_bf16 | (org_exp << 7)

        # Output layout: [H, T, head_dim], out_base = (pid_h * token_count + pid_t) * head_dim
        out_base = (pid_h * token_count + pid_t) * head_dim
        tl.store(out_ptr + out_base + d_offs, out_u16, mask=valid)

else:
    _decode_tiled_bf16_kernel = None
    _decode_open_rows_bf16_kernel = None


@dataclass
class DecodeNativeCompressedTensor:
    sign_mass_data: torch.Tensor
    exp_data: torch.Tensor
    invalid_exp: torch.Tensor
    tile_invalid_base: torch.Tensor
    left: int
    right: int
    nbits: int
    original_shape: tuple[int, ...]
    seq_tile_size: int
    head_dim: int
    tile_head_dim: int
    tile_num_lanes: int
    num_sequence_tiles: int
    num_kv_heads: int
    device: torch.device

    @property
    def n_elements(self) -> int:
        if not self.original_shape:
            return 0
        result = 1
        for d in self.original_shape:
            result *= d
        return result

    def get_memory(self) -> dict[str, float]:
        compressed_memory = (
            self.sign_mass_data.numel() * self.sign_mass_data.element_size()
            + self.exp_data.numel() * self.exp_data.element_size()
            + self.invalid_exp.numel() * self.invalid_exp.element_size()
            + self.tile_invalid_base.numel() * self.tile_invalid_base.element_size()
        )
        original_memory = self.n_elements * 2
        ratio = 0.0 if original_memory == 0 else compressed_memory / original_memory
        return {
            'compressed_memory': compressed_memory,
            'original_memory': original_memory,
            'ratio': ratio,
        }

    def export_state(self) -> dict[str, Any]:
        return {
            'layout': 'decode_native',
            'right': self.right,
            'nbits': self.nbits,
            'left': self.left,
            'device': self.device,
            'original_shape': self.original_shape,
            'sign_mass_data': self.sign_mass_data,
            'exp_data': self.exp_data,
            'invalid_exp': self.invalid_exp,
            'tile_invalid_base': self.tile_invalid_base,
            'seq_tile_size': self.seq_tile_size,
            'head_dim': self.head_dim,
            'tile_head_dim': self.tile_head_dim,
            'tile_num_lanes': self.tile_num_lanes,
            'num_kv_heads': self.num_kv_heads,
            'n_elements': self.n_elements,
        }

    def export_decode_view(self) -> dict[str, Any]:
        state = self.export_state()
        state['kind'] = 'decode_native'
        return state

    def materialize(self) -> torch.Tensor:
        batch_size, kv_heads, seq_len, head_dim = self.original_shape
        if not TRITON_AVAILABLE:
            raise RuntimeError('Triton is required for decode-native materialization')
        out_u16 = torch.empty((batch_size * kv_heads * seq_len * head_dim,), dtype=torch.uint16, device=self.device)
        total_tiles = batch_size * kv_heads * self.num_sequence_tiles
        with _device_context(self.device):
            _decode_tiled_bf16_kernel[(total_tiles,)](
                self.sign_mass_data,
                self.exp_data,
                self.invalid_exp,
                self.tile_invalid_base,
                out_u16,
                seq_len,
                self.seq_tile_size,
                head_dim,
                self.tile_num_lanes,
                self.num_sequence_tiles,
                kv_heads,
                self.left,
                nbits=self.nbits,
                BLOCK_SEQ=triton.next_power_of_2(self.seq_tile_size),
                BLOCK_D=triton.next_power_of_2(head_dim),
            )
        return out_u16.view(torch.bfloat16).view(self.original_shape).contiguous()


@dataclass
class _BaseTiledCompressedTensor:
    sign_mass_data: torch.Tensor
    exp_data: torch.Tensor
    invalid_exp: torch.Tensor
    tile_invalid_base: torch.Tensor
    left: int
    right: int
    nbits: int
    original_shape: tuple[int, ...]
    seq_tile_size: int
    head_dim: int
    tile_head_dim: int
    tile_num_lanes: int
    num_sequence_tiles: int
    num_kv_heads: int
    device: torch.device

    layout: str = 'tiled'
    decode_kind: str = 'tiled_decode'
    has_outliers: bool = True

    @property
    def n_elements(self) -> int:
        if not self.original_shape:
            return 0
        result = 1
        for d in self.original_shape:
            result *= d
        return result

    def get_memory(self) -> dict[str, float]:
        compressed_memory = (
            self.sign_mass_data.numel() * self.sign_mass_data.element_size()
            + self.exp_data.numel() * self.exp_data.element_size()
            + self.invalid_exp.numel() * self.invalid_exp.element_size()
            + self.tile_invalid_base.numel() * self.tile_invalid_base.element_size()
        )
        original_memory = self.n_elements * 2
        ratio = 0.0 if original_memory == 0 else compressed_memory / original_memory
        return {
            'compressed_memory': compressed_memory,
            'original_memory': original_memory,
            'ratio': ratio,
        }

    def export_state(self) -> dict[str, Any]:
        state = {
            'layout': self.layout,
            'right': self.right,
            'nbits': self.nbits,
            'left': self.left,
            'device': self.device,
            'original_shape': self.original_shape,
            'sign_mass_data': self.sign_mass_data,
            'exp_data': self.exp_data,
            'seq_tile_size': self.seq_tile_size,
            'head_dim': self.head_dim,
            'tile_head_dim': self.tile_head_dim,
            'tile_num_lanes': self.tile_num_lanes,
            'num_sequence_tiles': self.num_sequence_tiles,
            'num_kv_heads': self.num_kv_heads,
            'n_elements': self.n_elements,
            'has_outliers': self.has_outliers,
        }
        if self.has_outliers:
            state['invalid_exp'] = self.invalid_exp
            state['tile_invalid_base'] = self.tile_invalid_base
        return state

    def export_decode_view(self) -> dict[str, Any]:
        state = self.export_state()
        state['kind'] = self.decode_kind
        return state

    def materialize(self) -> torch.Tensor:
        batch_size, kv_heads, seq_len, head_dim = self.original_shape
        if not TRITON_AVAILABLE:
            raise RuntimeError('Triton is required for tiled materialization')
        out_u16 = torch.empty((batch_size * kv_heads * seq_len * head_dim,), dtype=torch.uint16, device=self.device)
        total_tiles = batch_size * kv_heads * self.num_sequence_tiles
        with _device_context(self.device):
            _decode_tiled_bf16_kernel[(total_tiles,)](
                self.sign_mass_data,
                self.exp_data,
                self.invalid_exp,
                self.tile_invalid_base,
                out_u16,
                seq_len,
                self.seq_tile_size,
                head_dim,
                self.tile_num_lanes,
                self.num_sequence_tiles,
                kv_heads,
                self.left,
                nbits=self.nbits,
                BLOCK_SEQ=triton.next_power_of_2(self.seq_tile_size),
                BLOCK_D=triton.next_power_of_2(head_dim),
            )
        return out_u16.view(torch.bfloat16).view(self.original_shape).contiguous()


@dataclass
class TiledCompressedTensor(_BaseTiledCompressedTensor):
    pass


@dataclass
class OpenTiledCompressionRows:
    sign_mass_rows: torch.Tensor
    exp_code_rows: torch.Tensor
    invalid_exp: torch.Tensor
    invalid_exp_row_offsets: torch.Tensor
    left: int
    right: int
    nbits: int
    original_shape: tuple[int, ...]
    seq_tile_size: int
    head_dim: int
    tile_head_dim: int
    tile_num_lanes: int
    num_kv_heads: int
    device: torch.device
    capacity: Optional[int] = None
    invalid_exp_capacity: Optional[int] = None

    @property
    def token_count(self) -> int:
        return int(self.original_shape[-2])

    @property
    def row_capacity(self) -> int:
        rows = 0 if self.sign_mass_rows.numel() == 0 else int(self.sign_mass_rows.shape[0])
        if self.num_kv_heads > 0:
            rows //= self.num_kv_heads
        if self.capacity is not None:
            rows = max(rows, int(self.capacity))
        return rows

    @property
    def is_full(self) -> bool:
        return self.token_count >= self.seq_tile_size

    @property
    def invalid_exp_count(self) -> int:
        # invalid_exp is already sliced to exactly the valid entries at construction time,
        # so shape[0] is a plain Python int — no GPU round-trip needed.
        return int(self.invalid_exp.shape[0])

    def export_state(self) -> dict[str, Any]:
        token_count = self.token_count
        n_rows = token_count * self.num_kv_heads
        return {
            'layout': 'tiled_open_rows',
            'kind': 'tiled_open_rows',
            'left': self.left,
            'right': self.right,
            'nbits': self.nbits,
            'device': self.device,
            'original_shape': self.original_shape,
            # Return views (no copy). Callers that need contiguous memory are
            # responsible for calling .contiguous() themselves before kernel launch.
            'sign_mass_rows': self.sign_mass_rows[:n_rows],
            'exp_code_rows': self.exp_code_rows[:n_rows],
            'invalid_exp': self.invalid_exp,
            'invalid_exp_row_offsets': self.invalid_exp_row_offsets[:n_rows + 1],
            'seq_tile_size': self.seq_tile_size,
            'head_dim': self.head_dim,
            'tile_head_dim': self.tile_head_dim,
            'tile_num_lanes': self.tile_num_lanes,
            'num_kv_heads': self.num_kv_heads,
            'num_sequence_tiles': 1,
            'n_elements': self.token_count * self.num_kv_heads * self.head_dim if self.original_shape else 0,
            'has_outliers': True,
            'open_tile_token_count': token_count,
            'capacity': self.row_capacity,
            'invalid_exp_capacity': int(self.invalid_exp.shape[0]),
        }

    def materialize(self) -> torch.Tensor:
        if self.token_count == 0:
            return torch.empty(self.original_shape, dtype=torch.bfloat16, device=self.device)
        if not TRITON_AVAILABLE:
            raise RuntimeError('Triton is required for open-rows materialization')
        token_count = self.token_count
        num_kv_heads = self.num_kv_heads
        head_dim = self.head_dim
        # sign_mass_rows / exp_code_rows: [T*H, head_dim], token-major, all on GPU.
        # invalid_exp: [total_invalids] uint8, token-major row order, on GPU.
        # invalid_exp_row_offsets: [T*H+1] int32 CSR offsets, on GPU.
        # The new kernel reads all three directly without any CPU round-trip or reordering.
        out_u16 = torch.empty((num_kv_heads * token_count * head_dim,), dtype=torch.uint16, device=self.device)
        grid = (num_kv_heads, token_count)
        with _device_context(self.device):
            _decode_open_rows_bf16_kernel[grid](
                self.sign_mass_rows,
                self.exp_code_rows,
                self.invalid_exp,
                self.invalid_exp_row_offsets,
                out_u16,
                num_kv_heads,
                token_count,
                head_dim,
                self.left,
                BLOCK_D=triton.next_power_of_2(head_dim),
            )
        # Output shape: [1, num_kv_heads, token_count, head_dim]
        return out_u16.view(torch.bfloat16).view(1, num_kv_heads, token_count, head_dim).contiguous()


def _pack_nbit_groups(exp_values: torch.Tensor, nbits: int) -> torch.Tensor:
    if exp_values.numel() == 0:
        return torch.empty((0,), dtype=torch.uint8, device=exp_values.device)
    if nbits not in _SUPPORTED_TILE_NBITS:
        raise ValueError(f'Unsupported tiled nbits={nbits}, expected one of {_SUPPORTED_TILE_NBITS}')
    if exp_values.numel() % 8 != 0:
        raise ValueError(f'Packed tiled exponents require groups of 8 values, got {exp_values.numel()}')
    groups = exp_values.view(-1, 8).to(torch.int64)
    bit_shifts = torch.tensor(
        [(7 - idx) * nbits for idx in range(8)],
        device=exp_values.device,
        dtype=torch.int64,
    )
    packed = torch.sum(groups << bit_shifts, dim=-1)
    out = torch.empty((packed.numel() * nbits,), dtype=torch.uint8, device=exp_values.device)
    for byte_idx in range(nbits):
        byte_shift = 8 * (nbits - 1 - byte_idx)
        out[byte_idx::nbits] = ((packed >> byte_shift) & 0xFF).to(torch.uint8)
    return out


def _build_open_tiled_rows(
    original_data_bfloat16: torch.Tensor,
    right: int,
    nbits: int,
    seq_tile_size: int,
    tile_head_dim: int,
    need_adjust: bool,
) -> tuple[OpenTiledCompressionRows, Optional[torch.Tensor]]:
    if original_data_bfloat16.dtype != torch.bfloat16:
        original_data_bfloat16 = original_data_bfloat16.to(torch.bfloat16)
    if original_data_bfloat16.dim() != 4:
        raise ValueError(f'Expected 4D [batch, kv_heads, seq, head_dim] tensor, got {tuple(original_data_bfloat16.shape)}')
    batch_size, num_kv_heads, seq_len, head_dim = original_data_bfloat16.shape
    if batch_size != 1:
        raise ValueError(f'Open tiled rows currently support batch_size == 1, got {batch_size}')
    if seq_len > seq_tile_size:
        raise ValueError(f'Open tiled rows expect seq_len <= seq_tile_size, got {seq_len} vs {seq_tile_size}')

    dense_reference = original_data_bfloat16.contiguous()
    left = right - 2**nbits + 1
    tile_num_lanes = head_dim // 8
    raw_u16 = dense_reference.view(torch.uint16)
    raw_i32 = raw_u16.to(torch.int32)
    sign_mass_dense = ((((raw_i32 >> 8) & 0x80) | (raw_i32 & 0x7F)).to(torch.uint8))
    exp_dense = (((raw_i32 >> 7) & 0xFF).to(torch.uint8))
    sign_mass_rows = sign_mass_dense[0].permute(1, 0, 2).reshape(seq_len * num_kv_heads, head_dim).contiguous()
    exp_rows = exp_dense[0].permute(1, 0, 2).reshape(seq_len * num_kv_heads, head_dim)
    invalid_mask = (exp_rows <= left) | (exp_rows > right)
    exp_code_rows = torch.where(invalid_mask, torch.zeros_like(exp_rows), exp_rows - left).contiguous()
    invalid_exp = exp_rows[invalid_mask].contiguous()
    invalid_exp_row_offsets = torch.zeros((exp_rows.shape[0] + 1,), dtype=torch.int32, device=original_data_bfloat16.device)
    if exp_rows.shape[0] > 0:
        invalid_exp_row_offsets[1:] = torch.cumsum(invalid_mask.sum(dim=-1, dtype=torch.int32), dim=0)
    extracted_hist = None
    if need_adjust:
        extracted_hist = torch.bincount(exp_dense.reshape(-1).to(torch.int64), minlength=256)
    open_rows = OpenTiledCompressionRows(
        sign_mass_rows=sign_mass_rows,
        exp_code_rows=exp_code_rows,
        invalid_exp=invalid_exp,
        invalid_exp_row_offsets=invalid_exp_row_offsets,
        left=left,
        right=right,
        nbits=nbits,
        original_shape=tuple(original_data_bfloat16.shape),
        seq_tile_size=seq_tile_size,
        head_dim=head_dim,
        tile_head_dim=tile_head_dim,
        tile_num_lanes=tile_num_lanes,
        num_kv_heads=num_kv_heads,
        device=original_data_bfloat16.device,
    )
    return open_rows, extracted_hist


def _build_tiled_tensor(
    original_data_bfloat16: torch.Tensor,
    right: int,
    nbits: int,
    seq_tile_size: int,
    tile_head_dim: int,
    need_adjust: bool,
) -> tuple[TiledCompressedTensor, Optional[torch.Tensor]]:
    legacy_compressor = CompressorBitMap(right=right, nbits=nbits)
    extracted_hist = legacy_compressor.compress(original_data_bfloat16, need_adjust=need_adjust)

    batch_size, num_kv_heads, seq_len, head_dim = original_data_bfloat16.shape
    num_sequence_tiles = (seq_len + seq_tile_size - 1) // seq_tile_size
    tile_num_lanes = head_dim // 8

    # Reuse the already-packed legacy compressor buffers directly. In this
    # block-compression path we compress exactly one closed block at a time, so the
    # flattened BHSD layout is already tile-major for tiled decode:
    #   [head0, seq[0:block), dim] [head1, ...] ...
    # This avoids rebuilding sign/exp/outlier metadata from dense bf16 with eager
    # torch ops on every flush.
    sign_mass_data = legacy_compressor.sign_mass_data.contiguous()
    exp_data = legacy_compressor.exp_data.contiguous()
    invalid_exp = legacy_compressor.invalid_exp.contiguous()
    left = legacy_compressor.left

    tile_elements = seq_tile_size * head_dim
    invalid_block_size = int(legacy_compressor.invalid_block_size)
    total_tiles = batch_size * num_kv_heads * num_sequence_tiles

    if tile_elements % invalid_block_size != 0:
        raise ValueError(
            'Tiled compressed path requires tile_elements to align with legacy invalid block size, '
            f'got tile_elements={tile_elements}, invalid_block_size={invalid_block_size}'
        )

    blocks_per_tile = tile_elements // invalid_block_size
    expected_block_count = total_tiles * blocks_per_tile
    if legacy_compressor.block_base.numel() != expected_block_count:
        raise ValueError(
            'Legacy compressor block_base size does not match tiled layout expectation: '
            f'got {legacy_compressor.block_base.numel()}, expected {expected_block_count}'
        )

    block_base_2d = legacy_compressor.block_base.view(total_tiles, blocks_per_tile)
    tile_invalid_base_tensor = block_base_2d[:, -1].to(torch.int32).contiguous()

    compressed = TiledCompressedTensor(
        sign_mass_data=sign_mass_data,
        exp_data=exp_data,
        invalid_exp=invalid_exp,
        tile_invalid_base=tile_invalid_base_tensor,
        left=left,
        right=right,
        nbits=nbits,
        original_shape=tuple(original_data_bfloat16.shape),
        seq_tile_size=seq_tile_size,
        head_dim=head_dim,
        tile_head_dim=tile_head_dim,
        tile_num_lanes=tile_num_lanes,
        num_sequence_tiles=num_sequence_tiles,
        num_kv_heads=num_kv_heads,
        device=original_data_bfloat16.device,
    )
    return compressed, extracted_hist


def compress_tiled_tensor(
    original_data_bfloat16: torch.Tensor,
    right: int,
    nbits: int = _DEFAULT_TILE_NBITS,
    seq_tile_size: int = 128,
    tile_head_dim: Optional[int] = None,
    need_adjust: bool = False,
) -> tuple[TiledCompressedTensor, Optional[torch.Tensor]]:
    if nbits not in _SUPPORTED_TILE_NBITS:
        raise ValueError(f'Tiled compressed path supports nbits in {_SUPPORTED_TILE_NBITS}, got {nbits}')
    if original_data_bfloat16.dtype != torch.bfloat16:
        original_data_bfloat16 = original_data_bfloat16.to(torch.bfloat16)
    if original_data_bfloat16.dim() != 4:
        raise ValueError(f'Expected 4D [batch, kv_heads, seq, head_dim] tensor, got {tuple(original_data_bfloat16.shape)}')

    batch_size, num_kv_heads, seq_len, head_dim = original_data_bfloat16.shape
    if seq_tile_size <= 0:
        raise ValueError('seq_tile_size must be positive')
    if tile_head_dim is None:
        tile_head_dim = head_dim
    if tile_head_dim != head_dim:
        raise ValueError(f'Tiled compressed path currently requires tile_head_dim == head_dim, got {tile_head_dim} vs {head_dim}')
    if head_dim % 8 != 0:
        raise ValueError(f'Tiled compressed path requires head_dim % 8 == 0, got {head_dim}')
    if seq_tile_size != seq_len:
        raise ValueError(
            f'Tiled compressed path currently requires seq_tile_size == seq_len per compressed block, '
            f'got seq_tile_size={seq_tile_size}, seq_len={seq_len}'
        )
    if seq_tile_size % 1 != 0:
        raise ValueError('seq_tile_size must be integral')

    return _build_tiled_tensor(
        original_data_bfloat16=original_data_bfloat16,
        right=right,
        nbits=nbits,
        seq_tile_size=seq_tile_size,
        tile_head_dim=tile_head_dim,
        need_adjust=need_adjust,
    )


def seal_open_tiled_rows(open_rows: OpenTiledCompressionRows) -> TiledCompressedTensor:
    token_count = open_rows.token_count
    seq_tile_size = open_rows.seq_tile_size
    head_dim = open_rows.head_dim
    num_kv_heads = open_rows.num_kv_heads
    device = open_rows.device

    # --- sign_mass and exp_code: token-major [T*H, D] -> head-major [H, seq_tile_size, D] ---
    # view as [T, H, D], permute to [H, T, D], then pad seq dim to seq_tile_size with zeros.
    sign_mass_data = torch.zeros((num_kv_heads, seq_tile_size, head_dim), dtype=torch.uint8, device=device)
    exp_code_data  = torch.zeros((num_kv_heads, seq_tile_size, head_dim), dtype=torch.uint8, device=device)
    if token_count > 0:
        # [T*H, D] -> [T, H, D] -> [H, T, D]
        sign_mass_data[:, :token_count, :] = open_rows.sign_mass_rows[:token_count * num_kv_heads].view(token_count, num_kv_heads, head_dim).permute(1, 0, 2)
        exp_code_data[:, :token_count, :]  = open_rows.exp_code_rows[:token_count * num_kv_heads].view(token_count, num_kv_heads, head_dim).permute(1, 0, 2)

    # --- invalid_exp: reorder from token-major to head-major, all on GPU ---
    # row_offsets layout: [T*H+1], row index = token_idx * H + kv_head (token-major).
    # We need to gather invalid_exp entries in head-major order:
    #   head 0: rows 0, H, 2H, ...  (token 0..T-1, head 0)
    #   head 1: rows 1, H+1, 2H+1, ...
    #   ...
    # Build a permutation of row indices in head-major order, then use it to
    # reindex row_offsets and scatter-gather invalid_exp — zero Python loops.
    n_rows = token_count * num_kv_heads
    if n_rows > 0:
        row_offsets = open_rows.invalid_exp_row_offsets[:n_rows + 1].to(torch.int64)
        row_counts = row_offsets[1:] - row_offsets[:-1]  # [T*H]

        # head-major permutation of row indices: for head h, rows are h, h+H, h+2H, ...
        # shape [H, T] -> flatten -> [H*T] = [T*H] in head-major order
        token_idx = torch.arange(token_count, device=device)
        head_idx  = torch.arange(num_kv_heads, device=device)
        # row_index[h, t] = t * H + h  (token-major row index for head h, token t)
        hm_row_indices = (token_idx.unsqueeze(0) * num_kv_heads + head_idx.unsqueeze(1)).reshape(-1)  # [H*T]

        # Counts and cumulative base in head-major order
        hm_counts = row_counts[hm_row_indices]           # [H*T]
        hm_offsets_src = row_offsets[:-1][hm_row_indices]  # source start in invalid_exp for each row

        # Build a flat index into invalid_exp for every output element.
        # For each row r in head-major order, output positions are:
        #   cumsum(hm_counts)[r-1] .. cumsum(hm_counts)[r]-1
        # Source positions are: hm_offsets_src[r] .. hm_offsets_src[r] + hm_counts[r] - 1
        total_invalids = int(hm_counts.sum().item())
        if total_invalids > 0:
            # Expand each row's source offsets into per-element source indices using
            # a repeat_interleave + cumulative correction — no Python loop needed.
            row_starts_out = torch.zeros(len(hm_counts) + 1, dtype=torch.int64, device=device)
            row_starts_out[1:] = torch.cumsum(hm_counts, dim=0)

            # For each output element, which row does it belong to?
            # Use repeat_interleave to replicate hm_offsets_src by hm_counts.
            src_base = torch.repeat_interleave(hm_offsets_src, hm_counts)  # [total_invalids]
            # Within-row local offset: position within each row's block
            local_offset = torch.arange(total_invalids, device=device) - torch.repeat_interleave(row_starts_out[:-1], hm_counts)
            src_indices = src_base + local_offset
            invalid_exp = open_rows.invalid_exp[src_indices]
        else:
            invalid_exp = torch.empty((0,), dtype=torch.uint8, device=device)

        # tile_invalid_base: cumulative invalid count per head (head-major, globally cumulative)
        # hm_counts is [H*T], sum per head = hm_counts.view(H, T).sum(dim=1)
        per_head_counts = hm_counts.view(num_kv_heads, token_count).sum(dim=1)  # [H]
        tile_invalid_base = torch.cumsum(per_head_counts, dim=0).to(torch.int32)
    else:
        invalid_exp = torch.empty((0,), dtype=torch.uint8, device=device)
        tile_invalid_base = torch.zeros((num_kv_heads,), dtype=torch.int32, device=device)

    return TiledCompressedTensor(
        sign_mass_data=sign_mass_data.reshape(-1),
        exp_data=_pack_nbit_groups(exp_code_data.reshape(-1), open_rows.nbits),
        invalid_exp=invalid_exp,
        tile_invalid_base=tile_invalid_base,
        left=open_rows.left,
        right=open_rows.right,
        nbits=open_rows.nbits,
        original_shape=(1, num_kv_heads, seq_tile_size, head_dim),
        seq_tile_size=seq_tile_size,
        head_dim=head_dim,
        tile_head_dim=open_rows.tile_head_dim,
        tile_num_lanes=open_rows.tile_num_lanes,
        num_sequence_tiles=1,
        num_kv_heads=num_kv_heads,
        device=device,
    )


def compress_open_tiled_rows(
    original_data_bfloat16: torch.Tensor,
    right: int,
    nbits: int = _DEFAULT_TILE_NBITS,
    seq_tile_size: int = 128,
    tile_head_dim: Optional[int] = None,
    need_adjust: bool = False,
) -> tuple[OpenTiledCompressionRows, Optional[torch.Tensor]]:
    if nbits not in _SUPPORTED_TILE_NBITS:
        raise ValueError(f'Tiled compressed path supports nbits in {_SUPPORTED_TILE_NBITS}, got {nbits}')
    if original_data_bfloat16.dtype != torch.bfloat16:
        original_data_bfloat16 = original_data_bfloat16.to(torch.bfloat16)
    if original_data_bfloat16.dim() != 4:
        raise ValueError(f'Expected 4D [batch, kv_heads, seq, head_dim] tensor, got {tuple(original_data_bfloat16.shape)}')
    batch_size, _, _, head_dim = original_data_bfloat16.shape
    if seq_tile_size <= 0:
        raise ValueError('seq_tile_size must be positive')
    if tile_head_dim is None:
        tile_head_dim = head_dim
    if tile_head_dim != head_dim:
        raise ValueError(f'Tiled compressed path currently requires tile_head_dim == head_dim, got {tile_head_dim} vs {head_dim}')
    if head_dim % 8 != 0:
        raise ValueError(f'Tiled compressed path requires head_dim % 8 == 0, got {head_dim}')
    return _build_open_tiled_rows(
        original_data_bfloat16=original_data_bfloat16,
        right=right,
        nbits=nbits,
        seq_tile_size=seq_tile_size,
        tile_head_dim=tile_head_dim,
        need_adjust=need_adjust,
    )


def materialize_tiled_state(state: dict[str, Any]) -> torch.Tensor:
    if state.get('layout') == 'tiled_open_rows':
        return OpenTiledCompressionRows(
            sign_mass_rows=state['sign_mass_rows'],
            exp_code_rows=state['exp_code_rows'],
            invalid_exp=state['invalid_exp'],
            invalid_exp_row_offsets=state['invalid_exp_row_offsets'],
            left=state['left'],
            right=state['right'],
            nbits=state['nbits'],
            original_shape=tuple(state['original_shape']),
            seq_tile_size=state['seq_tile_size'],
            head_dim=state['head_dim'],
            tile_head_dim=state['tile_head_dim'],
            tile_num_lanes=state['tile_num_lanes'],
            num_kv_heads=state['num_kv_heads'],
            device=state['device'],
        ).materialize()
    seq_tile_size = state['seq_tile_size']
    block_seq_len = tuple(state['original_shape'])[-2]
    num_sequence_tiles = (block_seq_len + seq_tile_size - 1) // seq_tile_size
    return TiledCompressedTensor(
        sign_mass_data=state['sign_mass_data'],
        exp_data=state['exp_data'],
        invalid_exp=state.get('invalid_exp', torch.empty((0,), dtype=torch.uint8, device=state['device'])),
        tile_invalid_base=state.get('tile_invalid_base', torch.empty((0,), dtype=torch.int32, device=state['device'])),
        left=state['left'],
        right=state['right'],
        nbits=state['nbits'],
        original_shape=tuple(state['original_shape']),
        seq_tile_size=seq_tile_size,
        head_dim=state['head_dim'],
        tile_head_dim=state['tile_head_dim'],
        tile_num_lanes=state['tile_num_lanes'],
        num_sequence_tiles=num_sequence_tiles,
        num_kv_heads=state['num_kv_heads'],
        device=state['device'],
    ).materialize()
