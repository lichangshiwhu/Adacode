from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional, Any
import concurrent.futures
import time

import torch
from torch.profiler import record_function
from transformers.cache_utils import DynamicLayer
from transformers import Cache

from kv_compression.compress_bitmap_merge import CompressorBitMap as Compressor
from kv_compression.compress_tiled_kv import (
    OpenTiledCompressionRows,
    TiledCompressedTensor,
    compress_open_tiled_rows,
    compress_tiled_tensor,
    materialize_tiled_state,
    seal_open_tiled_rows,
)


def _export_unified_tiled_state(compressed_tensor: Optional[TiledCompressedTensor], open_tile: Optional[OpenTiledCompressionRows] = None) -> Optional[dict[str, Any]]:
    if compressed_tensor is None and open_tile is None:
        return None
    state = compressed_tensor.export_state() if compressed_tensor is not None else None
    if state is None:
        state = {
            'layout': 'decode_native',
            'left': open_tile.left,
            'right': open_tile.right,
            'nbits': open_tile.nbits,
            'device': open_tile.device,
            'original_shape': (1, open_tile.num_kv_heads, 0, open_tile.head_dim),
            'sign_mass_data': torch.empty((0,), dtype=torch.uint8, device=open_tile.device),
            'exp_data': torch.empty((0,), dtype=torch.uint8, device=open_tile.device),
            'invalid_exp': torch.empty((0,), dtype=torch.uint8, device=open_tile.device),
            'tile_invalid_base': torch.empty((0,), dtype=torch.int32, device=open_tile.device),
            'seq_tile_size': open_tile.seq_tile_size,
            'head_dim': open_tile.head_dim,
            'tile_head_dim': open_tile.tile_head_dim,
            'tile_num_lanes': open_tile.tile_num_lanes,
            'num_kv_heads': open_tile.num_kv_heads,
            'num_sequence_tiles': 0,
            'n_elements': 0,
            'has_outliers': True,
        }
    state['kind'] = 'decode_native'
    state['closed_seq_len'] = int(state.get('original_shape', (1, 1, 0, 1))[-2])
    state['open_tile'] = None if open_tile is None else open_tile.export_state()
    state['open_tile_token_count'] = 0 if open_tile is None else open_tile.token_count
    state['logical_seq_len_total'] = state['closed_seq_len'] + state['open_tile_token_count']
    state['layout'] = 'unified_tiled_segments'
    return state


def _update_unified_tiled_state_open_tile(state: dict[str, Any], open_tile: Optional[OpenTiledCompressionRows]) -> None:
    """Update only the open-tile fields of an existing unified state dict in-place.

    This avoids replacing the dict object (which would break id()-based descriptor
    caching in kv_attention) while still reflecting the new open tile after
    each decode step.
    """
    open_tile_state = None if open_tile is None else open_tile.export_state()
    token_count = 0 if open_tile is None else open_tile.token_count
    state['open_tile'] = open_tile_state
    state['open_tile_token_count'] = token_count
    state['logical_seq_len_total'] = state['closed_seq_len'] + token_count


def _export_segmented_tiled_state(segment_states: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not segment_states:
        return None
    first = segment_states[0]
    batch_size, num_kv_heads, _, head_dim = tuple(first['original_shape'])
    total_seq_len = sum(int(tuple(state['original_shape'])[-2]) for state in segment_states)
    total_tiles = sum(int(state.get('num_sequence_tiles', 0)) for state in segment_states)
    return {
        'layout': 'segmented_tiled',
        'kind': 'segmented_tiled',
        'segments': list(segment_states),
        'num_segments': len(segment_states),
        'segment_seq_lens': [int(tuple(state['original_shape'])[-2]) for state in segment_states],
        'logical_seq_len_total': total_seq_len,
        'original_shape': (batch_size, num_kv_heads, total_seq_len, head_dim),
        'seq_tile_size': first['seq_tile_size'],
        'head_dim': first['head_dim'],
        'tile_head_dim': first['tile_head_dim'],
        'tile_num_lanes': first['tile_num_lanes'],
        'num_kv_heads': first['num_kv_heads'],
        'num_sequence_tiles': total_tiles,
        'device': first['device'],
        'nbits': first['nbits'],
        'left': first['left'],
        'right': first['right'],
    }


def _tile_counts_from_state(
    state: dict[str, Any],
    batch_size: int,
    num_kv_heads: int,
) -> torch.Tensor:
    num_tiles = int(state.get('num_sequence_tiles', 0))
    tile_invalid_base = state.get('tile_invalid_base')
    if tile_invalid_base is None or tile_invalid_base.numel() == 0:
        return torch.zeros(
            (batch_size * num_kv_heads * num_tiles,),
            dtype=torch.int32,
            device=state['device'],
        )
    counts = tile_invalid_base.to(torch.int32).clone()
    if counts.numel() > 1:
        counts[1:] = counts[1:] - counts[:-1]
    return counts


def _merge_tiled_segment_states(segment_states: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not segment_states:
        return None
    if len(segment_states) == 1:
        return dict(segment_states[0])

    first = segment_states[0]
    batch_size, num_kv_heads, _, head_dim = tuple(first['original_shape'])
    seq_tile_size = int(first['seq_tile_size'])
    tile_num_lanes = int(first['tile_num_lanes'])
    tile_sign_mass_elems = seq_tile_size * head_dim
    nbits = int(first['nbits'])
    tile_exp_elems = seq_tile_size * tile_num_lanes * nbits
    num_bh_groups = batch_size * num_kv_heads

    sign_mass_parts = []
    exp_parts = []
    counts_parts = []
    invalid_parts_by_group: list[list[torch.Tensor]] = [[] for _ in range(num_bh_groups)]
    total_seq_len = 0
    total_tiles = 0
    total_elements = 0

    for idx, state in enumerate(segment_states):
        state_shape = tuple(state['original_shape'])
        state_batch, state_kv_heads, state_seq_len, state_head_dim = state_shape
        if (state_batch, state_kv_heads, state_head_dim) != (batch_size, num_kv_heads, head_dim):
            raise ValueError(
                'Segmented tiled merge requires matching batch/kv_heads/head_dim, '
                f'got first={first["original_shape"]}, segment[{idx}]={state_shape}'
            )
        if int(state['seq_tile_size']) != seq_tile_size:
            raise ValueError(
                'Segmented tiled merge requires matching seq_tile_size, '
                f'got first={seq_tile_size}, segment[{idx}]={state["seq_tile_size"]}'
            )
        if int(state['tile_num_lanes']) != tile_num_lanes:
            raise ValueError(
                'Segmented tiled merge requires matching tile_num_lanes, '
                f'got first={tile_num_lanes}, segment[{idx}]={state["tile_num_lanes"]}'
            )
        if int(state['nbits']) != nbits:
            raise ValueError(
                'Segmented tiled merge requires matching nbits, '
                f'got first={nbits}, segment[{idx}]={state["nbits"]}'
            )

        num_tiles = int(state.get('num_sequence_tiles', 0))
        sign_mass_parts.append(
            state['sign_mass_data'].view(batch_size, num_kv_heads, num_tiles, tile_sign_mass_elems)
        )
        exp_parts.append(
            state['exp_data'].view(batch_size, num_kv_heads, num_tiles, tile_exp_elems)
        )

        counts_flat = _tile_counts_from_state(state, batch_size, num_kv_heads)
        counts_3d = counts_flat.view(batch_size, num_kv_heads, num_tiles)
        counts_parts.append(counts_3d)

        invalid_exp = state.get('invalid_exp')
        group_totals = counts_3d.sum(dim=2).reshape(-1)
        group_total_list = [int(v) for v in group_totals.tolist()]
        if invalid_exp is None or invalid_exp.numel() == 0:
            group_chunks = [
                torch.empty((0,), dtype=torch.uint8, device=state['device'])
                for _ in range(num_bh_groups)
            ]
        else:
            group_chunks = []
            group_base = 0
            for group_total in group_total_list:
                next_group_base = group_base + group_total
                group_chunks.append(invalid_exp[group_base:next_group_base])
                group_base = next_group_base
        for group_idx, chunk in enumerate(group_chunks):
            if chunk.numel() > 0:
                invalid_parts_by_group[group_idx].append(chunk)

        total_seq_len += int(state_seq_len)
        total_tiles += num_tiles
        total_elements += int(state.get('n_elements', 0))

    merged_sign_mass = torch.cat(sign_mass_parts, dim=2).contiguous().view(-1)
    merged_exp_data = torch.cat(exp_parts, dim=2).contiguous().view(-1)
    merged_counts = torch.cat(counts_parts, dim=2).contiguous().view(-1)
    merged_tile_invalid_base = (
        torch.cumsum(merged_counts, dim=0).to(torch.int32)
        if merged_counts.numel() > 0
        else torch.empty((0,), dtype=torch.int32, device=first['device'])
    )

    merged_invalid_groups = []
    for group_parts in invalid_parts_by_group:
        if not group_parts:
            continue
        merged_invalid_groups.append(
            group_parts[0] if len(group_parts) == 1 else torch.cat(group_parts, dim=0).contiguous()
        )
    merged_invalid_exp = (
        torch.cat(merged_invalid_groups, dim=0).contiguous()
        if merged_invalid_groups
        else torch.empty((0,), dtype=torch.uint8, device=first['device'])
    )

    merged_state = dict(first)
    merged_state['sign_mass_data'] = merged_sign_mass
    merged_state['exp_data'] = merged_exp_data
    merged_state['invalid_exp'] = merged_invalid_exp
    merged_state['tile_invalid_base'] = merged_tile_invalid_base
    merged_state['original_shape'] = (batch_size, num_kv_heads, total_seq_len, head_dim)
    merged_state['num_sequence_tiles'] = total_tiles
    merged_state['n_elements'] = total_elements
    return merged_state


def _concat_seq_tensors(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return torch.cat([first, second], dim=-2).contiguous()


def _append_flat_tensor(existing: Optional[torch.Tensor], new_part: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if new_part is None:
        return existing
    if existing is None:
        return new_part.contiguous()
    return torch.cat([existing, new_part], dim=0).contiguous()


def _append_seq_dim_tensor(existing: Optional[torch.Tensor], new_part: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if new_part is None:
        return existing
    if existing is None:
        return new_part.contiguous()
    return torch.cat([existing, new_part], dim=-2).contiguous()


def _append_legacy_prefix_state(existing: Optional[dict[str, Any]], new_state: dict[str, Any]) -> dict[str, Any]:
    if existing is None:
        return dict(new_state)
    for field in ('sign_mass_data', 'exp_data', 'invalid_exp'):
        existing[field] = _append_flat_tensor(existing.get(field), new_state.get(field))
    existing_block_base = existing.get('block_base')
    new_block_base = new_state.get('block_base')
    if new_block_base is None:
        pass
    elif existing_block_base is None or existing_block_base.numel() == 0:
        existing['block_base'] = new_block_base.contiguous()
    else:
        invalid_offset = int(existing_block_base[-1].item())
        existing['block_base'] = torch.cat([existing_block_base, new_block_base + invalid_offset], dim=0).contiguous()
    original_shape = tuple(existing['original_shape'])
    new_shape = tuple(new_state['original_shape'])
    existing['original_shape'] = (*original_shape[:-2], original_shape[-2] + new_shape[-2], original_shape[-1])
    existing['n_elements'] = int(existing.get('n_elements', 0)) + int(new_state.get('n_elements', 0))
    existing['n_exp_data'] = int(existing.get('n_exp_data', 0)) + int(new_state.get('n_exp_data', 0))
    return existing


def _append_tiled_prefix_state(existing: Optional[dict[str, Any]], new_state: dict[str, Any]) -> dict[str, Any]:
    if existing is None:
        result = dict(new_state)
        result['num_sequence_tiles'] = int(new_state.get('num_sequence_tiles', 0))
        return result
    original_shape = tuple(existing['original_shape'])
    new_shape = tuple(new_state['original_shape'])
    if original_shape[:2] != new_shape[:2] or original_shape[-1] != new_shape[-1]:
        raise ValueError(
            'Tiled prefix append requires matching batch/kv_heads/head_dim, '
            f'got existing={original_shape}, new={new_shape}'
        )

    batch_size, num_kv_heads, _, head_dim = original_shape
    seq_tile_size = int(existing['seq_tile_size'])
    if seq_tile_size != int(new_state['seq_tile_size']):
        raise ValueError(
            'Tiled prefix append requires matching seq_tile_size, '
            f'got existing={seq_tile_size}, new={new_state["seq_tile_size"]}'
        )
    tile_num_lanes = int(existing['tile_num_lanes'])
    if tile_num_lanes != int(new_state['tile_num_lanes']):
        raise ValueError(
            'Tiled prefix append requires matching tile_num_lanes, '
            f'got existing={tile_num_lanes}, new={new_state["tile_num_lanes"]}'
        )
    nbits = int(existing['nbits'])
    if nbits != int(new_state['nbits']):
        raise ValueError(
            'Tiled prefix append requires matching nbits, '
            f'got existing={nbits}, new={new_state["nbits"]}'
        )

    existing_tiles = int(existing.get('num_sequence_tiles', 0))
    new_tiles = int(new_state.get('num_sequence_tiles', 0))
    tile_sign_mass_elems = seq_tile_size * head_dim
    tile_exp_elems = seq_tile_size * tile_num_lanes * nbits

    existing['sign_mass_data'] = torch.cat(
        [
            existing['sign_mass_data'].view(batch_size, num_kv_heads, existing_tiles, tile_sign_mass_elems),
            new_state['sign_mass_data'].view(batch_size, num_kv_heads, new_tiles, tile_sign_mass_elems),
        ],
        dim=2,
    ).contiguous().view(-1)
    existing['exp_data'] = torch.cat(
        [
            existing['exp_data'].view(batch_size, num_kv_heads, existing_tiles, tile_exp_elems),
            new_state['exp_data'].view(batch_size, num_kv_heads, new_tiles, tile_exp_elems),
        ],
        dim=2,
    ).contiguous().view(-1)

    def _tile_counts(state: dict[str, Any]) -> torch.Tensor:
        tile_invalid_base = state.get('tile_invalid_base')
        if tile_invalid_base is None or tile_invalid_base.numel() == 0:
            return torch.zeros((batch_size * num_kv_heads * int(state.get('num_sequence_tiles', 0)),), dtype=torch.int32, device=state['device'])
        counts = tile_invalid_base.to(torch.int32).clone()
        if counts.numel() > 1:
            counts[1:] = counts[1:] - counts[:-1]
        return counts

    existing_counts_flat = _tile_counts(existing)
    new_counts_flat = _tile_counts(new_state)
    existing_counts = existing_counts_flat.view(batch_size, num_kv_heads, existing_tiles)
    new_counts = new_counts_flat.view(batch_size, num_kv_heads, new_tiles)
    merged_counts = torch.cat([existing_counts, new_counts], dim=2).contiguous().view(-1)
    existing['tile_invalid_base'] = torch.cumsum(merged_counts, dim=0).to(torch.int32)

    def _split_invalid_exp(state: dict[str, Any], counts: torch.Tensor) -> list[torch.Tensor]:
        invalid_exp = state.get('invalid_exp')
        if invalid_exp is None:
            return []
        count_list = [int(v) for v in counts.tolist()]
        if not count_list:
            return []
        return list(torch.split(invalid_exp, count_list))

    existing_invalid_chunks = _split_invalid_exp(existing, existing_counts_flat)
    new_invalid_chunks = _split_invalid_exp(new_state, new_counts_flat)
    merged_invalid_chunks = []
    for batch_idx in range(batch_size):
        for kv_head_idx in range(num_kv_heads):
            base_existing = (batch_idx * num_kv_heads + kv_head_idx) * existing_tiles
            base_new = (batch_idx * num_kv_heads + kv_head_idx) * new_tiles
            merged_invalid_chunks.extend(existing_invalid_chunks[base_existing : base_existing + existing_tiles])
            merged_invalid_chunks.extend(new_invalid_chunks[base_new : base_new + new_tiles])
    existing['invalid_exp'] = torch.cat(merged_invalid_chunks, dim=0).contiguous() if merged_invalid_chunks else torch.empty((0,), dtype=torch.uint8, device=existing['device'])

    existing['original_shape'] = (*original_shape[:-2], original_shape[-2] + new_shape[-2], original_shape[-1])
    existing['n_elements'] = int(existing.get('n_elements', 0)) + int(new_state.get('n_elements', 0))
    existing['num_sequence_tiles'] = existing_tiles + new_tiles
    return existing

_ADJUST_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=8)


def analysis_best_bits(data_freq, store_index=False):
    data_freq_list = data_freq.tolist()
    total_numbers = sum(data_freq_list)
    original_memory = total_numbers * 8

    indexed = sorted(enumerate(data_freq_list), key=lambda x: x[1], reverse=True)

    best_compressed_memory = original_memory
    best_indices = []
    best_nbit = 0

    for nbit in [2, 3, 4]:
        number_space = 2**nbit - 1
        topk = indexed[:number_space]
        total_norm_values = sum(v for _, v in topk)
        outlier_values = total_numbers - total_norm_values
        compressed_memory = total_numbers * nbit + outlier_values * 8
        if store_index:
            compressed_memory += outlier_values * 32
        if compressed_memory < best_compressed_memory:
            best_compressed_memory = compressed_memory
            best_indices = [i for i, _ in topk]
            best_nbit = nbit

    best_indices.sort()
    return best_indices[-1], best_nbit


@dataclass
class Float11CompressedBlock:
    key_compressor: Compressor
    value_compressor: Compressor
    seq_len: int
    layout: str = 'legacy'

    @property
    def device(self):
        return self.key_compressor.device

    def get_memory(self):
        keys_mem = self.key_compressor.get_memory()
        values_mem = self.value_compressor.get_memory()
        return {
            'compressed_memory': keys_mem['compressed_memory'] + values_mem['compressed_memory'],
            'original_memory': keys_mem['original_memory'] + values_mem['original_memory'],
            'keys_outlier_ratios': self._outlier_ratio(self.key_compressor),
            'values_outlier_ratios': self._outlier_ratio(self.value_compressor),
        }

    def materialize(self):
        return self.key_compressor.materialize(), self.value_compressor.materialize()

    def export_metadata(self):
        key_state = self.key_compressor.export_state()
        value_state = self.value_compressor.export_state()
        return {
            'layout': self.layout,
            'seq_len': self.seq_len,
            'device': self.device,
            'key': key_state,
            'value': value_state,
            'key_decode': Compressor.export_decode_view_from_state(key_state),
            'value_decode': Compressor.export_decode_view_from_state(value_state),
        }

    @staticmethod
    def _outlier_ratio(compressor):
        if compressor.invalid_exp is None or compressor.n_elements in (None, 0):
            return 0.0
        return compressor.invalid_exp.numel() / compressor.n_elements


@dataclass
class Float11TiledCompressedBlock:
    key_tensor: TiledCompressedTensor
    value_tensor: TiledCompressedTensor
    seq_len: int
    layout: str = 'tiled'

    @property
    def device(self):
        return self.key_tensor.device

    def get_memory(self):
        keys_mem = self.key_tensor.get_memory()
        values_mem = self.value_tensor.get_memory()
        return {
            'compressed_memory': keys_mem['compressed_memory'] + values_mem['compressed_memory'],
            'original_memory': keys_mem['original_memory'] + values_mem['original_memory'],
            'keys_outlier_ratios': self._outlier_ratio(self.key_tensor),
            'values_outlier_ratios': self._outlier_ratio(self.value_tensor),
        }

    def materialize(self):
        return self.key_tensor.materialize(), self.value_tensor.materialize()

    def export_metadata(self):
        return {
            'layout': self.layout,
            'seq_len': self.seq_len,
            'device': self.device,
            'key': self.key_tensor.export_state(),
            'value': self.value_tensor.export_state(),
            'key_decode': self.key_tensor.export_decode_view(),
            'value_decode': self.value_tensor.export_decode_view(),
        }

    @staticmethod
    def _outlier_ratio(compressed_tensor: TiledCompressedTensor):
        n_elements = compressed_tensor.n_elements
        if n_elements == 0:
            return 0.0
        return compressed_tensor.invalid_exp.numel() / n_elements


class float11Layer(DynamicLayer):
    def __init__(
        self,
        keys_right: int = 0,
        values_right: int = 0,
        keys_nbits: int = 3,
        values_nbits: int = 3,
        residual_length: int = 0,
        block_size: int = 128,
        layout: str = 'legacy',
        seq_tile_size: Optional[int] = None,
    ):
        super().__init__()
        self.residual_length = residual_length
        self.block_size = block_size
        self.layout = layout
        self.seq_tile_size = seq_tile_size if seq_tile_size is not None else block_size
        self.keys_right, self.keys_nbits = keys_right, keys_nbits
        self.values_right, self.values_nbits = values_right, values_nbits
        self.cumulative_length = 0
        self.keys_right, self.keys_nbits = keys_right, keys_nbits
        self.values_right, self.values_nbits = values_right, values_nbits
        self.new_keys_right, self.new_keys_nbits = keys_right, keys_nbits
        self.new_values_right, self.new_values_nbits = values_right, values_nbits
        self.keys_update_times = 0
        self.values_update_times = 0
        self.residual_keys: Optional[torch.Tensor] = None
        self.residual_values: Optional[torch.Tensor] = None
        self._unified_key_tensor: Optional[TiledCompressedTensor] = None
        self._unified_value_tensor: Optional[TiledCompressedTensor] = None
        self.cache_device = None
        self.cache_dtype = None
        self.materialize_calls = 0
        self.materialize_cache_hits = 0
        self.materialize_time = 0.0
        self.prefill_reuse_hits = 0
        self._compressed_prefix_key_state: Optional[dict[str, Any]] = None
        self._compressed_prefix_value_state: Optional[dict[str, Any]] = None
        self._compressed_prefix_key_metadata: Optional[dict[str, Any]] = None
        self._compressed_prefix_value_metadata: Optional[dict[str, Any]] = None
        self._compressed_prefix_key_segments: list[dict[str, Any]] = []
        self._compressed_prefix_value_segments: list[dict[str, Any]] = []
        self._compressed_prefix_key_segment_metadata: list[dict[str, Any]] = []
        self._compressed_prefix_value_segment_metadata: list[dict[str, Any]] = []
        self._compressed_prefix_key_blocks: list[dict[str, Any]] = []
        self._compressed_prefix_value_blocks: list[dict[str, Any]] = []
        self._compressed_prefix_seq_len = 0
        self._materialized_prefix_keys_cache: Optional[torch.Tensor] = None
        self._materialized_prefix_values_cache: Optional[torch.Tensor] = None
        self._materialized_prefix_seq_len = 0
        self._materialized_residual_len = 0
        self._materialized_residual_keys_ref: Optional[torch.Tensor] = None
        self._materialized_residual_values_ref: Optional[torch.Tensor] = None
        self._cache_view_static: Optional[dict[str, Any]] = None
        self._cache_view_runtime: Optional[dict[str, Any]] = None
        self._compressed_prefix_ready_event: Optional[torch.cuda.Event] = None
        self._compressed_prefix_ready_event_pending = False
        self._compressed_block_flushes = 0
        self._compressed_tokens_flushed = 0

    def __getstate__(self):
        state = self.__dict__.copy()
        event = state.get('_compressed_prefix_ready_event')
        if state.get('_compressed_prefix_ready_event_pending') and event is not None:
            event.synchronize()
        # CUDA events are runtime-only synchronization handles and cannot be pickled.
        state['_compressed_prefix_ready_event'] = None
        state['_compressed_prefix_ready_event_pending'] = False
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._compressed_prefix_ready_event = None
        self._compressed_prefix_ready_event_pending = False

    def adjust_compressor(self, keys_exp_count, values_exp_count):
        self.new_keys_right, self.new_keys_nbits = analysis_best_bits(keys_exp_count)
        self.new_values_right, self.new_values_nbits = analysis_best_bits(values_exp_count)

    def _active_residual_tensors(self) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self.residual_keys, self.residual_values

    def _set_active_residual_tensors(
        self,
        key_states: Optional[torch.Tensor],
        value_states: Optional[torch.Tensor],
    ):
        self.residual_keys = key_states
        self.residual_values = value_states

    def _append_residual(self, key_states: torch.Tensor, value_states: torch.Tensor):
        if key_states.dtype != torch.bfloat16:
            key_states = key_states.to(torch.bfloat16)
        if value_states.dtype != torch.bfloat16:
            value_states = value_states.to(torch.bfloat16)
        active_keys, active_values = self._active_residual_tensors()
        if active_keys is None:
            self._set_active_residual_tensors(key_states, value_states)
        else:
            self._set_active_residual_tensors(
                _concat_seq_tensors(active_keys, key_states),
                _concat_seq_tensors(active_values, value_states),
            )

    def _refresh_unified_compressed_state(self):
        if self.layout != 'tiled':
            return
        if self._compressed_prefix_key_state is not None:
            self._unified_key_tensor = TiledCompressedTensor(
                sign_mass_data=self._compressed_prefix_key_state['sign_mass_data'],
                exp_data=self._compressed_prefix_key_state['exp_data'],
                invalid_exp=self._compressed_prefix_key_state.get('invalid_exp', torch.empty((0,), dtype=torch.uint8, device=self._compressed_prefix_key_state['device'])),
                tile_invalid_base=self._compressed_prefix_key_state.get('tile_invalid_base', torch.empty((0,), dtype=torch.int32, device=self._compressed_prefix_key_state['device'])),
                left=self._compressed_prefix_key_state['left'],
                right=self._compressed_prefix_key_state['right'],
                nbits=self._compressed_prefix_key_state['nbits'],
                original_shape=tuple(self._compressed_prefix_key_state['original_shape']),
                seq_tile_size=self._compressed_prefix_key_state['seq_tile_size'],
                head_dim=self._compressed_prefix_key_state['head_dim'],
                tile_head_dim=self._compressed_prefix_key_state['tile_head_dim'],
                tile_num_lanes=self._compressed_prefix_key_state['tile_num_lanes'],
                num_sequence_tiles=int(self._compressed_prefix_key_state['num_sequence_tiles']),
                num_kv_heads=self._compressed_prefix_key_state['num_kv_heads'],
                device=self._compressed_prefix_key_state['device'],
            )
            self._unified_value_tensor = TiledCompressedTensor(
                sign_mass_data=self._compressed_prefix_value_state['sign_mass_data'],
                exp_data=self._compressed_prefix_value_state['exp_data'],
                invalid_exp=self._compressed_prefix_value_state.get('invalid_exp', torch.empty((0,), dtype=torch.uint8, device=self._compressed_prefix_value_state['device'])),
                tile_invalid_base=self._compressed_prefix_value_state.get('tile_invalid_base', torch.empty((0,), dtype=torch.int32, device=self._compressed_prefix_value_state['device'])),
                left=self._compressed_prefix_value_state['left'],
                right=self._compressed_prefix_value_state['right'],
                nbits=self._compressed_prefix_value_state['nbits'],
                original_shape=tuple(self._compressed_prefix_value_state['original_shape']),
                seq_tile_size=self._compressed_prefix_value_state['seq_tile_size'],
                head_dim=self._compressed_prefix_value_state['head_dim'],
                tile_head_dim=self._compressed_prefix_value_state['tile_head_dim'],
                tile_num_lanes=self._compressed_prefix_value_state['tile_num_lanes'],
                num_sequence_tiles=int(self._compressed_prefix_value_state['num_sequence_tiles']),
                num_kv_heads=self._compressed_prefix_value_state['num_kv_heads'],
                device=self._compressed_prefix_value_state['device'],
            )
        else:
            self._unified_key_tensor = None
            self._unified_value_tensor = None

    def _compress_span(self, key_states: torch.Tensor, value_states: torch.Tensor, need_adjust: bool):
        if self.layout in {'tiled', 'segmented_tiled'}:
            key_tensor, keys_exp_count = compress_tiled_tensor(
                key_states,
                right=self.keys_right,
                nbits=self.keys_nbits,
                seq_tile_size=self.seq_tile_size,
                need_adjust=need_adjust,
            )
            value_tensor, values_exp_count = compress_tiled_tensor(
                value_states,
                right=self.values_right,
                nbits=self.values_nbits,
                seq_tile_size=self.seq_tile_size,
                need_adjust=need_adjust,
            )
            return key_tensor.export_state(), value_tensor.export_state(), key_tensor.export_decode_view(), value_tensor.export_decode_view(), keys_exp_count, values_exp_count

        key_compressor, keys_exp_count = Compressor.compress_block(
            key_states,
            right=self.keys_right,
            nbits=self.keys_nbits,
            need_adjust=need_adjust,
        )
        value_compressor, values_exp_count = Compressor.compress_block(
            value_states,
            right=self.values_right,
            nbits=self.values_nbits,
            need_adjust=need_adjust,
        )
        key_state = key_compressor.export_state()
        value_state = value_compressor.export_state()
        return key_state, value_state, Compressor.export_decode_view_from_state(key_state), Compressor.export_decode_view_from_state(value_state), keys_exp_count, values_exp_count

    def _append_compressed_prefix_span(
        self,
        key_state: dict[str, Any],
        value_state: dict[str, Any],
        key_decode_state: dict[str, Any],
        value_decode_state: dict[str, Any],
        seq_len: int,
    ):
        if self.layout == 'segmented_tiled':
            self._compressed_prefix_key_segments.append(key_state)
            self._compressed_prefix_value_segments.append(value_state)
            self._compressed_prefix_key_segment_metadata.append(key_decode_state)
            self._compressed_prefix_value_segment_metadata.append(value_decode_state)
            self._compressed_prefix_key_state = _export_segmented_tiled_state(self._compressed_prefix_key_segments)
            self._compressed_prefix_value_state = _export_segmented_tiled_state(self._compressed_prefix_value_segments)
            self._compressed_prefix_key_metadata = _export_segmented_tiled_state(self._compressed_prefix_key_segment_metadata)
            self._compressed_prefix_value_metadata = _export_segmented_tiled_state(self._compressed_prefix_value_segment_metadata)
            flushed_seq_len = int(seq_len)
            self._compressed_prefix_seq_len += flushed_seq_len
            self._compressed_block_flushes += 1
            self._compressed_tokens_flushed += flushed_seq_len
            self._materialized_prefix_keys_cache = None
            self._materialized_prefix_values_cache = None
            self._materialized_prefix_seq_len = 0
            self._cache_view_static = None
            self._cache_view_runtime = None
            self._record_compressed_prefix_ready_event()
            return
        if self.layout != 'tiled':
            self._compressed_prefix_key_blocks.append(key_state)
            self._compressed_prefix_value_blocks.append(value_state)
            flushed_seq_len = int(seq_len)
            self._compressed_prefix_seq_len += flushed_seq_len
            self._compressed_block_flushes += 1
            self._compressed_tokens_flushed += flushed_seq_len
            self._materialized_prefix_keys_cache = None
            self._materialized_prefix_values_cache = None
            self._materialized_prefix_seq_len = 0
            self._cache_view_static = None
            self._cache_view_runtime = None
            self._record_compressed_prefix_ready_event()
            return
        append_state = _append_tiled_prefix_state if self.layout == 'tiled' else _append_legacy_prefix_state
        self._compressed_prefix_key_state = append_state(self._compressed_prefix_key_state, key_state)
        self._compressed_prefix_value_state = append_state(self._compressed_prefix_value_state, value_state)
        # append_state updates existing in-place (when not None) and returns the same object,
        # so id(_compressed_prefix_key_metadata) stays stable across flushes — enabling
        # descriptor cache hits in kv_attention which keys on id().
        self._compressed_prefix_key_metadata = append_state(self._compressed_prefix_key_metadata, key_decode_state)
        self._compressed_prefix_value_metadata = append_state(self._compressed_prefix_value_metadata, value_decode_state)
        flushed_seq_len = int(seq_len)
        self._compressed_prefix_seq_len += flushed_seq_len
        self._compressed_block_flushes += 1
        self._compressed_tokens_flushed += flushed_seq_len
        self._materialized_prefix_keys_cache = None
        self._materialized_prefix_values_cache = None
        self._materialized_prefix_seq_len = 0
        self._refresh_unified_compressed_state()
        self._cache_view_static = None
        self._cache_view_runtime = None
        self._record_compressed_prefix_ready_event()

    def _record_compressed_prefix_ready_event(self):
        if self.cache_device is None or torch.device(self.cache_device).type != 'cuda':
            self._compressed_prefix_ready_event = None
            self._compressed_prefix_ready_event_pending = False
            return
        if self._compressed_prefix_seq_len <= 0:
            self._compressed_prefix_ready_event = None
            self._compressed_prefix_ready_event_pending = False
            return
        with torch.cuda.device(self.cache_device):
            if self._compressed_prefix_ready_event is None:
                self._compressed_prefix_ready_event = torch.cuda.Event(blocking=False, interprocess=False)
            self._compressed_prefix_ready_event.record(torch.cuda.current_stream(self.cache_device))
        self._compressed_prefix_ready_event_pending = True

    def wait_for_compressed_prefix_ready(self, device: Optional[torch.device] = None) -> bool:
        if not self._compressed_prefix_ready_event_pending or self._compressed_prefix_ready_event is None:
            return False
        wait_device = self.cache_device if device is None else device
        if wait_device is None or torch.device(wait_device).type != 'cuda':
            return False
        with torch.cuda.device(wait_device):
            torch.cuda.current_stream(wait_device).wait_event(self._compressed_prefix_ready_event)
        self._compressed_prefix_ready_event_pending = False
        return True

    def _flush_one_block(self, need_adjust: bool):
        active_keys, active_values = self._active_residual_tensors()
        block_keys = active_keys[..., : self.block_size, :].contiguous()
        block_values = active_values[..., : self.block_size, :].contiguous()
        flushed_seq_len = int(block_keys.shape[-2])
        remaining_keys = active_keys[..., self.block_size :, :].contiguous()
        remaining_values = active_values[..., self.block_size :, :].contiguous()
        if remaining_keys.shape[-2] == 0:
            remaining_keys = None
            remaining_values = None
        self._set_active_residual_tensors(remaining_keys, remaining_values)
        key_state, value_state, key_decode_state, value_decode_state, keys_exp_count, values_exp_count = self._compress_span(
            block_keys,
            block_values,
            need_adjust=need_adjust,
        )
        self._append_compressed_prefix_span(
            key_state,
            value_state,
            key_decode_state,
            value_decode_state,
            seq_len=flushed_seq_len,
        )
        return keys_exp_count, values_exp_count

    def _flush_ready_blocks(self, need_adjust: bool):
        keys_hist = None
        values_hist = None
        flushed_any = False
        active_keys, _ = self._active_residual_tensors()
        while active_keys is not None and active_keys.shape[-2] >= self.block_size:
            flushed_any = True
            block_keys_hist, block_values_hist = self._flush_one_block(need_adjust=need_adjust)
            if need_adjust and block_keys_hist is not None:
                if keys_hist is None:
                    keys_hist = block_keys_hist.clone()
                    values_hist = block_values_hist.clone()
                else:
                    keys_hist = keys_hist + block_keys_hist
                    values_hist = values_hist + block_values_hist
            active_keys, _ = self._active_residual_tensors()
        return keys_hist, values_hist, flushed_any

    def _materialize_full_tiled_cache(self):
        key_parts = []
        value_parts = []
        if self._compressed_prefix_key_state is not None:
            key_tensor = self._unified_key_tensor
            value_tensor = self._unified_value_tensor
            if key_tensor is None or value_tensor is None:
                self._refresh_unified_compressed_state()
                key_tensor = self._unified_key_tensor
                value_tensor = self._unified_value_tensor
            if key_tensor is not None and value_tensor is not None:
                key_parts.append(key_tensor.materialize())
                value_parts.append(value_tensor.materialize())
        if key_parts:
            return torch.cat(key_parts, dim=-2).contiguous(), torch.cat(value_parts, dim=-2).contiguous()
        return self._empty_states(), self._empty_states()

    def _materialize_prefix_legacy(self):
        if self.layout == 'tiled':
            return None, None
        if self.layout == 'segmented_tiled':
            return None, None
        if not self._compressed_prefix_key_blocks:
            return None, None
        if (
            self._materialized_prefix_keys_cache is not None
            and self._materialized_prefix_values_cache is not None
            and self._materialized_prefix_seq_len == int(self._compressed_prefix_seq_len)
        ):
            self.materialize_cache_hits += 1
            return self._materialized_prefix_keys_cache, self._materialized_prefix_values_cache
        key_parts = [Compressor.from_state(state).materialize() for state in self._compressed_prefix_key_blocks]
        value_parts = [Compressor.from_state(state).materialize() for state in self._compressed_prefix_value_blocks]
        prefix_keys = key_parts[0] if len(key_parts) == 1 else torch.cat(key_parts, dim=-2).contiguous()
        prefix_values = value_parts[0] if len(value_parts) == 1 else torch.cat(value_parts, dim=-2).contiguous()
        if not prefix_keys.is_contiguous():
            prefix_keys = prefix_keys.contiguous()
        if not prefix_values.is_contiguous():
            prefix_values = prefix_values.contiguous()
        self._materialized_prefix_keys_cache = prefix_keys
        self._materialized_prefix_values_cache = prefix_values
        self._materialized_prefix_seq_len = int(self._compressed_prefix_seq_len)
        return prefix_keys, prefix_values

    def _materialize_prefix_segmented_tiled(self):
        if self.layout != 'segmented_tiled':
            return None, None
        if not self._compressed_prefix_key_segments:
            return None, None
        if (
            self._materialized_prefix_keys_cache is not None
            and self._materialized_prefix_values_cache is not None
            and self._materialized_prefix_seq_len == int(self._compressed_prefix_seq_len)
        ):
            self.materialize_cache_hits += 1
            return self._materialized_prefix_keys_cache, self._materialized_prefix_values_cache
        key_parts = [materialize_tiled_state(state) for state in self._compressed_prefix_key_segments]
        value_parts = [materialize_tiled_state(state) for state in self._compressed_prefix_value_segments]
        prefix_keys = key_parts[0] if len(key_parts) == 1 else torch.cat(key_parts, dim=-2).contiguous()
        prefix_values = value_parts[0] if len(value_parts) == 1 else torch.cat(value_parts, dim=-2).contiguous()
        self._materialized_prefix_keys_cache = prefix_keys if prefix_keys.is_contiguous() else prefix_keys.contiguous()
        self._materialized_prefix_values_cache = prefix_values if prefix_values.is_contiguous() else prefix_values.contiguous()
        self._materialized_prefix_seq_len = int(self._compressed_prefix_seq_len)
        return self._materialized_prefix_keys_cache, self._materialized_prefix_values_cache

    def materialize_dense(self):
        begin_time = time.perf_counter()
        self.materialize_calls += 1
        if self.layout == 'segmented_tiled':
            prefix_keys, prefix_values = self._materialize_prefix_segmented_tiled()
            residual_keys, residual_values = self._active_residual_tensors()
            key_parts = []
            value_parts = []
            if prefix_keys is not None:
                key_parts.append(prefix_keys)
                value_parts.append(prefix_values)
            if residual_keys is not None and residual_keys.shape[-2] > 0:
                key_parts.append(residual_keys)
                value_parts.append(residual_values)
            self._materialized_residual_len = 0 if residual_keys is None else int(residual_keys.shape[-2])
            self._materialized_residual_keys_ref = residual_keys
            self._materialized_residual_values_ref = residual_values
            if len(key_parts) == 1:
                self.materialize_time += time.perf_counter() - begin_time
                return key_parts[0], value_parts[0]
            if len(key_parts) > 1:
                self.materialize_time += time.perf_counter() - begin_time
                return torch.cat(key_parts, dim=-2).contiguous(), torch.cat(value_parts, dim=-2).contiguous()
            self.materialize_time += time.perf_counter() - begin_time
            return self._empty_states(), self._empty_states()
        prefix_keys, prefix_values = self._materialize_prefix_legacy()
        residual_keys, residual_values = self._active_residual_tensors()
        residual_len = 0 if residual_keys is None else int(residual_keys.shape[-2])
        residual_changed = (
            residual_len != self._materialized_residual_len
            or residual_keys is not self._materialized_residual_keys_ref
            or residual_values is not self._materialized_residual_values_ref
        )
        if not residual_changed and prefix_keys is None:
            self.materialize_cache_hits += 1
        key_parts = []
        value_parts = []
        if prefix_keys is not None:
            key_parts.append(prefix_keys)
            value_parts.append(prefix_values)
        if residual_len > 0:
            key_parts.append(residual_keys)
            value_parts.append(residual_values)
        self._materialized_residual_len = residual_len
        self._materialized_residual_keys_ref = residual_keys
        self._materialized_residual_values_ref = residual_values
        if len(key_parts) == 1:
            only_keys = key_parts[0]
            only_values = value_parts[0]
            self.materialize_time += time.perf_counter() - begin_time
            return (
                only_keys if only_keys.is_contiguous() else only_keys.contiguous(),
                only_values if only_values.is_contiguous() else only_values.contiguous(),
            )
        if len(key_parts) > 1:
            self.materialize_time += time.perf_counter() - begin_time
            return torch.cat(key_parts, dim=-2).contiguous(), torch.cat(value_parts, dim=-2).contiguous()
        self.materialize_time += time.perf_counter() - begin_time
        return self._empty_states(), self._empty_states()

    def _empty_states(self):
        if self.cache_device is None:
            return None
        return torch.empty((0,), device=self.cache_device, dtype=self.cache_dtype)

    def get_cache_view(self):
        with record_function('float11.cache_view_export.inner'):
            if self._cache_view_static is None:
                self._cache_view_static = {
                    'layout': self.layout,
                    'compressed_prefix_key': self._compressed_prefix_key_metadata,
                    'compressed_prefix_value': self._compressed_prefix_value_metadata,
                    'compressed_prefix_seq_len': self._compressed_prefix_seq_len,
                    'closed_compressed_prefix_seq_len': self._compressed_prefix_seq_len,
                    'block_size': self.block_size,
                    'seq_tile_size': self.seq_tile_size,
                    'decode_impl': 'materialize_dense',
                }
                self._cache_view_runtime = {
                    **self._cache_view_static,
                    'residual_keys': None,
                    'residual_values': None,
                    'seq_length': 0,
                    '_float11_layer': self,
                }

            active_residual_keys, active_residual_values = self._active_residual_tensors()

            cache_view = self._cache_view_runtime
            cache_view['residual_keys'] = active_residual_keys
            cache_view['residual_values'] = active_residual_values
            cache_view['seq_length'] = self.cumulative_length
            cache_view['compressed_prefix_key'] = self._compressed_prefix_key_metadata
            cache_view['compressed_prefix_value'] = self._compressed_prefix_value_metadata

            return cache_view

    def get_memory(self):
        compressed_memory = 0
        original_memory = 0
        keys_outlier_ratios = 0.0
        values_outlier_ratios = 0.0
        prefix_count = (
            1 if self.layout == 'tiled' and self._compressed_prefix_key_state is not None
            else len(self._compressed_prefix_key_segments) if self.layout == 'segmented_tiled'
            else len(self._compressed_prefix_key_blocks)
        )
        if self.layout == 'tiled' and self._compressed_prefix_key_state is not None:
            if self.layout == 'tiled':
                prefix_key_tensor = TiledCompressedTensor(
                    sign_mass_data=self._compressed_prefix_key_state['sign_mass_data'],
                    exp_data=self._compressed_prefix_key_state['exp_data'],
                    invalid_exp=self._compressed_prefix_key_state.get('invalid_exp', torch.empty((0,), dtype=torch.uint8, device=self._compressed_prefix_key_state['device'])),
                    tile_invalid_base=self._compressed_prefix_key_state.get('tile_invalid_base', torch.empty((0,), dtype=torch.int32, device=self._compressed_prefix_key_state['device'])),
                    left=self._compressed_prefix_key_state['left'],
                    right=self._compressed_prefix_key_state['right'],
                    nbits=self._compressed_prefix_key_state['nbits'],
                    original_shape=tuple(self._compressed_prefix_key_state['original_shape']),
                    seq_tile_size=self._compressed_prefix_key_state['seq_tile_size'],
                    head_dim=self._compressed_prefix_key_state['head_dim'],
                    tile_head_dim=self._compressed_prefix_key_state['tile_head_dim'],
                    tile_num_lanes=self._compressed_prefix_key_state['tile_num_lanes'],
                    num_sequence_tiles=int(self._compressed_prefix_key_state['num_sequence_tiles']),
                    num_kv_heads=self._compressed_prefix_key_state['num_kv_heads'],
                    device=self._compressed_prefix_key_state['device'],
                )
                prefix_value_tensor = TiledCompressedTensor(
                    sign_mass_data=self._compressed_prefix_value_state['sign_mass_data'],
                    exp_data=self._compressed_prefix_value_state['exp_data'],
                    invalid_exp=self._compressed_prefix_value_state.get('invalid_exp', torch.empty((0,), dtype=torch.uint8, device=self._compressed_prefix_value_state['device'])),
                    tile_invalid_base=self._compressed_prefix_value_state.get('tile_invalid_base', torch.empty((0,), dtype=torch.int32, device=self._compressed_prefix_value_state['device'])),
                    left=self._compressed_prefix_value_state['left'],
                    right=self._compressed_prefix_value_state['right'],
                    nbits=self._compressed_prefix_value_state['nbits'],
                    original_shape=tuple(self._compressed_prefix_value_state['original_shape']),
                    seq_tile_size=self._compressed_prefix_value_state['seq_tile_size'],
                    head_dim=self._compressed_prefix_value_state['head_dim'],
                    tile_head_dim=self._compressed_prefix_value_state['tile_head_dim'],
                    tile_num_lanes=self._compressed_prefix_value_state['tile_num_lanes'],
                    num_sequence_tiles=int(self._compressed_prefix_value_state['num_sequence_tiles']),
                    num_kv_heads=self._compressed_prefix_value_state['num_kv_heads'],
                    device=self._compressed_prefix_value_state['device'],
                )
                key_mem = prefix_key_tensor.get_memory()
                value_mem = prefix_value_tensor.get_memory()
                compressed_memory += key_mem['compressed_memory'] + value_mem['compressed_memory']
                original_memory += key_mem['original_memory'] + value_mem['original_memory']
                keys_outlier_ratios += Float11TiledCompressedBlock._outlier_ratio(prefix_key_tensor)
                values_outlier_ratios += Float11TiledCompressedBlock._outlier_ratio(prefix_value_tensor)
        elif self.layout == 'segmented_tiled':
            for key_state, value_state in zip(self._compressed_prefix_key_segments, self._compressed_prefix_value_segments):
                prefix_key_tensor = TiledCompressedTensor(
                    sign_mass_data=key_state['sign_mass_data'],
                    exp_data=key_state['exp_data'],
                    invalid_exp=key_state.get('invalid_exp', torch.empty((0,), dtype=torch.uint8, device=key_state['device'])),
                    tile_invalid_base=key_state.get('tile_invalid_base', torch.empty((0,), dtype=torch.int32, device=key_state['device'])),
                    left=key_state['left'],
                    right=key_state['right'],
                    nbits=key_state['nbits'],
                    original_shape=tuple(key_state['original_shape']),
                    seq_tile_size=key_state['seq_tile_size'],
                    head_dim=key_state['head_dim'],
                    tile_head_dim=key_state['tile_head_dim'],
                    tile_num_lanes=key_state['tile_num_lanes'],
                    num_sequence_tiles=int(key_state['num_sequence_tiles']),
                    num_kv_heads=key_state['num_kv_heads'],
                    device=key_state['device'],
                )
                prefix_value_tensor = TiledCompressedTensor(
                    sign_mass_data=value_state['sign_mass_data'],
                    exp_data=value_state['exp_data'],
                    invalid_exp=value_state.get('invalid_exp', torch.empty((0,), dtype=torch.uint8, device=value_state['device'])),
                    tile_invalid_base=value_state.get('tile_invalid_base', torch.empty((0,), dtype=torch.int32, device=value_state['device'])),
                    left=value_state['left'],
                    right=value_state['right'],
                    nbits=value_state['nbits'],
                    original_shape=tuple(value_state['original_shape']),
                    seq_tile_size=value_state['seq_tile_size'],
                    head_dim=value_state['head_dim'],
                    tile_head_dim=value_state['tile_head_dim'],
                    tile_num_lanes=value_state['tile_num_lanes'],
                    num_sequence_tiles=int(value_state['num_sequence_tiles']),
                    num_kv_heads=value_state['num_kv_heads'],
                    device=value_state['device'],
                )
                key_mem = prefix_key_tensor.get_memory()
                value_mem = prefix_value_tensor.get_memory()
                compressed_memory += key_mem['compressed_memory'] + value_mem['compressed_memory']
                original_memory += key_mem['original_memory'] + value_mem['original_memory']
                keys_outlier_ratios += Float11TiledCompressedBlock._outlier_ratio(prefix_key_tensor)
                values_outlier_ratios += Float11TiledCompressedBlock._outlier_ratio(prefix_value_tensor)
        elif self.layout != 'tiled':
            for key_state, value_state in zip(self._compressed_prefix_key_blocks, self._compressed_prefix_value_blocks):
                prefix_key_compressor = Compressor.from_state(key_state)
                prefix_value_compressor = Compressor.from_state(value_state)
                key_mem = prefix_key_compressor.get_memory()
                value_mem = prefix_value_compressor.get_memory()
                compressed_memory += key_mem['compressed_memory'] + value_mem['compressed_memory']
                original_memory += key_mem['original_memory'] + value_mem['original_memory']
                keys_outlier_ratios += Float11CompressedBlock._outlier_ratio(prefix_key_compressor)
                values_outlier_ratios += Float11CompressedBlock._outlier_ratio(prefix_value_compressor)
        active_residual_keys, active_residual_values = self._active_residual_tensors()
        if active_residual_keys is not None:
            compressed_memory += active_residual_keys.numel() * active_residual_keys.element_size()
            compressed_memory += active_residual_values.numel() * active_residual_values.element_size()
            original_memory += active_residual_keys.numel() * active_residual_keys.element_size()
            original_memory += active_residual_values.numel() * active_residual_values.element_size()
        if prefix_count > 0:
            keys_outlier_ratios /= prefix_count
            values_outlier_ratios /= prefix_count
        active_residual_keys, _ = self._active_residual_tensors()
        return {
            'compressed_memory': compressed_memory,
            'original_memory': original_memory,
            'keys_outlier_ratios': keys_outlier_ratios,
            'values_outlier_ratios': values_outlier_ratios,
            'materialize_calls': self.materialize_calls,
            'materialize_cache_hits': self.materialize_cache_hits,
            'materialize_time': self.materialize_time,
            'stored_compressed_prefix_seq_len': self._compressed_prefix_seq_len,
            'stored_residual_seq_len': 0 if active_residual_keys is None else int(active_residual_keys.shape[-2]),
            'compressed_block_flushes': self._compressed_block_flushes,
            'compressed_tokens_flushed': self._compressed_tokens_flushed,
            'residual_tokens_kept': 0 if active_residual_keys is None else int(active_residual_keys.shape[-2]),
            'stored_compressed_block_count': 0 if self.block_size <= 0 else self._compressed_tokens_flushed // self.block_size,
            'prefill_reuse_hits': self.prefill_reuse_hits,
        }

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        need_adjust: bool = False,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ):
        self.cumulative_length += key_states.shape[-2]
        first_update = not self.is_initialized
        if first_update:
            self.lazy_initialization(key_states, value_states)
            self.cache_device = key_states.device
            self.cache_dtype = key_states.dtype
        if need_adjust:
            self.keys_right, self.keys_nbits = self.new_keys_right, self.new_keys_nbits
            self.values_right, self.values_nbits = self.new_values_right, self.new_values_nbits

        q_len = key_states.shape[-2]
        prefill_fastpath = q_len > 1 and first_update

        self._append_residual(key_states, value_states)
        keys_exp_count, values_exp_count, flushed_any = self._flush_ready_blocks(need_adjust=need_adjust)

        if flushed_any:
            self._cache_view_static = None
            self._cache_view_runtime = None

        dense_keys = dense_values = None
        if prefill_fastpath:
            self.prefill_reuse_hits += 1
            dense_keys = key_states if key_states.dtype == torch.bfloat16 else key_states.to(torch.bfloat16)
            dense_values = value_states if value_states.dtype == torch.bfloat16 else value_states.to(torch.bfloat16)
        else:
            dense_keys, dense_values = self.materialize_dense()

        return {
            'key_states': dense_keys,
            'value_states': dense_values,
            'keys_exp_count': keys_exp_count,
            'values_exp_count': values_exp_count,
        }

    def get_seq_length(self) -> int:
        return self.cumulative_length


class Float11Cache(Cache):
    def __init__(
        self,
        config,
        right,
        exp_bits,
        need_adjust: bool = False,
        update_steps: int = 64,
        residual_length: int = 0,
        block_size: int = 128,
        layout: str = 'legacy',
        seq_tile_size: Optional[int] = None,
    ):
        config = config.get_text_config(decoder=True)

        if isinstance(right, int):
            layers = [
                float11Layer(
                    right,
                    right,
                    exp_bits,
                    exp_bits,
                    residual_length=residual_length,
                    block_size=block_size,
                    layout=layout,
                    seq_tile_size=seq_tile_size,
                )
                for _ in range(config.num_hidden_layers)
            ]
        else:
            layers = [
                float11Layer(
                    right[i][0],
                    right[i][1],
                    exp_bits[i][0],
                    exp_bits[i][1],
                    residual_length=residual_length,
                    block_size=block_size,
                    layout=layout,
                    seq_tile_size=seq_tile_size,
                )
                for i in range(config.num_hidden_layers)
            ]

        self.steps = 0
        self.need_adjust = bool(need_adjust and update_steps != 0)
        self.update_steps = int(update_steps) if self.need_adjust else 0
        self._pending_adjusts: dict[int, concurrent.futures.Future] = {}

        self.peak_mem = 0
        self.layout = layout
        self.seq_tile_size = seq_tile_size if seq_tile_size is not None else block_size
        super().__init__(layers=layers)

    def __getstate__(self):
        for fut in self._pending_adjusts.values():
            fut.result()
        state = self.__dict__.copy()
        # Futures/executor internals contain thread synchronization objects that are not picklable.
        state['_pending_adjusts'] = {}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._pending_adjusts = {}

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ):
        if self.layer_class_to_replicate is not None:
            while len(self.layers) <= layer_idx:
                self.layers.append(self.layer_class_to_replicate())

        need_adjust = self.need_adjust and self.steps % self.update_steps == 0
        update_result = self.layers[layer_idx].update(key_states, value_states, need_adjust, cache_kwargs)
        keys_exp_count = update_result['keys_exp_count']
        values_exp_count = update_result['values_exp_count']

        if need_adjust and keys_exp_count is not None:
            if layer_idx in self._pending_adjusts:
                self._pending_adjusts.pop(layer_idx).result()
            keys_hist_cpu = keys_exp_count.to('cpu') if keys_exp_count is not None else None
            values_hist_cpu = values_exp_count.to('cpu') if values_exp_count is not None else None
            self._pending_adjusts[layer_idx] = _ADJUST_EXECUTOR.submit(
                self.layers[layer_idx].adjust_compressor,
                keys_hist_cpu,
                values_hist_cpu,
            )

        if layer_idx == len(self.layers) - 1:
            self.steps += 1

        return update_result['key_states'], update_result['value_states']

    def get_layer_cache_view(self, layer_idx: int):
        return self.layers[layer_idx].get_cache_view()

    def materialize_layer(self, layer_idx: int):
        return self.layers[layer_idx].materialize_dense()

    def get_memory(self):
        total_info = {
            'kv_compressed_memory': 0,
            'kv_original_memory': 0,
            'keys_update_times': 0,
            'values_update_times': 0,
            'keys_outlier_ratios': 0.0,
            'values_outlier_ratios': 0.0,
            'materialize_calls': 0,
            'materialize_cache_hits': 0,
            'materialize_time': 0.0,
            'stored_compressed_prefix_seq_len': 0,
            'stored_residual_seq_len': 0,
            'compressed_block_flushes': 0,
            'compressed_tokens_flushed': 0,
            'residual_tokens_kept': 0,
            'stored_compressed_block_count': 0,
            'prefill_reuse_hits': 0,
        }
        for layer in self.layers:
            layer_mem = layer.get_memory()
            total_info['kv_compressed_memory'] += layer_mem['compressed_memory']
            total_info['kv_original_memory'] += layer_mem['original_memory']
            total_info['keys_update_times'] += layer.keys_update_times
            total_info['values_update_times'] += layer.values_update_times
            total_info['materialize_calls'] += layer_mem['materialize_calls']
            total_info['materialize_cache_hits'] += layer_mem['materialize_cache_hits']
            total_info['materialize_time'] += layer_mem['materialize_time']
            total_info['stored_compressed_prefix_seq_len'] += layer_mem['stored_compressed_prefix_seq_len']
            total_info['stored_residual_seq_len'] += layer_mem['stored_residual_seq_len']
            total_info['compressed_block_flushes'] += layer_mem['compressed_block_flushes']
            total_info['compressed_tokens_flushed'] += layer_mem['compressed_tokens_flushed']
            total_info['residual_tokens_kept'] += layer_mem['residual_tokens_kept']
            total_info['stored_compressed_block_count'] += layer_mem['stored_compressed_block_count']
            total_info['prefill_reuse_hits'] += layer_mem['prefill_reuse_hits']
            total_info['keys_outlier_ratios'] += layer_mem['keys_outlier_ratios']
            total_info['values_outlier_ratios'] += layer_mem['values_outlier_ratios']
        total_info['keys_outlier_ratios'] /= len(self.layers)
        total_info['values_outlier_ratios'] /= len(self.layers)
        total_info['kv_compressed_memory'] /= 1024**3
        total_info['kv_original_memory'] /= 1024**3
        return total_info

    def synchronize_and_release(self):
        for fut in self._pending_adjusts.values():
            fut.result()
        self._pending_adjusts.clear()

        for layer in self.layers:
            layer.residual_keys = None
            layer.residual_values = None
            layer._materialized_residual_len = 0
            layer._materialized_residual_keys_ref = None
            layer._materialized_residual_values_ref = None
            layer._compressed_prefix_key_state = None
            layer._compressed_prefix_value_state = None
            layer._compressed_prefix_key_metadata = None
            layer._compressed_prefix_value_metadata = None
            layer._compressed_prefix_key_segments = []
            layer._compressed_prefix_value_segments = []
            layer._compressed_prefix_key_segment_metadata = []
            layer._compressed_prefix_value_segment_metadata = []
            layer._compressed_prefix_key_blocks = []
            layer._compressed_prefix_value_blocks = []
            layer._compressed_prefix_seq_len = 0
            layer._materialized_prefix_keys_cache = None
            layer._materialized_prefix_values_cache = None
            layer._materialized_prefix_seq_len = 0
            layer._unified_key_tensor = None
            layer._unified_value_tensor = None
            layer._compressed_prefix_ready_event = None
            layer._compressed_prefix_ready_event_pending = False
            layer._compressed_block_flushes = 0
            layer._compressed_tokens_flushed = 0
            layer._cache_view_static = None


# BHSD
# (i,j,k,l)
# i*HSD+j*SD+k*D+l
# BHSD -> BH(S+1)D
# i*H(S+1)D+j*(S+1)D+k*D+l
