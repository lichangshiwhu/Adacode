from __future__ import annotations

from typing import Any, Optional
import concurrent.futures
import time
import warnings

import torch
from transformers import Cache

from kv_compression.float_cache_utils import (
    Float11Cache,
    _ADJUST_EXECUTOR,
    _compute_adjusted_params,
    _concat_seq_tensors,
    float11Layer,
)


def _component_pairs(
    num_layers: int,
    first_right: int,
    second_right: int,
    first_nbits: int,
    second_nbits: int,
) -> tuple[list[list[int]], list[list[int]]]:
    rights = [[int(first_right), int(second_right)] for _ in range(num_layers)]
    nbits = [[int(first_nbits), int(second_nbits)] for _ in range(num_layers)]
    return rights, nbits


def _normalize_component_pairs(value, num_layers: int, name: str) -> list[list[int]]:
    if isinstance(value, int):
        return [[int(value), int(value)] for _ in range(num_layers)]
    if len(value) != num_layers:
        raise ValueError(f'{name} must have {num_layers} layer entries, got {len(value)}.')
    normalized = []
    for layer_idx, pair in enumerate(value):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f'{name}[{layer_idx}] must be a pair for latent/rope, got {pair!r}.')
        normalized.append([int(pair[0]), int(pair[1])])
    return normalized


class MLALatentFloat11DenseRopeLayer(float11Layer):
    def __init__(
        self,
        latent_right: int = 127,
        latent_nbits: int = 4,
        residual_length: int = 0,
        block_size: int = 128,
        layout: str = 'segmented_tiled',
        seq_tile_size: Optional[int] = None,
    ):
        super().__init__(
            keys_right=latent_right,
            values_right=latent_right,
            keys_nbits=latent_nbits,
            values_nbits=latent_nbits,
            residual_length=residual_length,
            block_size=block_size,
            layout=layout,
            seq_tile_size=seq_tile_size,
        )
        self.residual_rope: Optional[torch.Tensor] = None
        self._dense_rope_prefix_blocks: list[torch.Tensor] = []
        self._dense_rope_prefix_seq_len = 0
        self._materialized_rope_cache: Optional[torch.Tensor] = None
        self._materialized_rope_seq_len = 0

    def _active_residual_tensors(self):
        return self.residual_keys, self.residual_rope

    def _set_active_residual_tensors(
        self,
        key_states: Optional[torch.Tensor],
        value_states: Optional[torch.Tensor],
    ):
        self.residual_keys = key_states
        self.residual_rope = value_states
        self.residual_values = value_states

    def _compress_span(self, key_states: torch.Tensor, value_states: torch.Tensor, need_adjust: bool):
        key_state, _, key_decode_state, _, keys_exp_count, _ = super()._compress_span(
            key_states,
            key_states,
            need_adjust=need_adjust,
        )
        return key_state, None, key_decode_state, None, keys_exp_count, keys_exp_count

    def _compress_latent_span(
        self,
        latent_states: torch.Tensor,
        need_adjust: bool,
    ):
        original_seq_tile_size = self.seq_tile_size
        if self.layout == 'segmented_tiled':
            self.seq_tile_size = int(latent_states.shape[-2])
        try:
            return self._compress_span(latent_states, latent_states, need_adjust=need_adjust)
        finally:
            self.seq_tile_size = original_seq_tile_size

    def _append_compressed_prefix_span(
        self,
        key_state: dict[str, Any],
        value_state: Optional[dict[str, Any]],
        key_decode_state: dict[str, Any],
        value_decode_state: Optional[dict[str, Any]],
        seq_len: int,
    ):
        active_latent, active_rope = self._active_residual_tensors()
        if active_rope is None:
            raise RuntimeError('Dense RoPE residual is missing while flushing MLA latent block.')
        rope_block = active_rope[..., : int(seq_len), :].contiguous()
        super()._append_compressed_prefix_span(
            key_state,
            key_state,
            key_decode_state,
            key_decode_state,
            seq_len,
        )
        self._dense_rope_prefix_blocks.append(rope_block)
        self._dense_rope_prefix_seq_len += int(seq_len)
        self._materialized_rope_cache = None
        self._materialized_rope_seq_len = 0

    def _flush_one_block(self, need_adjust: bool):
        return self._flush_span(min(self.block_size, self._active_residual_tensors()[0].shape[-2]), need_adjust)

    def _flush_span(self, seq_len: int, need_adjust: bool):
        active_latent, active_rope = self._active_residual_tensors()
        seq_len = int(seq_len)
        if active_latent is None or active_rope is None or seq_len <= 0:
            return None, None
        seq_len = min(seq_len, int(active_latent.shape[-2]))
        block_latent = active_latent[..., :seq_len, :].contiguous()
        block_rope = active_rope[..., :seq_len, :].contiguous()
        remaining_latent = active_latent[..., seq_len:, :].contiguous()
        remaining_rope = active_rope[..., seq_len:, :].contiguous()
        if remaining_latent.shape[-2] == 0:
            remaining_latent = None
            remaining_rope = None
        self._set_active_residual_tensors(remaining_latent, remaining_rope)
        key_state, _, key_decode_state, _, keys_exp_count, _ = self._compress_latent_span(
            block_latent,
            need_adjust=need_adjust,
        )
        super()._append_compressed_prefix_span(
            key_state,
            key_state,
            key_decode_state,
            key_decode_state,
            int(block_latent.shape[-2]),
        )
        self._dense_rope_prefix_blocks.append(block_rope)
        self._dense_rope_prefix_seq_len += int(block_rope.shape[-2])
        self._materialized_rope_cache = None
        self._materialized_rope_seq_len = 0
        return keys_exp_count, keys_exp_count

    def _flush_all_available_spans(self, need_adjust: bool):
        keys_hist = values_hist = None
        flushed_any = False
        active_latent, _ = self._active_residual_tensors()
        while active_latent is not None and active_latent.shape[-2] >= self.block_size:
            flushed_any = True
            span_len = int(self.block_size)
            adjust_start_segment = (
                len(self._compressed_prefix_key_segments)
                if self.layout == 'segmented_tiled' and need_adjust
                else None
            )
            block_keys_hist, block_values_hist = self._flush_span(span_len, need_adjust=need_adjust)
            if adjust_start_segment is not None and block_keys_hist is not None:
                self.mark_adjust_recompress_start(adjust_start_segment)
            if need_adjust and block_keys_hist is not None:
                if keys_hist is None:
                    keys_hist = block_keys_hist.clone()
                    values_hist = block_values_hist.clone()
                else:
                    keys_hist = keys_hist + block_keys_hist
                    values_hist = values_hist + block_values_hist
            active_latent, _ = self._active_residual_tensors()
        return keys_hist, values_hist, flushed_any

    def adjust_compressor(self, keys_exp_count, values_exp_count):
        self.new_keys_right, self.new_keys_nbits = _compute_adjusted_params(keys_exp_count, None)[0]
        self.new_values_right, self.new_values_nbits = self.new_keys_right, self.new_keys_nbits

    def apply_pending_adjust_recompress_if_needed(
        self,
        keys_params: tuple[int, int],
        values_params: tuple[int, int],
    ) -> bool:
        if self.layout != 'segmented_tiled':
            return False
        start_segment = self._pending_adjust_recompress_start_segment
        self._pending_adjust_recompress_start_segment = None
        if start_segment is None:
            return False

        new_keys_right, new_keys_nbits = keys_params
        old_params = (
            self.keys_right,
            self.keys_nbits,
            self.values_right,
            self.values_nbits,
        )
        new_params = (
            int(new_keys_right),
            int(new_keys_nbits),
            int(new_keys_right),
            int(new_keys_nbits),
        )
        self._record_adjust_proposal(
            (int(new_keys_right), int(new_keys_nbits)),
            (int(new_keys_right), int(new_keys_nbits)),
        )
        self.new_keys_right, self.new_keys_nbits = int(new_keys_right), int(new_keys_nbits)
        self.new_values_right, self.new_values_nbits = self.new_keys_right, self.new_keys_nbits
        self.keys_right, self.keys_nbits = self.new_keys_right, self.new_keys_nbits
        self.values_right, self.values_nbits = self.new_values_right, self.new_values_nbits
        if old_params == new_params:
            return False
        self._record_effective_param_update(
            (old_params[0], old_params[1]),
            (old_params[2], old_params[3]),
            (self.keys_right, self.keys_nbits),
            (self.values_right, self.values_nbits),
        )
        if start_segment >= len(self._compressed_prefix_key_segments):
            return False

        latent_states = self._materialize_segments(
            self._compressed_prefix_key_segments[start_segment:]
        )
        rope_parts = self._dense_rope_prefix_blocks[start_segment:]
        rope_states = None
        if rope_parts:
            rope_states = rope_parts[0] if len(rope_parts) == 1 else torch.cat(rope_parts, dim=-2).contiguous()
        if (
            latent_states is None
            or rope_states is None
            or latent_states.shape[-2] == 0
            or rope_states.shape[-2] == 0
        ):
            return False

        self._compressed_prefix_key_segments = self._compressed_prefix_key_segments[:start_segment]
        self._compressed_prefix_value_segments = self._compressed_prefix_value_segments[:start_segment]
        self._compressed_prefix_key_segment_metadata = self._compressed_prefix_key_segment_metadata[:start_segment]
        self._compressed_prefix_value_segment_metadata = self._compressed_prefix_value_segment_metadata[:start_segment]
        self._dense_rope_prefix_blocks = self._dense_rope_prefix_blocks[:start_segment]
        self._dense_rope_prefix_seq_len = sum(
            int(block.shape[-2])
            for block in self._dense_rope_prefix_blocks
        )
        self._compressed_prefix_key_state = self._export_segmented_state_or_none(self._compressed_prefix_key_segments)
        self._compressed_prefix_value_state = self._export_segmented_state_or_none(self._compressed_prefix_value_segments)
        self._compressed_prefix_key_metadata = self._export_segmented_state_or_none(self._compressed_prefix_key_segment_metadata)
        self._compressed_prefix_value_metadata = self._export_segmented_state_or_none(self._compressed_prefix_value_segment_metadata)
        self._compressed_prefix_seq_len = sum(
            int(tuple(state['original_shape'])[-2])
            for state in self._compressed_prefix_key_segments
        )
        self._compressed_block_flushes = len(self._compressed_prefix_key_segments)
        self._compressed_tokens_flushed = self._compressed_prefix_seq_len
        self._materialized_prefix_keys_cache = None
        self._materialized_prefix_values_cache = None
        self._materialized_prefix_seq_len = 0
        self._materialized_rope_cache = None
        self._materialized_rope_seq_len = 0
        self._cache_view_static = None
        self._cache_view_runtime = None

        active_latent, active_rope = self._active_residual_tensors()
        self._set_active_residual_tensors(latent_states, rope_states)
        self._flush_all_available_spans(need_adjust=False)
        recompressed_tail_latent, recompressed_tail_rope = self._active_residual_tensors()
        if active_latent is not None and active_latent.shape[-2] > 0:
            if recompressed_tail_latent is None:
                self._set_active_residual_tensors(active_latent, active_rope)
            else:
                self._set_active_residual_tensors(
                    _concat_seq_tensors(recompressed_tail_latent, active_latent),
                    _concat_seq_tensors(recompressed_tail_rope, active_rope),
                )
        self._adjust_recompresses += 1
        return True

    @staticmethod
    def _export_segmented_state_or_none(segment_states: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if not segment_states:
            return None
        from kv_compression.float_cache_utils import _export_segmented_tiled_state

        return _export_segmented_tiled_state(segment_states)

    def apply_pending_dense_block_if_needed(
        self,
        keys_params: Optional[tuple[int, int]] = None,
        values_params: Optional[tuple[int, int]] = None,
    ) -> bool:
        return super().apply_pending_dense_block_if_needed(keys_params, keys_params)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        need_adjust: bool = False,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ):
        if self.layout != 'segmented_tiled':
            return super().update(key_states, value_states, need_adjust, cache_kwargs)

        self.cumulative_length += key_states.shape[-2]
        first_update = not self.is_initialized
        if first_update:
            self.lazy_initialization(key_states, value_states)
            self.cache_device = key_states.device
            self.cache_dtype = key_states.dtype
        if need_adjust and not self.has_pending_adjust_work():
            self.keys_right, self.keys_nbits = self.new_keys_right, self.new_keys_nbits
            self.values_right, self.values_nbits = self.new_values_right, self.new_values_nbits

        q_len = key_states.shape[-2]
        prefill_fastpath = q_len > 1 and first_update

        self._append_residual(key_states, value_states)
        keys_exp_count, values_exp_count, flushed_any = self._flush_all_available_spans(
            need_adjust=need_adjust,
        )
        if need_adjust and keys_exp_count is None:
            keys_exp_count, values_exp_count = self._collect_adjust_hist_from_active_residual()
        if need_adjust:
            if prefill_fastpath and flushed_any:
                self.mark_adjust_recompress_start(0)
            self._update_adjust_params_from_hist(keys_exp_count, values_exp_count)

        if flushed_any:
            self._cache_view_static = None
            self._cache_view_runtime = None

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

    def _materialize_rope(self) -> Optional[torch.Tensor]:
        parts = []
        if self._dense_rope_prefix_blocks:
            if (
                self._materialized_rope_cache is not None
                and self._materialized_rope_seq_len == self._dense_rope_prefix_seq_len
            ):
                parts.append(self._materialized_rope_cache)
            else:
                prefix = (
                    self._dense_rope_prefix_blocks[0]
                    if len(self._dense_rope_prefix_blocks) == 1
                    else torch.cat(self._dense_rope_prefix_blocks, dim=-2).contiguous()
                )
                self._materialized_rope_cache = prefix
                self._materialized_rope_seq_len = self._dense_rope_prefix_seq_len
                parts.append(prefix)
        if self.residual_rope is not None and self.residual_rope.shape[-2] > 0:
            parts.append(self.residual_rope)
        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]
        return torch.cat(parts, dim=-2).contiguous()

    def materialize_dense(self):
        begin_time = time.perf_counter()
        self.materialize_calls += 1
        latent_parts = []
        if self.layout == 'segmented_tiled':
            prefix_latent, _ = self._materialize_prefix_segmented_tiled()
        elif self.layout == 'tiled':
            prefix_latent, _ = self._materialize_full_tiled_cache()
        else:
            prefix_latent, _ = self._materialize_prefix_legacy()
        if prefix_latent is not None and prefix_latent.shape[-2] > 0:
            latent_parts.append(prefix_latent)
        if self.residual_keys is not None and self.residual_keys.shape[-2] > 0:
            latent_parts.append(self.residual_keys)
        if len(latent_parts) == 0:
            latent = self._empty_states()
        elif len(latent_parts) == 1:
            latent = latent_parts[0]
        else:
            latent = torch.cat(latent_parts, dim=-2).contiguous()
        rope = self._materialize_rope()
        if rope is None:
            rope = self._empty_states()
        self.materialize_time += time.perf_counter() - begin_time
        return latent, rope

    def get_cache_view(self):
        cache_view = super().get_cache_view()
        cache_view['layout'] = f'{self.layout}_mla_latent_dense_rope'
        cache_view['compressed_prefix_value'] = None
        cache_view['residual_values'] = self.residual_rope
        cache_view['dense_rope_prefix_blocks'] = self._dense_rope_prefix_blocks
        cache_view['dense_rope_prefix_seq_len'] = self._dense_rope_prefix_seq_len
        return cache_view

    def get_memory(self):
        info = super().get_memory()
        residual_latent_memory = 0
        residual_rope_memory = 0
        if self.residual_keys is not None:
            residual_latent_memory = self.residual_keys.numel() * self.residual_keys.element_size()
        if self.residual_rope is not None:
            residual_rope_memory = self.residual_rope.numel() * self.residual_rope.element_size()
        residual_memory = residual_latent_memory + residual_rope_memory
        dense_rope_prefix_memory = 0
        for block in self._dense_rope_prefix_blocks:
            dense_rope_prefix_memory += block.numel() * block.element_size()
        duplicated_prefix_compressed = max(0, info['compressed_memory'] - residual_memory) / 2
        duplicated_prefix_original = max(0, info['original_memory'] - residual_memory) / 2
        info['compressed_memory'] = duplicated_prefix_compressed + residual_memory + dense_rope_prefix_memory
        info['original_memory'] = duplicated_prefix_original + residual_memory + dense_rope_prefix_memory
        info['dense_rope_memory'] = dense_rope_prefix_memory + residual_rope_memory
        info['dense_rope_prefix_seq_len'] = self._dense_rope_prefix_seq_len
        info['rope_outlier_ratios'] = 0.0
        return info

    def release_dense_rope(self):
        self.residual_rope = None
        self._dense_rope_prefix_blocks = []
        self._dense_rope_prefix_seq_len = 0
        self._materialized_rope_cache = None
        self._materialized_rope_seq_len = 0


class MLAFloat11Cache(Float11Cache):
    """Lossless Float11 cache wrapper for DeepSeek-style MLA cache tensors."""

    def __init__(
        self,
        config,
        right=None,
        exp_bits=None,
        latent_right: int = 127,
        rope_right: int = 127,
        latent_nbits: int = 4,
        rope_nbits: int = 4,
        need_adjust: bool = False,
        update_steps: int = 64,
        residual_length: int = 0,
        block_size: int = 128,
        layout: str = 'segmented_tiled',
        seq_tile_size: Optional[int] = None,
        mla_cache_mode: str = 'auto',
        validate_shapes: bool = True,
        compress_rope: bool = True,
    ):
        text_config = config.get_text_config(decoder=True)
        self.mla_cache_mode = str(mla_cache_mode)
        self.validate_shapes = bool(validate_shapes)
        self.compress_rope = bool(compress_rope)
        self._shape_warning_emitted = False
        self.latent_right = int(latent_right)
        self.rope_right = int(rope_right)
        self.latent_nbits = int(latent_nbits)
        self.rope_nbits = int(rope_nbits)
        self.expected_latent_dim = getattr(text_config, 'kv_lora_rank', None)
        self.expected_rope_dim = getattr(text_config, 'qk_rope_head_dim', None)
        num_layers = int(text_config.num_hidden_layers)

        if (right is None) != (exp_bits is None):
            raise ValueError('MLAFloat11Cache expects right and exp_bits to be provided together.')
        if right is not None and exp_bits is not None:
            self.rights = _normalize_component_pairs(right, num_layers, 'right')
            self.nbits = _normalize_component_pairs(exp_bits, num_layers, 'exp_bits')
            latent_right = int(self.rights[0][0])
            rope_right = int(self.rights[0][1])
            latent_nbits = int(self.nbits[0][0])
            rope_nbits = int(self.nbits[0][1])
        else:
            self.rights, self.nbits = _component_pairs(
                num_layers=num_layers,
                first_right=latent_right,
                second_right=rope_right,
                first_nbits=latent_nbits,
                second_nbits=rope_nbits,
            )
        self.latent_right = int(latent_right)
        self.rope_right = int(rope_right)
        self.latent_nbits = int(latent_nbits)
        self.rope_nbits = int(rope_nbits)

        if self.compress_rope:
            super().__init__(
                config=config,
                right=self.rights,
                exp_bits=self.nbits,
                need_adjust=need_adjust,
                update_steps=update_steps,
                residual_length=residual_length,
                block_size=block_size,
                layout=layout,
                seq_tile_size=seq_tile_size,
            )
            return

        layers = [
            MLALatentFloat11DenseRopeLayer(
                latent_right=int(self.rights[layer_idx][0]),
                latent_nbits=int(self.nbits[layer_idx][0]),
                residual_length=residual_length,
                block_size=block_size,
                layout=layout,
                seq_tile_size=seq_tile_size,
            )
            for layer_idx in range(num_layers)
        ]
        self.steps = 0
        self.need_adjust = bool(need_adjust and update_steps != 0)
        self.update_steps = int(update_steps) if self.need_adjust else 0
        self._pending_adjusts: dict[int, concurrent.futures.Future] = {}
        self._segmented_adjust = self.need_adjust and layout == 'segmented_tiled'
        if self._segmented_adjust:
            update_steps_int = max(1, int(update_steps))
            block_size_int = max(1, int(block_size))
            self.update_steps = ((update_steps_int + block_size_int - 1) // block_size_int) * block_size_int
        self.peak_mem = 0
        self.layout = layout
        self.seq_tile_size = seq_tile_size if seq_tile_size is not None else block_size
        if self._segmented_adjust and int(update_steps) != int(self.update_steps):
            warnings.warn(
                'segmented_tiled adjuster rounds kv_adjust_update_steps up to a block_size multiple: '
                f'{int(update_steps)} -> {int(self.update_steps)}.',
                RuntimeWarning,
            )
        Cache.__init__(self, layers=layers)

    def _validate_update_shapes(self, first_states: torch.Tensor, second_states: torch.Tensor) -> None:
        if not self.validate_shapes or self._shape_warning_emitted:
            return
        if first_states.dim() != 4 or second_states.dim() != 4:
            warnings.warn(
                'MLAFloat11Cache expects 4D cache tensors [batch, heads, seq, dim]. '
                f'Got first={tuple(first_states.shape)}, second={tuple(second_states.shape)}.',
                RuntimeWarning,
            )
            self._shape_warning_emitted = True
            return
        if first_states.shape[0] != second_states.shape[0] or first_states.shape[-2] != second_states.shape[-2]:
            warnings.warn(
                'MLAFloat11Cache received cache tensors with different batch or sequence length: '
                f'first={tuple(first_states.shape)}, second={tuple(second_states.shape)}.',
                RuntimeWarning,
            )
            self._shape_warning_emitted = True
            return
        if self.mla_cache_mode == 'latent' and (first_states.shape[1] != 1 or second_states.shape[1] != 1):
            warnings.warn(
                'MLAFloat11Cache latent mode normally receives [batch, 1, seq, dim] tensors '
                f'for compressed_kv and k_pe. Got first={tuple(first_states.shape)}, '
                f'second={tuple(second_states.shape)}.',
                RuntimeWarning,
            )
            self._shape_warning_emitted = True
            return
        if self.mla_cache_mode in {'auto', 'latent'}:
            if self.expected_latent_dim is not None and int(first_states.shape[-1]) != int(self.expected_latent_dim):
                warnings.warn(
                    'MLAFloat11Cache first tensor last dim does not match config.kv_lora_rank: '
                    f'got {int(first_states.shape[-1])}, expected {int(self.expected_latent_dim)}. '
                    'If this model passes expanded full key/value tensors, use --mla_cache_mode expanded.',
                    RuntimeWarning,
                )
                self._shape_warning_emitted = True
                return
            if self.expected_rope_dim is not None and int(second_states.shape[-1]) != int(self.expected_rope_dim):
                warnings.warn(
                    'MLAFloat11Cache second tensor last dim does not match config.qk_rope_head_dim: '
                    f'got {int(second_states.shape[-1])}, expected {int(self.expected_rope_dim)}. '
                    'If this model passes expanded full key/value tensors, use --mla_cache_mode expanded.',
                    RuntimeWarning,
                )
                self._shape_warning_emitted = True

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ):
        self._validate_update_shapes(key_states, value_states)
        if self.compress_rope:
            return super().update(key_states, value_states, layer_idx, cache_kwargs)

        if self.layer_class_to_replicate is not None:
            while len(self.layers) <= layer_idx:
                self.layers.append(self.layer_class_to_replicate())

        if self._segmented_adjust and layer_idx in self._pending_adjusts:
            layer = self.layers[layer_idx]
            if layer.has_pending_adjust_work():
                keys_params = self._pending_adjusts.pop(layer_idx).result()
                if layer.has_pending_adjust_recompress():
                    layer.apply_pending_adjust_recompress_if_needed(keys_params, keys_params)
                else:
                    layer.apply_pending_dense_block_if_needed(keys_params, keys_params)

        if self._segmented_adjust:
            layer = self.layers[layer_idx]
            if layer.has_pending_dense_block() and not layer.has_pending_adjust_work():
                layer.apply_pending_dense_block_if_needed()

        need_adjust = self.need_adjust and self.steps % self.update_steps == 0
        update_result = self.layers[layer_idx].update(key_states, value_states, need_adjust, cache_kwargs)
        keys_exp_count = update_result['keys_exp_count']
        if need_adjust and keys_exp_count is not None:
            if layer_idx in self._pending_adjusts:
                self._pending_adjusts.pop(layer_idx).result()
            keys_hist_cpu = keys_exp_count.to('cpu')
            self._pending_adjusts[layer_idx] = _ADJUST_EXECUTOR.submit(
                lambda hist: _compute_adjusted_params(hist, hist)[0],
                keys_hist_cpu,
            )

        if layer_idx == len(self.layers) - 1:
            self.steps += 1

        return update_result['key_states'], update_result['value_states']

    def get_memory(self):
        info = super().get_memory()
        info['mla_cache_mode'] = self.mla_cache_mode
        info['mla_compress_rope'] = self.compress_rope
        info['mla_latent_right'] = self.latent_right
        info['mla_rope_right'] = self.rope_right
        info['mla_latent_nbits'] = self.latent_nbits
        info['mla_rope_nbits'] = self.rope_nbits
        info['mla_rights'] = self.rights
        info['mla_nbits'] = self.nbits
        info['mla_expected_latent_dim'] = self.expected_latent_dim
        info['mla_expected_rope_dim'] = self.expected_rope_dim
        info['latent_update_times'] = info.get('keys_update_times', 0)
        info['rope_update_times'] = info.get('values_update_times', 0)
        info['latent_outlier_ratios'] = info.get('keys_outlier_ratios', 0.0)
        info['rope_outlier_ratios'] = info.get('values_outlier_ratios', 0.0)
        return info

    def synchronize_and_release(self):
        super().synchronize_and_release()
        if not self.compress_rope:
            for layer in self.layers:
                if hasattr(layer, 'release_dense_rope'):
                    layer.release_dense_rope()
