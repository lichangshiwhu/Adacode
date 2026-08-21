from __future__ import annotations

from typing import Any, Optional
import time

import torch
from transformers.cache_utils import DynamicLayer
from transformers import Cache


def _concat_seq_tensors(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return torch.cat([first, second], dim=-2).contiguous()


def _tensor_bytes(tensor: Optional[torch.Tensor]) -> int:
    if tensor is None:
        return 0
    return int(tensor.numel() * tensor.element_size())


class CPUOffloadLayer(DynamicLayer):
    def __init__(
        self,
        gpu_keep_ratio: float = 0.75,
        pin_memory: bool = True,
        min_offload_tokens: int = 1,
    ):
        super().__init__()
        if not 0.0 < gpu_keep_ratio <= 1.0:
            raise ValueError(f'gpu_keep_ratio must be in (0, 1], got {gpu_keep_ratio}')
        self.gpu_keep_ratio = float(gpu_keep_ratio)
        self.pin_memory = bool(pin_memory)
        self.min_offload_tokens = max(1, int(min_offload_tokens))
        self.cumulative_length = 0
        self.cache_device: Optional[torch.device] = None
        self.cache_dtype: Optional[torch.dtype] = None

        self.cpu_keys: Optional[torch.Tensor] = None
        self.cpu_values: Optional[torch.Tensor] = None
        self.gpu_keys: Optional[torch.Tensor] = None
        self.gpu_values: Optional[torch.Tensor] = None

        self.materialize_calls = 0
        self.materialize_time = 0.0
        self.offload_calls = 0
        self.offloaded_tokens = 0

    def _append_gpu(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        if self.gpu_keys is None:
            self.gpu_keys = key_states.contiguous()
            self.gpu_values = value_states.contiguous()
            return
        self.gpu_keys = _concat_seq_tensors(self.gpu_keys, key_states)
        self.gpu_values = _concat_seq_tensors(self.gpu_values, value_states)

    def _append_cpu(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        cpu_keys = key_states.detach().to('cpu', non_blocking=True).contiguous()
        cpu_values = value_states.detach().to('cpu', non_blocking=True).contiguous()
        if self.pin_memory and torch.cuda.is_available():
            cpu_keys = cpu_keys.pin_memory()
            cpu_values = cpu_values.pin_memory()
        if self.cpu_keys is None:
            self.cpu_keys = cpu_keys
            self.cpu_values = cpu_values
            return
        self.cpu_keys = _concat_seq_tensors(self.cpu_keys, cpu_keys)
        self.cpu_values = _concat_seq_tensors(self.cpu_values, cpu_values)

    def _rebalance_gpu_cache(self) -> None:
        if self.gpu_keys is None:
            return
        gpu_len = int(self.gpu_keys.shape[-2])
        total_len = int(self.cumulative_length)
        target_gpu_len = max(1, int(round(total_len * self.gpu_keep_ratio)))
        offload_len = gpu_len - target_gpu_len
        if offload_len < self.min_offload_tokens:
            return

        offload_keys = self.gpu_keys[..., :offload_len, :].contiguous()
        offload_values = self.gpu_values[..., :offload_len, :].contiguous()
        keep_keys = self.gpu_keys[..., offload_len:, :].contiguous()
        keep_values = self.gpu_values[..., offload_len:, :].contiguous()
        self._append_cpu(offload_keys, offload_values)
        self.gpu_keys = keep_keys if keep_keys.shape[-2] > 0 else None
        self.gpu_values = keep_values if keep_values.shape[-2] > 0 else None
        self.offload_calls += 1
        self.offloaded_tokens += int(offload_len)

    def materialize_dense(self) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        begin_time = time.perf_counter()
        self.materialize_calls += 1
        key_parts = []
        value_parts = []
        if self.cpu_keys is not None and self.cpu_keys.shape[-2] > 0:
            key_parts.append(self.cpu_keys.to(self.cache_device, non_blocking=True))
            value_parts.append(self.cpu_values.to(self.cache_device, non_blocking=True))
        if self.gpu_keys is not None and self.gpu_keys.shape[-2] > 0:
            key_parts.append(self.gpu_keys)
            value_parts.append(self.gpu_values)
        if not key_parts:
            self.materialize_time += time.perf_counter() - begin_time
            return self._empty_states(), self._empty_states()
        if len(key_parts) == 1:
            keys, values = key_parts[0].contiguous(), value_parts[0].contiguous()
        else:
            keys = torch.cat(key_parts, dim=-2).contiguous()
            values = torch.cat(value_parts, dim=-2).contiguous()
        self.materialize_time += time.perf_counter() - begin_time
        return keys, values

    def _empty_states(self):
        if self.cache_device is None:
            return None
        return torch.empty((0,), device=self.cache_device, dtype=self.cache_dtype)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        self.cumulative_length += int(key_states.shape[-2])
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
            self.cache_device = key_states.device
            self.cache_dtype = key_states.dtype
        self._append_gpu(key_states, value_states)
        self._rebalance_gpu_cache()
        return self.materialize_dense()

    def get_seq_length(self) -> int:
        return int(self.cumulative_length)

    def get_memory(self) -> dict[str, float]:
        gpu_bytes = _tensor_bytes(self.gpu_keys) + _tensor_bytes(self.gpu_values)
        cpu_bytes = _tensor_bytes(self.cpu_keys) + _tensor_bytes(self.cpu_values)
        original_bytes = gpu_bytes + cpu_bytes
        return {
            'gpu_cache_memory': gpu_bytes,
            'cpu_cache_memory': cpu_bytes,
            'original_memory': original_bytes,
            'gpu_cache_ratio': 0.0 if original_bytes == 0 else gpu_bytes / original_bytes,
            'cpu_offload_ratio': 0.0 if original_bytes == 0 else cpu_bytes / original_bytes,
            'materialize_calls': self.materialize_calls,
            'materialize_time': self.materialize_time,
            'offload_calls': self.offload_calls,
            'offloaded_tokens': self.offloaded_tokens,
            'stored_gpu_seq_len': 0 if self.gpu_keys is None else int(self.gpu_keys.shape[-2]),
            'stored_cpu_seq_len': 0 if self.cpu_keys is None else int(self.cpu_keys.shape[-2]),
        }

    def get_cache_view(self) -> dict[str, Any]:
        keys, values = self.materialize_dense()
        return {
            'layout': 'cpu_offload',
            'key_states': keys,
            'value_states': values,
            'seq_length': self.cumulative_length,
            '_cpu_offload_layer': self,
        }

    def release(self) -> None:
        self.cpu_keys = None
        self.cpu_values = None
        self.gpu_keys = None
        self.gpu_values = None


class CPUOffloadCache(Cache):
    def __init__(
        self,
        config,
        gpu_keep_ratio: float = 0.75,
        pin_memory: bool = True,
        min_offload_tokens: int = 1,
    ):
        config = config.get_text_config(decoder=True)
        self.gpu_keep_ratio = float(gpu_keep_ratio)
        self.pin_memory = bool(pin_memory)
        self.min_offload_tokens = int(min_offload_tokens)
        layers = [
            CPUOffloadLayer(
                gpu_keep_ratio=gpu_keep_ratio,
                pin_memory=pin_memory,
                min_offload_tokens=min_offload_tokens,
            )
            for _ in range(config.num_hidden_layers)
        ]
        super().__init__(layers=layers)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.layer_class_to_replicate is not None:
            while len(self.layers) <= layer_idx:
                self.layers.append(self.layer_class_to_replicate())
        return self.layers[layer_idx].update(key_states, value_states, cache_kwargs)

    def get_layer_cache_view(self, layer_idx: int) -> dict[str, Any]:
        return self.layers[layer_idx].get_cache_view()

    def materialize_layer(self, layer_idx: int):
        return self.layers[layer_idx].materialize_dense()

    def get_memory(self) -> dict[str, float]:
        total = {
            'kv_gpu_cache_memory': 0,
            'kv_cpu_cache_memory': 0,
            'kv_original_memory': 0,
            'gpu_cache_ratio': 0.0,
            'cpu_offload_ratio': 0.0,
            'materialize_calls': 0,
            'materialize_time': 0.0,
            'offload_calls': 0,
            'offloaded_tokens': 0,
            'stored_gpu_seq_len': 0,
            'stored_cpu_seq_len': 0,
        }
        for layer in self.layers:
            info = layer.get_memory()
            total['kv_gpu_cache_memory'] += info['gpu_cache_memory']
            total['kv_cpu_cache_memory'] += info['cpu_cache_memory']
            total['kv_original_memory'] += info['original_memory']
            total['materialize_calls'] += info['materialize_calls']
            total['materialize_time'] += info['materialize_time']
            total['offload_calls'] += info['offload_calls']
            total['offloaded_tokens'] += info['offloaded_tokens']
            total['stored_gpu_seq_len'] += info['stored_gpu_seq_len']
            total['stored_cpu_seq_len'] += info['stored_cpu_seq_len']
        original = total['kv_original_memory']
        total['gpu_cache_ratio'] = 0.0 if original == 0 else total['kv_gpu_cache_memory'] / original
        total['cpu_offload_ratio'] = 0.0 if original == 0 else total['kv_cpu_cache_memory'] / original
        total['kv_gpu_cache_memory'] /= 1024 ** 3
        total['kv_cpu_cache_memory'] /= 1024 ** 3
        total['kv_original_memory'] /= 1024 ** 3
        return total

    def synchronize_and_release(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        for layer in self.layers:
            layer.release()
