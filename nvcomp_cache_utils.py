from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import time

import torch
from transformers.cache_utils import DynamicLayer
from transformers import Cache


try:
    import cupy as cp
except Exception:
    cp = None

try:
    from nvidia import nvcomp
except Exception:
    try:
        import nvidia.nvcomp as nvcomp
    except Exception:
        nvcomp = None


def _concat_seq_tensors(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return torch.cat([first, second], dim=-2).contiguous()


def _torch_to_nvcomp_array(tensor: torch.Tensor, device: torch.device | None = None):
    if nvcomp is None:
        raise RuntimeError(
            'nvidia.nvcomp is required for NvcompANSCache. Install nvCOMP Python bindings first.'
        )
    target_device = device if device is not None else tensor.device
    if not tensor.is_cuda or tensor.device != target_device:
        tensor = tensor.to(device=target_device)
    tensor = tensor.contiguous()
    if target_device.type == 'cuda' and target_device.index is not None:
        with torch.cuda.device(target_device):
            try:
                return nvcomp.as_array(tensor)
            except Exception:
                if cp is None:
                    raise RuntimeError(
                        'nvCOMP could not wrap the Torch tensor directly, and CuPy is not available as a fallback.'
                    )
                cupy_arr = cp.from_dlpack(torch.utils.dlpack.to_dlpack(tensor))
                return nvcomp.as_array(cupy_arr)
    try:
        return nvcomp.as_array(tensor)
    except Exception:
        if cp is None:
            raise RuntimeError(
                'nvCOMP could not wrap the Torch tensor directly, and CuPy is not available as a fallback.'
            )
        cupy_arr = cp.from_dlpack(torch.utils.dlpack.to_dlpack(tensor))
        return nvcomp.as_array(cupy_arr)


def _dlpack_to_torch(array) -> torch.Tensor:
    if isinstance(array, torch.Tensor):
        return array
    try:
        return torch.utils.dlpack.from_dlpack(array)
    except Exception:
        pass
    if cp is not None:
        try:
            cupy_arr = cp.asarray(array)
            return torch.utils.dlpack.from_dlpack(cupy_arr)
        except Exception:
            pass
    to_dlpack = getattr(array, 'to_dlpack', None)
    if to_dlpack is not None:
        try:
            return torch.utils.dlpack.from_dlpack(to_dlpack())
        except Exception:
            pass
    to_dlpack_old = getattr(array, 'toDlpack', None)
    if to_dlpack_old is not None:
        try:
            return torch.utils.dlpack.from_dlpack(to_dlpack_old())
        except Exception:
            pass
    raise RuntimeError(
        'Decoded nvCOMP array does not expose a usable Torch conversion path. '
        f'array_type={type(array)!r}'
    )


def _compressed_nbytes(compressed: Any) -> int:
    buffer_size = getattr(compressed, 'buffer_size', None)
    if buffer_size is not None:
        try:
            return int(buffer_size)
        except Exception:
            pass
    for attr in ('nbytes', 'capacity'):
        value = getattr(compressed, attr, None)
        if value is not None:
            try:
                return int(value)
            except Exception:
                pass
    size = getattr(compressed, 'size', None)
    if size is not None:
        item_size = getattr(compressed, 'item_size', 1)
        try:
            return int(size) * int(item_size)
        except Exception:
            pass
    try:
        return int(len(compressed))
    except Exception:
        return 0


def _make_ans_codec(device: torch.device, uncomp_chunk_size: int):
    if nvcomp is None:
        raise RuntimeError(
            'nvidia.nvcomp is required for NvcompANSCache. Install nvCOMP Python bindings first.'
        )
    device_id = torch.device(device).index
    if device_id is None:
        device_id = torch.cuda.current_device()
    if torch.cuda.is_available():
        with torch.cuda.device(int(device_id)):
            return nvcomp.Codec(
                algorithm='ANS',
                data_type='|u1',
                device_id=int(device_id),
                uncomp_chunk_size=int(uncomp_chunk_size),
            )
    return nvcomp.Codec(
        algorithm='ANS',
        data_type='|u1',
        device_id=int(device_id),
        uncomp_chunk_size=int(uncomp_chunk_size),
    )


@dataclass
class NvcompANSCompressedTensor:
    sign_mass_data: torch.Tensor
    exp_compressed: Any
    original_shape: tuple[int, ...]
    exp_numel: int
    device: torch.device
    codec: Any

    @property
    def n_elements(self) -> int:
        result = 1
        for dim in self.original_shape:
            result *= int(dim)
        return result

    def get_memory(self) -> dict[str, float]:
        sign_mass_bytes = self.sign_mass_data.numel() * self.sign_mass_data.element_size()
        exp_bytes = _compressed_nbytes(self.exp_compressed)
        original_bytes = self.n_elements * 2
        compressed_bytes = sign_mass_bytes + exp_bytes
        return {
            'compressed_memory': compressed_bytes,
            'original_memory': original_bytes,
            'ratio': 0.0 if original_bytes == 0 else compressed_bytes / original_bytes,
            'sign_mass_memory': sign_mass_bytes,
            'exp_compressed_memory': exp_bytes,
        }

    def materialize(self) -> torch.Tensor:
        exp_u8 = torch.empty((self.exp_numel,), device=self.device, dtype=torch.uint8)
        exp_out = _torch_to_nvcomp_array(exp_u8, device=self.device)
        if self.device.type == 'cuda' and self.device.index is not None:
            with torch.cuda.device(self.device):
                try:
                    self.codec.decode(self.exp_compressed, '|u1', out=exp_out)
                except Exception:
                    decoded_exp = self.codec.decode(self.exp_compressed, '|u1')
                    exp_u8 = _dlpack_to_torch(decoded_exp).to(device=self.device, dtype=torch.uint8).view(-1)
        else:
            try:
                self.codec.decode(self.exp_compressed, '|u1', out=exp_out)
            except Exception:
                decoded_exp = self.codec.decode(self.exp_compressed, '|u1')
                exp_u8 = _dlpack_to_torch(decoded_exp).to(device=self.device, dtype=torch.uint8).view(-1)
        if exp_u8.numel() != self.exp_numel:
            exp_u8 = exp_u8[: self.exp_numel]
        sign_mass_i32 = self.sign_mass_data.view(-1).to(torch.int32)
        exp_i32 = exp_u8.to(torch.int32)
        out_u16 = (((sign_mass_i32 & 0x0080) << 8) | (sign_mass_i32 & 0x007F) | (exp_i32 << 7)).to(torch.uint16)
        return out_u16.view(torch.bfloat16).view(self.original_shape).contiguous()


def compress_nvcomp_ans_tensor(
    original_data_bfloat16: torch.Tensor,
    codec: Any,
) -> NvcompANSCompressedTensor:
    if original_data_bfloat16.dtype != torch.bfloat16:
        original_data_bfloat16 = original_data_bfloat16.to(torch.bfloat16)
    original_data_bfloat16 = original_data_bfloat16.contiguous()
    raw_u16 = original_data_bfloat16.view(torch.uint16)
    raw_i32 = raw_u16.to(torch.int32)
    sign_mass_data = ((((raw_i32 >> 8) & 0x80) | (raw_i32 & 0x7F)).to(torch.uint8)).contiguous()
    exp_data = (((raw_i32 >> 7) & 0xFF).to(torch.uint8)).contiguous()
    exp_nvcomp = _torch_to_nvcomp_array(exp_data.view(-1), device=original_data_bfloat16.device)
    if original_data_bfloat16.device.type == 'cuda' and original_data_bfloat16.device.index is not None:
        with torch.cuda.device(original_data_bfloat16.device):
            exp_compressed = codec.encode(exp_nvcomp)
    else:
        exp_compressed = codec.encode(exp_nvcomp)
    return NvcompANSCompressedTensor(
        sign_mass_data=sign_mass_data.view(-1).contiguous(),
        exp_compressed=exp_compressed,
        original_shape=tuple(original_data_bfloat16.shape),
        exp_numel=int(exp_data.numel()),
        device=original_data_bfloat16.device,
        codec=codec,
    )


class NvcompANSLayer(DynamicLayer):
    def __init__(
        self,
        block_size: int = 128,
        uncomp_chunk_size: int = 65536,
    ):
        super().__init__()
        self.block_size = max(1, int(block_size))
        self.uncomp_chunk_size = int(uncomp_chunk_size)
        self.cumulative_length = 0
        self.cache_device: Optional[torch.device] = None
        self.cache_dtype: Optional[torch.dtype] = None
        self.codec = None
        self.blocks_key: list[NvcompANSCompressedTensor] = []
        self.blocks_value: list[NvcompANSCompressedTensor] = []
        self.residual_keys: Optional[torch.Tensor] = None
        self.residual_values: Optional[torch.Tensor] = None
        self.compress_calls = 0
        self.materialize_calls = 0
        self.compress_time = 0.0
        self.materialize_time = 0.0
        self.compressed_tokens = 0

    def _ensure_codec(self, device: torch.device) -> None:
        if self.codec is None:
            self.codec = _make_ans_codec(device, self.uncomp_chunk_size)

    def _append_residual(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        if key_states.dtype != torch.bfloat16:
            key_states = key_states.to(torch.bfloat16)
        if value_states.dtype != torch.bfloat16:
            value_states = value_states.to(torch.bfloat16)
        if self.residual_keys is None:
            self.residual_keys = key_states.contiguous()
            self.residual_values = value_states.contiguous()
            return
        self.residual_keys = _concat_seq_tensors(self.residual_keys, key_states)
        self.residual_values = _concat_seq_tensors(self.residual_values, value_states)

    def _flush_ready_blocks(self) -> bool:
        if self.residual_keys is None:
            return False
        flushed = False
        while self.residual_keys is not None and self.residual_keys.shape[-2] >= self.block_size:
            begin = time.perf_counter()
            block_keys = self.residual_keys[..., : self.block_size, :].contiguous()
            block_values = self.residual_values[..., : self.block_size, :].contiguous()
            remaining_keys = self.residual_keys[..., self.block_size :, :].contiguous()
            remaining_values = self.residual_values[..., self.block_size :, :].contiguous()
            if remaining_keys.shape[-2] == 0:
                remaining_keys = None
                remaining_values = None
            self.residual_keys = remaining_keys
            self.residual_values = remaining_values
            self.blocks_key.append(compress_nvcomp_ans_tensor(block_keys, self.codec))
            self.blocks_value.append(compress_nvcomp_ans_tensor(block_values, self.codec))
            self.compress_calls += 2
            self.compressed_tokens += int(block_keys.shape[-2])
            self.compress_time += time.perf_counter() - begin
            flushed = True
        return flushed

    def materialize_dense(self) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        begin = time.perf_counter()
        self.materialize_calls += 1
        key_parts = [block.materialize() for block in self.blocks_key]
        value_parts = [block.materialize() for block in self.blocks_value]
        if self.residual_keys is not None and self.residual_keys.shape[-2] > 0:
            key_parts.append(self.residual_keys)
            value_parts.append(self.residual_values)
        if not key_parts:
            self.materialize_time += time.perf_counter() - begin
            return self._empty_states(), self._empty_states()
        if len(key_parts) == 1:
            keys = key_parts[0].contiguous()
            values = value_parts[0].contiguous()
        else:
            keys = torch.cat(key_parts, dim=-2).contiguous()
            values = torch.cat(value_parts, dim=-2).contiguous()
        self.materialize_time += time.perf_counter() - begin
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
            self._ensure_codec(key_states.device)
        self._append_residual(key_states, value_states)
        self._flush_ready_blocks()
        return self.materialize_dense()

    def get_seq_length(self) -> int:
        return int(self.cumulative_length)

    def get_memory(self) -> dict[str, float]:
        compressed_memory = 0
        original_memory = 0
        sign_mass_memory = 0
        exp_compressed_memory = 0
        for block in self.blocks_key + self.blocks_value:
            info = block.get_memory()
            compressed_memory += info['compressed_memory']
            original_memory += info['original_memory']
            sign_mass_memory += info['sign_mass_memory']
            exp_compressed_memory += info['exp_compressed_memory']
        residual_memory = 0
        if self.residual_keys is not None:
            residual_memory += self.residual_keys.numel() * self.residual_keys.element_size()
            residual_memory += self.residual_values.numel() * self.residual_values.element_size()
        compressed_memory += residual_memory
        original_memory += residual_memory
        return {
            'compressed_memory': compressed_memory,
            'original_memory': original_memory,
            'ratio': 0.0 if original_memory == 0 else compressed_memory / original_memory,
            'sign_mass_memory': sign_mass_memory,
            'exp_compressed_memory': exp_compressed_memory,
            'residual_memory': residual_memory,
            'compress_calls': self.compress_calls,
            'materialize_calls': self.materialize_calls,
            'compress_time': self.compress_time,
            'materialize_time': self.materialize_time,
            'compressed_tokens': self.compressed_tokens,
            'residual_tokens': 0 if self.residual_keys is None else int(self.residual_keys.shape[-2]),
            'compressed_block_count': len(self.blocks_key),
        }

    def get_cache_view(self) -> dict[str, Any]:
        keys, values = self.materialize_dense()
        return {
            'layout': 'nvcomp_ans_exp',
            'key_states': keys,
            'value_states': values,
            'seq_length': self.cumulative_length,
            '_nvcomp_ans_layer': self,
        }

    def release(self) -> None:
        self.blocks_key = []
        self.blocks_value = []
        self.residual_keys = None
        self.residual_values = None


class NvcompANSCache(Cache):
    def __init__(
        self,
        config,
        block_size: int = 128,
        uncomp_chunk_size: int = 65536,
    ):
        config = config.get_text_config(decoder=True)
        self.block_size = int(block_size)
        self.uncomp_chunk_size = int(uncomp_chunk_size)
        layers = [
            NvcompANSLayer(
                block_size=block_size,
                uncomp_chunk_size=uncomp_chunk_size,
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
            'kv_compressed_memory': 0,
            'kv_original_memory': 0,
            'kv_compression_ratio': 0.0,
            'kv_sign_mass_memory': 0,
            'kv_exp_compressed_memory': 0,
            'kv_residual_memory': 0,
            'compress_calls': 0,
            'materialize_calls': 0,
            'compress_time': 0.0,
            'materialize_time': 0.0,
            'compressed_tokens': 0,
            'residual_tokens': 0,
            'compressed_block_count': 0,
        }
        for layer in self.layers:
            info = layer.get_memory()
            total['kv_compressed_memory'] += info['compressed_memory']
            total['kv_original_memory'] += info['original_memory']
            total['kv_sign_mass_memory'] += info['sign_mass_memory']
            total['kv_exp_compressed_memory'] += info['exp_compressed_memory']
            total['kv_residual_memory'] += info['residual_memory']
            total['compress_calls'] += info['compress_calls']
            total['materialize_calls'] += info['materialize_calls']
            total['compress_time'] += info['compress_time']
            total['materialize_time'] += info['materialize_time']
            total['compressed_tokens'] += info['compressed_tokens']
            total['residual_tokens'] += info['residual_tokens']
            total['compressed_block_count'] += info['compressed_block_count']
        original = total['kv_original_memory']
        total['kv_compression_ratio'] = 0.0 if original == 0 else total['kv_compressed_memory'] / original
        total['kv_compressed_memory'] /= 1024 ** 3
        total['kv_original_memory'] /= 1024 ** 3
        total['kv_sign_mass_memory'] /= 1024 ** 3
        total['kv_exp_compressed_memory'] /= 1024 ** 3
        total['kv_residual_memory'] /= 1024 ** 3
        return total

    def synchronize_and_release(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        for layer in self.layers:
            layer.release()
