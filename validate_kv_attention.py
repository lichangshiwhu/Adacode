import argparse
import json
import time
from typing import Any, Callable, Optional

import torch

from kv_compression.ops.kv_attention import (
    _infer_materialized_num_key_value_groups,
    dense_reference_attention,
    materialize_dense_from_cache_view,
)


def attention_validation_metrics(
    actual_output: torch.Tensor,
    reference_output: torch.Tensor,
    atol: float = 1e-2,
    rtol: float = 0.0,
) -> dict[str, Any]:
    diff = (actual_output - reference_output).abs().float()
    max_abs_diff = diff.max().item() if diff.numel() > 0 else 0.0
    mean_abs_diff = diff.mean().item() if diff.numel() > 0 else 0.0
    return {
        'max_abs_diff': max_abs_diff,
        'mean_abs_diff': mean_abs_diff,
        'allclose': bool(torch.allclose(actual_output, reference_output, atol=atol, rtol=rtol)),
        'atol': float(atol),
        'rtol': float(rtol),
    }


def _load_tensor(path: str, map_location: Optional[str] = None) -> Any:
    return torch.load(path, map_location=map_location)


def _load_cache_view(path: str, device: torch.device) -> dict[str, Any]:
    cache_view = _load_tensor(path, map_location='cpu')

    def _move_value(key: Optional[str], value: Any):
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if key == 'device':
            return device
        if isinstance(value, dict):
            return {child_key: _move_value(child_key, child_value) for child_key, child_value in value.items()}
        if isinstance(value, list):
            return [_move_value(None, item) for item in value]
        if isinstance(value, tuple):
            return tuple(_move_value(None, item) for item in value)
        return value

    return _move_value(None, cache_view)


def _estimate_tensor_bytes(tensor: Optional[torch.Tensor]) -> int:
    if tensor is None:
        return 0
    return tensor.numel() * tensor.element_size()


def _materialize_dense_reference(
    query_states: torch.Tensor,
    cache_view: dict[str, Any],
    attention_mask: Optional[torch.Tensor],
    scaling: Optional[float],
    num_key_value_groups: int,
) -> torch.Tensor:
    key_states = materialize_dense_from_cache_view(cache_view, 'key')
    value_states = materialize_dense_from_cache_view(cache_view, 'value')
    if key_states is None or value_states is None:
        raise ValueError('No dense key/value states available for materialized reference')
    materialized_num_key_value_groups = _infer_materialized_num_key_value_groups(
        query_states,
        key_states,
        num_key_value_groups,
    )
    return dense_reference_attention(
        query_states,
        key_states,
        value_states,
        attention_mask=attention_mask,
        scaling=scaling,
        num_key_value_groups=materialized_num_key_value_groups,
    )


def _load_dense_states(
    path: Optional[str],
    cache_view: dict[str, Any],
    kind: str,
    device: torch.device,
) -> torch.Tensor:
    if path:
        return _load_tensor(path, map_location=device)
    dense = materialize_dense_from_cache_view(cache_view, kind)
    if dense is None:
        raise ValueError(f'Unable to materialize dense {kind} states from cache view')
    return dense


def _profile_call(fn: Callable[[], Any], warmup: int, iters: int) -> dict[str, Any]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    begin = time.perf_counter()
    for _ in range(iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - begin) * 1000.0
    return {
        'warmup': int(warmup),
        'iters': int(iters),
        'total_ms': elapsed_ms,
        'avg_ms': elapsed_ms / max(iters, 1),
    }


def _profile_call_or_error(fn: Callable[[], Any], warmup: int, iters: int) -> dict[str, Any]:
    try:
        return {'status': 'ok', **_profile_call(fn, warmup=warmup, iters=iters)}
    except Exception as exc:
        return {
            'status': 'failed',
            'error_type': type(exc).__name__,
            'error': str(exc),
        }


def _validate_materialize_mode(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: Optional[float],
    num_key_value_groups: int,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    actual = dense_reference_attention(
        query_states,
        key_states,
        value_states,
        attention_mask=attention_mask,
        scaling=scaling,
        num_key_value_groups=num_key_value_groups,
    )
    reference = dense_reference_attention(
        query_states,
        key_states,
        value_states,
        attention_mask=attention_mask,
        scaling=scaling,
        num_key_value_groups=num_key_value_groups,
    )
    return {
        'mode': 'materialize',
        'metrics': attention_validation_metrics(actual, reference, atol=atol, rtol=rtol),
    }

def _profile_modes(
    query_states: torch.Tensor,
    cache_view: dict[str, Any],
    attention_mask: Optional[torch.Tensor],
    scaling: Optional[float],
    num_key_value_groups: int,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    materialize_profile = _profile_call_or_error(
        lambda: _materialize_dense_reference(
            query_states,
            cache_view,
            attention_mask=attention_mask,
            scaling=scaling,
            num_key_value_groups=num_key_value_groups,
        ),
        warmup=warmup,
        iters=iters,
    )
    return {
        'requested_profiles': ['materialize'],
        'profiles': {
            'materialize_dense_reference': materialize_profile,
        },
    }


def main():
    parser = argparse.ArgumentParser(description='Validate compressed KV attention outputs against dense reference.')
    parser.add_argument('--query_states', type=str, required=True, help='Path to torch-saved query_states tensor.')
    parser.add_argument('--cache_view', type=str, required=True, help='Path to torch-saved cache view dict.')
    parser.add_argument('--attention_mask', type=str, default=None, help='Optional path to torch-saved attention_mask tensor.')
    parser.add_argument('--key_states', type=str, default=None, help='Optional path to torch-saved dense key_states tensor.')
    parser.add_argument('--value_states', type=str, default=None, help='Optional path to torch-saved dense value_states tensor.')
    parser.add_argument('--mode', choices=['materialize', 'profile'], required=True)
    parser.add_argument('--scaling', type=float, default=None)
    parser.add_argument('--num_key_value_groups', type=int, default=1)
    parser.add_argument('--atol', type=float, default=1e-2)
    parser.add_argument('--rtol', type=float, default=0.0)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--json', action='store_true', help='Print JSON only.')
    args = parser.parse_args()

    device = torch.device(args.device)
    query_states = _load_tensor(args.query_states, map_location=device)
    cache_view = _load_cache_view(args.cache_view, device)
    attention_mask = _load_tensor(args.attention_mask, map_location=device) if args.attention_mask else None

    if query_states.shape[-2] != 1:
        raise ValueError(f'Decode attention validation expects q_len == 1, got {query_states.shape[-2]}')

    if args.mode == 'materialize':
        key_states = _load_dense_states(args.key_states, cache_view, 'key', device)
        value_states = _load_dense_states(args.value_states, cache_view, 'value', device)
        result = _validate_materialize_mode(
            query_states,
            key_states,
            value_states,
            attention_mask,
            args.scaling,
            args.num_key_value_groups,
            args.atol,
            args.rtol,
        )
    else:
        result = {
            'mode': 'profile',
            'efficiency_diagnostics': _profile_modes(
                query_states,
                cache_view,
                attention_mask,
                args.scaling,
                args.num_key_value_groups,
                warmup=args.warmup,
                iters=args.iters,
            ),
        }

    if args.json:
        print(json.dumps(result, ensure_ascii=False))
        return
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
