import argparse
import glob
import json
import math
import os
from typing import Any

import numpy as np
from scipy import stats


DTYPE_CONFIGS = {
    "bf16": {
        "storage_bits": 16,
        "exp_bits": 8,
        "mantissa_bits": 7,
        "bias": 127,
    },
    "fp16": {
        "storage_bits": 16,
        "exp_bits": 5,
        "mantissa_bits": 10,
        "bias": 15,
    },
    "e4m3": {
        "storage_bits": 8,
        "exp_bits": 4,
        "mantissa_bits": 3,
        "bias": 7,
    },
    "e5m2": {
        "storage_bits": 8,
        "exp_bits": 5,
        "mantissa_bits": 2,
        "bias": 15,
    },
}

DTYPE_ALIASES = {
    "bfloat16": "bf16",
    "float16": "fp16",
    "half": "fp16",
    "fp8_e4m3fn": "e4m3",
    "float8_e4m3fn": "e4m3",
    "e4m3fn": "e4m3",
    "fp8_e5m2": "e5m2",
    "float8_e5m2": "e5m2",
}


def _normalize_dtype(dtype: str) -> str:
    dtype = DTYPE_ALIASES.get(dtype.lower(), dtype.lower())
    if dtype not in DTYPE_CONFIGS:
        raise ValueError(f"Unsupported dtype {dtype}. Supported: {sorted(DTYPE_CONFIGS)}")
    return dtype


def _infer_dtype_from_path(file_path: str) -> str:
    name = os.path.basename(file_path).lower()
    for marker, dtype in [
        ("kv_bf16_", "bf16"),
        ("kv_fp16_", "fp16"),
        ("kv_e4m3_", "e4m3"),
        ("kv_e5m2_", "e5m2"),
    ]:
        if marker in name:
            return dtype
    return "bf16"


def _infer_model_name(file_path: str) -> str:
    name = os.path.basename(file_path)
    suffix = "_all.json"
    stem = name[:-len(suffix)] if name.endswith(suffix) else os.path.splitext(name)[0]
    if stem.startswith("kv_"):
        parts = stem.split("_", 2)
        if len(parts) == 3:
            return parts[2]
    return stem


def raw_bit_values(dtype: str) -> tuple[np.ndarray, np.ndarray]:
    dtype = _normalize_dtype(dtype)
    cfg = DTYPE_CONFIGS[dtype]
    storage_bits = cfg["storage_bits"]
    exp_bits = cfg["exp_bits"]
    mantissa_bits = cfg["mantissa_bits"]
    bias = cfg["bias"]

    raw = np.arange(1 << storage_bits, dtype=np.uint32)
    sign = np.where((raw >> (storage_bits - 1)) & 0x1, -1.0, 1.0)
    exp_mask = (1 << exp_bits) - 1
    mantissa_mask = (1 << mantissa_bits) - 1
    exp = ((raw >> mantissa_bits) & exp_mask).astype(np.int32)
    mantissa = (raw & mantissa_mask).astype(np.float64)

    values = np.empty_like(mantissa, dtype=np.float64)
    is_zero_exp = exp == 0
    is_all_one_exp = exp == exp_mask
    is_normal = ~(is_zero_exp | is_all_one_exp)

    values[is_zero_exp] = sign[is_zero_exp] * np.ldexp(
        mantissa[is_zero_exp] / (1 << mantissa_bits),
        1 - bias,
    )
    values[is_normal] = sign[is_normal] * np.ldexp(
        1.0 + mantissa[is_normal] / (1 << mantissa_bits),
        exp[is_normal] - bias,
    )

    if dtype == "e4m3":
        # PyTorch float8_e4m3fn uses the all-ones exponent for finite values.
        values[is_all_one_exp] = sign[is_all_one_exp] * np.ldexp(
            1.0 + mantissa[is_all_one_exp] / (1 << mantissa_bits),
            exp[is_all_one_exp] - bias,
        )
        values[raw == 0x7F] = np.nan
        values[raw == 0xFF] = np.nan
    else:
        values[is_all_one_exp & (mantissa == 0)] = sign[is_all_one_exp & (mantissa == 0)] * np.inf
        values[is_all_one_exp & (mantissa != 0)] = np.nan

    finite = np.isfinite(values)
    return values, finite


def _combine_freq_lists(freq_lists: list[list[int]]) -> list[int]:
    if not freq_lists:
        return []
    merged = [0] * len(freq_lists[0])
    for freq in freq_lists:
        if len(freq) != len(merged):
            raise ValueError(f"Histogram length mismatch: {len(freq)} != {len(merged)}")
        for idx, count in enumerate(freq):
            merged[idx] += int(count)
    return merged


def _stage_tensor_freqs(layer_freq: dict[str, list[int]], stage: str) -> dict[str, list[int]]:
    if stage == "merge":
        grouped: dict[str, list[list[int]]] = {}
        for name, freq in layer_freq.items():
            if name.startswith("prefill_"):
                grouped.setdefault(name[len("prefill_"):], []).append(freq)
            elif name.startswith("decode_"):
                grouped.setdefault(name[len("decode_"):], []).append(freq)
            else:
                grouped.setdefault(name, []).append(freq)
        return {name: _combine_freq_lists(freqs) for name, freqs in grouped.items()}

    prefix = f"{stage}_"
    matched = {
        name[len(prefix):]: freq
        for name, freq in layer_freq.items()
        if name.startswith(prefix)
    }
    if matched:
        return matched
    return dict(layer_freq)


def _iter_series(data: dict[str, Any], dataset_name: str, stage: str):
    if dataset_name not in data:
        raise KeyError(f"Dataset {dataset_name} not found. Available: {list(data.keys())}")
    dataset_freq = data[dataset_name]
    for layer_id in sorted(dataset_freq.keys(), key=lambda x: int(x)):
        for tensor_name, freq in _stage_tensor_freqs(dataset_freq[layer_id], stage).items():
            yield int(layer_id), tensor_name, freq


def _hist_to_finite_arrays(freq: list[int], dtype: str, drop_zero: bool) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    values, finite_mask = raw_bit_values(dtype)
    counts = np.asarray(freq, dtype=np.float64)
    if counts.shape[0] != values.shape[0]:
        raise ValueError(f"Expected {values.shape[0]} buckets for {dtype}, got {counts.shape[0]}")

    total = float(counts.sum())
    finite_counts = counts[finite_mask]
    finite_values = values[finite_mask]
    if drop_zero:
        keep = finite_values != 0
        finite_values = finite_values[keep]
        finite_counts = finite_counts[keep]

    used_total = float(finite_counts.sum())
    zero_mask = finite_mask & (values == 0)
    return finite_values, finite_counts, {
        "total_count": total,
        "finite_count": float(counts[finite_mask].sum()),
        "used_count": used_total,
        "finite_fraction": 0.0 if total == 0 else float(counts[finite_mask].sum() / total),
        "zero_fraction": 0.0 if total == 0 else float(counts[zero_mask].sum() / total),
    }


def _quantile_sample(values: np.ndarray, counts: np.ndarray, sample_count: int) -> np.ndarray:
    total = int(counts.sum())
    if total <= 0:
        return np.empty((0,), dtype=np.float64)
    order = np.argsort(values)
    values = values[order]
    counts = counts[order]
    cdf_counts = np.cumsum(counts)
    n = min(int(sample_count), total)
    positions = (np.arange(n, dtype=np.float64) + 0.5) * total / n
    indices = np.searchsorted(cdf_counts, positions, side="left")
    return values[np.clip(indices, 0, len(values) - 1)]


def _empirical_quantiles(values: np.ndarray, counts: np.ndarray, probs: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    values = values[order]
    counts = counts[order]
    cdf_counts = np.cumsum(counts)
    total = cdf_counts[-1]
    indices = np.searchsorted(cdf_counts, probs * total, side="left")
    return values[np.clip(indices, 0, len(values) - 1)]


def _fit_alpha_stable(sample: np.ndarray, symmetric: bool) -> tuple[float, float, float, float]:
    sample = sample[np.isfinite(sample)]
    if sample.size < 32:
        raise ValueError("Need at least 32 finite samples to fit alpha-stable distribution")
    if np.all(sample == sample[0]):
        raise ValueError("Cannot fit alpha-stable distribution to a constant sample")
    if symmetric:
        alpha, beta, loc, scale = stats.levy_stable.fit(sample, fbeta=0)
    else:
        alpha, beta, loc, scale = stats.levy_stable.fit(sample)
    return float(alpha), float(beta), float(loc), float(scale)


def _distance_to_fitted(
    values: np.ndarray,
    counts: np.ndarray,
    params: tuple[float, float, float, float],
    quantile_points: int,
) -> dict[str, float]:
    alpha, beta, loc, scale = params
    occupied = counts > 0
    x = values[occupied]
    w = counts[occupied]
    order = np.argsort(x)
    x = x[order]
    w = w[order]
    empirical_cdf = np.cumsum(w) / w.sum()
    model_cdf = stats.levy_stable.cdf(x, alpha, beta, loc=loc, scale=scale)
    model_cdf = np.nan_to_num(model_cdf, nan=0.0, posinf=1.0, neginf=0.0)
    ks = float(np.max(np.abs(empirical_cdf - model_cdf)))

    probs = (np.arange(quantile_points, dtype=np.float64) + 0.5) / quantile_points
    empirical_q = _empirical_quantiles(values, counts, probs)
    model_q = stats.levy_stable.ppf(probs, alpha, beta, loc=loc, scale=scale)
    finite = np.isfinite(model_q)
    q_wasserstein = float(np.mean(np.abs(empirical_q[finite] - model_q[finite]))) if finite.any() else math.inf
    q25, q75 = _empirical_quantiles(values, counts, np.asarray([0.25, 0.75], dtype=np.float64))
    iqr = float(q75 - q25)
    normalized = q_wasserstein / iqr if iqr > 0 and math.isfinite(q_wasserstein) else math.inf

    return {
        "ks_distance": ks,
        "quantile_wasserstein": q_wasserstein,
        "normalized_quantile_wasserstein": float(normalized),
    }


def analyze_series(
    freq: list[int],
    dtype: str,
    max_fit_samples: int,
    quantile_points: int,
    symmetric: bool,
    drop_zero: bool,
) -> dict[str, Any]:
    values, counts, quality = _hist_to_finite_arrays(freq, dtype, drop_zero=drop_zero)
    if quality["used_count"] <= 0:
        return {**quality, "fit_ok": False, "error": "empty finite histogram"}

    sample = _quantile_sample(values, counts, max_fit_samples)
    params = _fit_alpha_stable(sample, symmetric=symmetric)
    distances = _distance_to_fitted(values, counts, params, quantile_points=quantile_points)
    alpha, beta, loc, scale = params
    return {
        **quality,
        "fit_ok": True,
        "alpha": alpha,
        "beta": beta,
        "loc": loc,
        "scale": scale,
        **distances,
    }


def _summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    ok_items = [item for item in items if item.get("fit_ok")]
    if not ok_items:
        return {"num_series": len(items), "num_fit_ok": 0}

    weights = np.asarray([item["used_count"] for item in ok_items], dtype=np.float64)
    weights = weights / weights.sum()

    def weighted_mean(key: str) -> float:
        return float(np.sum(weights * np.asarray([item[key] for item in ok_items], dtype=np.float64)))

    return {
        "num_series": len(items),
        "num_fit_ok": len(ok_items),
        "weighted_alpha": weighted_mean("alpha"),
        "weighted_beta": weighted_mean("beta"),
        "weighted_ks_distance": weighted_mean("ks_distance"),
        "weighted_quantile_wasserstein": weighted_mean("quantile_wasserstein"),
        "weighted_normalized_quantile_wasserstein": weighted_mean("normalized_quantile_wasserstein"),
        "max_ks_distance": float(max(item["ks_distance"] for item in ok_items)),
        "max_normalized_quantile_wasserstein": float(max(item["normalized_quantile_wasserstein"] for item in ok_items)),
        "total_used_count": float(sum(item["used_count"] for item in ok_items)),
    }


def analyze_file(
    file_path: str,
    dataset_name: str | None,
    stages: list[str],
    dtype: str | None,
    max_fit_samples: int,
    quantile_points: int,
    symmetric: bool,
    drop_zero: bool,
) -> dict[str, Any]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if dataset_name is None:
        dataset_name = next(iter(data.keys()))
    dtype = _normalize_dtype(dtype or _infer_dtype_from_path(file_path))

    stage_results = {}
    for stage in stages:
        items = []
        for layer, tensor_name, freq in _iter_series(data, dataset_name, stage):
            item = analyze_series(
                freq,
                dtype=dtype,
                max_fit_samples=max_fit_samples,
                quantile_points=quantile_points,
                symmetric=symmetric,
                drop_zero=drop_zero,
            )
            item["layer"] = layer
            item["tensor_name"] = tensor_name
            items.append(item)
        stage_results[stage] = {
            "items": items,
            "summary": _summary(items),
        }

    return {
        "file_path": file_path,
        "model_name": _infer_model_name(file_path),
        "dataset_name": dataset_name,
        "dtype": dtype,
        "fit_family": "symmetric_alpha_stable" if symmetric else "alpha_stable",
        "drop_zero": drop_zero,
        "stages": stage_results,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze distances between exported full raw-bit KV histograms and fitted alpha-stable distributions."
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=["model_configs/kv_*_all.json"],
        help="Glob patterns for all-bit JSON files exported by export_kvcache_status.py --freq-types all.",
    )
    parser.add_argument("--dataset-name", default=None, help="Dataset key inside JSON. Defaults to the first dataset.")
    parser.add_argument("--stages", nargs="+", default=["merge"], choices=["merge", "prefill", "decode"])
    parser.add_argument("--dtype", default=None, help="Override dtype: bf16, fp16, e4m3, e5m2.")
    parser.add_argument("--max-fit-samples", type=int, default=2000, help="Deterministic quantile samples used for fitting. Increase this for final reporting if runtime is acceptable.")
    parser.add_argument("--quantile-points", type=int, default=10000, help="Quantile points for Wasserstein-style distance.")
    parser.add_argument("--symmetric", action="store_true", help="Fit symmetric alpha-stable distribution by fixing beta=0.")
    parser.add_argument("--drop-zero", action="store_true", help="Exclude exact zeros before fitting and distance evaluation.")
    parser.add_argument("--output-json", default=None, help="Optional path to save the full analysis report.")
    return parser.parse_args()


def main():
    args = parse_args()
    file_paths = []
    for pattern in args.patterns:
        file_paths.extend(glob.glob(pattern))
    file_paths = sorted(set(file_paths))
    if not file_paths:
        raise FileNotFoundError(f"No JSON files matched patterns: {args.patterns}")

    results = [
        analyze_file(
            file_path=file_path,
            dataset_name=args.dataset_name,
            stages=args.stages,
            dtype=args.dtype,
            max_fit_samples=args.max_fit_samples,
            quantile_points=args.quantile_points,
            symmetric=args.symmetric,
            drop_zero=args.drop_zero,
        )
        for file_path in file_paths
    ]

    for result in results:
        print(f"[{result['dtype']}] {result['model_name']} dataset={result['dataset_name']} fit={result['fit_family']}")
        for stage in args.stages:
            summary = result["stages"][stage]["summary"]
            if summary.get("num_fit_ok", 0) == 0:
                print(f"  - {stage}: no valid fitted series")
                continue
            print(
                f"  - {stage}: alpha={summary['weighted_alpha']:.4f}, "
                f"beta={summary['weighted_beta']:.4f}, "
                f"KS={summary['weighted_ks_distance']:.6f}, "
                f"normQW={summary['weighted_normalized_quantile_wasserstein']:.6f}, "
                f"series={summary['num_fit_ok']}/{summary['num_series']}"
            )
        print()

    if args.output_json:
        output_dir = os.path.dirname(args.output_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved JSON report to {args.output_json}")


if __name__ == "__main__":
    main()
