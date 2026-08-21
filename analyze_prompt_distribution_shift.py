import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def normalize(counts: list[int] | np.ndarray) -> np.ndarray:
    values = np.asarray(counts, dtype=np.float64)
    total = values.sum()
    if total <= 0:
        return np.ones_like(values) / len(values)
    return values / total


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    mask = (p > 0) & (q > 0)
    return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    midpoint = (p + q) / 2
    return 0.5 * kl_divergence(p, midpoint) + 0.5 * kl_divergence(q, midpoint)


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(p - q).sum())


def wasserstein_on_exponents(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.abs(np.cumsum(p) - np.cumsum(q)).sum())


def mean_exponent(p: np.ndarray) -> float:
    return float((np.arange(len(p), dtype=np.float64) * p).sum())


def top_set(p: np.ndarray, k: int) -> set[int]:
    return set(np.argsort(p)[::-1][:k].tolist())


def best_contiguous_window(p: np.ndarray, window_size: int) -> tuple[int, float]:
    best_start = 0
    best_mass = -1.0
    for start in range(0, len(p) - window_size + 1):
        mass = float(p[start : start + window_size].sum())
        if mass > best_mass:
            best_start = start
            best_mass = mass
    return best_start, best_mass


def combine_layer_counts(layer: dict[str, list[int]], names: tuple[str, ...]) -> np.ndarray:
    merged = np.zeros(len(layer[names[0]]), dtype=np.float64)
    for name in names:
        merged += np.asarray(layer[name], dtype=np.float64)
    return merged


def metric_for_layer(layer: dict[str, list[int]], group: str, window_size: int) -> dict[str, float]:
    if group == "keys":
        pre_counts = combine_layer_counts(layer, ("prefill_keys",))
        decode_counts = combine_layer_counts(layer, ("decode_keys",))
    elif group == "values":
        pre_counts = combine_layer_counts(layer, ("prefill_values",))
        decode_counts = combine_layer_counts(layer, ("decode_values",))
    elif group == "merged":
        pre_counts = combine_layer_counts(layer, ("prefill_keys", "prefill_values"))
        decode_counts = combine_layer_counts(layer, ("decode_keys", "decode_values"))
    else:
        raise ValueError(f"Unsupported group: {group}")

    pre = normalize(pre_counts)
    decode = normalize(decode_counts)

    pre_start, pre_best_mass = best_contiguous_window(pre, window_size)
    decode_start, decode_best_mass = best_contiguous_window(decode, window_size)
    decode_mass_on_prefill_window = float(decode[pre_start : pre_start + window_size].sum())

    return {
        "js": js_divergence(pre, decode),
        "tv": total_variation(pre, decode),
        "w1": wasserstein_on_exponents(pre, decode),
        "mean_exp_shift": abs(mean_exponent(pre) - mean_exponent(decode)),
        "top_window_overlap": len(top_set(pre, window_size) & top_set(decode, window_size)) / window_size,
        "prefill_best_start": float(pre_start),
        "decode_best_start": float(decode_start),
        "best_start_shift": abs(float(pre_start) - float(decode_start)),
        "prefill_best_mass": pre_best_mass,
        "decode_best_mass": decode_best_mass,
        "decode_mass_on_prefill_window": decode_mass_on_prefill_window,
        "decode_gain_if_adjusted": decode_best_mass - decode_mass_on_prefill_window,
    }


def summarize_metrics(metrics: list[dict[str, float]]) -> tuple[dict[str, float], dict[str, float]]:
    keys = metrics[0].keys()
    avg = {key: float(np.mean([item[key] for item in metrics])) for key in keys}
    max_values = {key: float(np.max([item[key] for item in metrics])) for key in keys}
    return avg, max_values


def analyze_file(path: Path, dataset_name: str | None, window_size: int) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if dataset_name is None:
        dataset_name = next(iter(data))
    layers = data[dataset_name]

    rows = []
    for group in ("keys", "values", "merged"):
        metrics = [
            metric_for_layer(layers[layer_id], group, window_size)
            for layer_id in sorted(layers.keys(), key=lambda value: int(value))
        ]
        avg, max_values = summarize_metrics(metrics)
        rows.append(
            {
                "file": path.name,
                "dataset": dataset_name,
                "group": group,
                "layers": len(metrics),
                "average": avg,
                "max": max_values,
                "shifted_layers": sum(1 for item in metrics if item["best_start_shift"] >= 1),
            }
        )
    return rows


def print_markdown(rows: list[dict[str, Any]]) -> None:
    print("| Model | Tensor | JS avg | TV avg | W1 avg | Mean exp shift | Top-window overlap | Window start shift | Decode gain if adjusted |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        model = row["file"].replace("kv_prompt_test_", "").replace("_exp.json", "")
        avg = row["average"]
        print(
            f"| {model} | {row['group']} | {avg['js']:.4f} | {avg['tv']:.4f} | "
            f"{avg['w1']:.3f} | {avg['mean_exp_shift']:.3f} | "
            f"{avg['top_window_overlap']:.2f} | {avg['best_start_shift']:.2f} | "
            f"{100 * avg['decode_gain_if_adjusted']:.2f}% |"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze prefill/decode exponent-distribution shift in export_kvcache_status.py outputs."
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=["model_configs/kv_prompt_test*_exp.json"],
        help="JSON glob patterns to analyze.",
    )
    parser.add_argument("--dataset-name", default=None, help="Dataset key. Defaults to the first key in each file.")
    parser.add_argument("--window-size", type=int, default=8, help="Contiguous exponent window size.")
    parser.add_argument("--output-json", default=None, help="Optional JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths: list[Path] = []
    for pattern in args.patterns:
        paths.extend(Path().glob(pattern))
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError(f"No files matched: {args.patterns}")

    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(analyze_file(path, args.dataset_name, args.window_size))
    print_markdown(rows)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True) if output_path.parent != Path(".") else None
        output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nSaved JSON report to {output_path}")


if __name__ == "__main__":
    main()
