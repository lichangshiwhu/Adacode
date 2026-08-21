import argparse
import glob
import json
import os
import sys
from collections import Counter

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from kv_compression.best_bit_analysis_kv_cache import (
    get_freq_by_tensor_names,
    infer_tensor_names_from_layer_freq,
)


FORMAT_BITS = {
    "bf16": {
        "original_bits": 16,
        "sign_bits": 1,
        "mantissa_bits": 7,
    },
    "fp16": {
        "original_bits": 16,
        "sign_bits": 1,
        "mantissa_bits": 10,
    },
    "e4m3": {
        "original_bits": 8,
        "sign_bits": 1,
        "mantissa_bits": 3,
    },
    "e5m2": {
        "original_bits": 8,
        "sign_bits": 1,
        "mantissa_bits": 2,
    },
}


def _infer_format_and_model_name(file_path):
    file_name = os.path.basename(file_path)
    stem = file_name[:-len("_exp.json")] if file_name.endswith("_exp.json") else os.path.splitext(file_name)[0]
    if not stem.startswith("kv_"):
        return "unknown", stem

    parts = stem.split("_", 2)
    if len(parts) < 3:
        return "unknown", stem
    return parts[1], parts[2]


def _get_format_bits(fmt):
    if fmt not in FORMAT_BITS:
        raise KeyError(f"Unsupported format {fmt}. Supported formats: {list(FORMAT_BITS.keys())}")
    return FORMAT_BITS[fmt]


def _best_result_for_single_list(data_list, title, fmt, is_coo=False):
    del title
    bits_cfg = _get_format_bits(fmt)
    original_bits = bits_cfg["original_bits"]
    sign_bits = bits_cfg["sign_bits"]
    mantissa_bits = bits_cfg["mantissa_bits"]

    total_numbers = sum(data_list)
    original_memory = total_numbers * original_bits
    best_nbit = None
    best_right = None
    best_compressed_memory = None
    sorted_indexed = sorted(enumerate(data_list), key=lambda x: x[1], reverse=True)

    for nbit in range(1, 9):
        number_space = 2 ** nbit if is_coo else 2 ** nbit - 1
        topk = sorted_indexed[:number_space]
        kept_values = sum(value for _, value in topk)
        outlier_values = total_numbers - kept_values
        compressed_memory = total_numbers * (nbit + sign_bits + mantissa_bits)
        compressed_memory += outlier_values * original_bits
        if is_coo:
            compressed_memory += outlier_values * 32

        if best_compressed_memory is None or compressed_memory < best_compressed_memory:
            best_compressed_memory = compressed_memory
            best_nbit = nbit
            best_right = max(index for index, _ in topk) if topk else None

    compress_ratio = best_compressed_memory / original_memory
    return {
        "best_right": best_right,
        "best_nbit": best_nbit,
        "total_numbers": total_numbers,
        "original_memory": original_memory,
        "best_compressed_memory": best_compressed_memory,
        "compress_ratio": compress_ratio,
        "sign_bits": sign_bits,
        "mantissa_bits": mantissa_bits,
        "original_bits": original_bits,
    }


def _analyze_dataset_layers(dataset_freq, infer_stage, fmt, is_coo=False):
    layer_ids = sorted(dataset_freq.keys(), key=lambda x: int(x))
    if not layer_ids:
        return {"backend": "empty", "tensor_names": [], "items": [], "summary": {}}

    tensor_names = infer_tensor_names_from_layer_freq(dataset_freq[layer_ids[0]])
    items = []
    for layer_id in layer_ids:
        layer_freq = dataset_freq[layer_id]
        tensor_freqs = get_freq_by_tensor_names(layer_freq, infer_stage, tensor_names=tensor_names)
        for tensor_name, freq_list in tensor_freqs.items():
            item = _best_result_for_single_list(
                freq_list,
                title=f"layer_{layer_id}_{tensor_name}",
                fmt=fmt,
                is_coo=is_coo,
            )
            item["layer"] = int(layer_id)
            item["tensor_name"] = tensor_name
            items.append(item)

    total_original_memory = sum(item["original_memory"] for item in items)
    total_best_compressed_memory = sum(item["best_compressed_memory"] for item in items)
    dominant_nbit = Counter(item["best_nbit"] for item in items).most_common(1)[0][0]
    dominant_right = Counter(item["best_right"] for item in items).most_common(1)[0][0]
    weighted_compress_ratio = total_best_compressed_memory / total_original_memory

    return {
        "backend": "traditional_kv" if tensor_names == ["keys", "values"] else "mla",
        "tensor_names": tensor_names,
        "items": items,
        "summary": {
            "num_layers": len(layer_ids),
            "num_tensor_series": len(items),
            "total_original_memory": total_original_memory,
            "total_best_compressed_memory": total_best_compressed_memory,
            "weighted_compress_ratio": weighted_compress_ratio,
            "average_compress_ratio": sum(item["compress_ratio"] for item in items) / len(items),
            "dominant_nbit": dominant_nbit,
            "dominant_right": dominant_right,
        },
    }


def analyze_json_file(file_path, stages, dataset_name=None, is_coo=False):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if dataset_name is None:
        dataset_name = list(data.keys())[0]
    if dataset_name not in data:
        raise KeyError(f"Dataset {dataset_name} not found in {file_path}. Available: {list(data.keys())}")

    fmt, model_name = _infer_format_and_model_name(file_path)
    dataset_freq = data[dataset_name]
    stage_results = {
        stage: _analyze_dataset_layers(dataset_freq, stage, fmt=fmt, is_coo=is_coo)
        for stage in stages
    }
    return {
        "file_path": file_path,
        "format": fmt,
        "model_name": model_name,
        "dataset_name": dataset_name,
        "stages": stage_results,
    }


def _print_summary(results, stages):
    for result in results:
        print(f"[{result['format']}] {result['model_name']}  dataset={result['dataset_name']}")
        for stage in stages:
            summary = result["stages"][stage]["summary"]
            print(
                f"  - {stage}: ratio={summary['weighted_compress_ratio']:.9f}, "
                f"dominant_nbit={summary['dominant_nbit']}, dominant_right={summary['dominant_right']}, "
                f"series={summary['num_tensor_series']}"
            )
        print()


def _print_merge_ratio_table(results):
    stage = "merge"
    preferred_formats = ["bf16", "fp16", "e4m3", "e5m2"]
    formats = []
    models = []
    table = {}

    for result in results:
        fmt = result["format"]
        model = result["model_name"]
        ratio = result["stages"][stage]["summary"]["weighted_compress_ratio"]
        if fmt not in formats:
            formats.append(fmt)
        if model not in models:
            models.append(model)
        table[(fmt, model)] = ratio

    formats = [fmt for fmt in preferred_formats if fmt in formats] + [
        fmt for fmt in formats if fmt not in preferred_formats
    ]

    header = ["dtype"] + models
    rows = [header]
    for fmt in formats:
        row = [fmt]
        for model in models:
            value = table.get((fmt, model))
            row.append(f"{value:.9f}" if value is not None else "-")
        rows.append(row)

    col_widths = [max(len(row[idx]) for row in rows) for idx in range(len(header))]
    for row in rows:
        print(" | ".join(cell.ljust(col_widths[idx]) for idx, cell in enumerate(row)))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze best compression ratios from exp-frequency JSON files using analysis_best_bits logic."
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=[
            "model_configs/kv_bf16_*_exp.json",
            "model_configs/kv_fp16_*_exp.json",
            "model_configs/kv_e4m3_*_exp.json",
            "model_configs/kv_e5m2_*_exp.json",
        ],
        help="Glob patterns for exp-frequency JSON files.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["merge"],
        choices=["merge", "prefill", "decode"],
        help="Inference stages to analyze.",
    )
    parser.add_argument(
        "--dataset-name",
        default="c4",
        help="Dataset key inside the JSON. Defaults to the first dataset in each file.",
    )
    parser.add_argument(
        "--is-coo",
        action="store_true",
        help="Use the COO-style cost model, consistent with analysis_best_bits(..., is_coo=True).",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to export the full analysis result as JSON.",
    )
    parser.add_argument(
        "--print-merge-table",
        action="store_true",
        help="Print a merge-stage table with rows=dtype and columns=model_name.",
    )
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
        analyze_json_file(
            file_path=file_path,
            stages=args.stages,
            dataset_name=args.dataset_name,
            is_coo=args.is_coo,
        )
        for file_path in file_paths
    ]

    if args.print_merge_table:
        _print_merge_ratio_table(results)
    else:
        _print_summary(results, args.stages)

    if args.output_json:
        output_dir = os.path.dirname(args.output_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved JSON report to {args.output_json}")


if __name__ == "__main__":
    main()
