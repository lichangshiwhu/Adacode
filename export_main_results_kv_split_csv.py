import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any

from export_approx_kv_split_ratio import split_sample_memory


DEFAULT_INPUT_DIR = Path(r"E:\lichangshi\research\kv_compression\configs\main_results")
DEFAULT_OUTPUT_CSV = Path("main_results_kv_split_ratio_approx.csv")


def safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def parse_file_name(path: Path) -> dict[str, str]:
    name = path.name
    match = re.match(
        r"^kv_test_(?P<model>.+?)_False_True_adjuster_True_steps(?P<steps>\d+)_(?P<experiment>kv_(?:long|short))_info\.json$",
        name,
    )
    if match:
        return match.groupdict()
    return {
        "model": path.stem,
        "steps": "",
        "experiment": "",
    }


def summarize_split_rows(split_rows: list[dict[str, float]]) -> dict[str, float | int]:
    return {
        "count": len(split_rows),
        "key_compressed_memory_gb": safe_mean([row["key_compressed_memory"] for row in split_rows]),
        "value_compressed_memory_gb": safe_mean([row["value_compressed_memory"] for row in split_rows]),
        "key_original_memory_gb": safe_mean([row["key_original_memory"] for row in split_rows]),
        "value_original_memory_gb": safe_mean([row["value_original_memory"] for row in split_rows]),
        "key_compression_ratio": safe_mean([row["key_compression_ratio"] for row in split_rows]),
        "value_compression_ratio": safe_mean([row["value_compression_ratio"] for row in split_rows]),
        "kv_compressed_memory_gb": safe_mean([row["kv_compressed_memory"] for row in split_rows]),
        "kv_original_memory_gb": safe_mean([row["kv_original_memory"] for row in split_rows]),
        "kv_compression_ratio": safe_mean([row["kv_compression_ratio"] for row in split_rows]),
        "keys_outlier_ratio": safe_mean([row["keys_outlier_ratios"] for row in split_rows]),
        "values_outlier_ratio": safe_mean([row["values_outlier_ratios"] for row in split_rows]),
    }


def summarize_dataset(
    samples: list[Any],
    *,
    base_bits: float,
    original_bits: float,
    outlier_extra_bits: float,
) -> dict[str, float | int]:
    split_rows = []
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        split = split_sample_memory(
            sample,
            base_bits=base_bits,
            original_bits=original_bits,
            outlier_extra_bits=outlier_extra_bits,
        )
        if split is not None:
            split_rows.append(split)
    return summarize_split_rows(split_rows)


def rows_for_file(
    path: Path,
    *,
    base_bits: float,
    original_bits: float,
    outlier_extra_bits: float,
    include_overall: bool,
) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return []

    meta = parse_file_name(path)
    rows: list[dict[str, Any]] = []
    all_split_rows: list[dict[str, float]] = []

    for dataset, samples in sorted(data.items()):
        if not isinstance(samples, list):
            continue
        summary = summarize_dataset(
            samples,
            base_bits=base_bits,
            original_bits=original_bits,
            outlier_extra_bits=outlier_extra_bits,
        )
        if summary["count"] <= 0:
            continue
        rows.append(
            {
                **meta,
                "dataset": dataset,
                "source_file": path.name,
                **summary,
            }
        )

        for sample in samples:
            if not isinstance(sample, dict):
                continue
            split = split_sample_memory(
                sample,
                base_bits=base_bits,
                original_bits=original_bits,
                outlier_extra_bits=outlier_extra_bits,
            )
            if split is not None:
                all_split_rows.append(split)

    if include_overall and all_split_rows:
        rows.append(
            {
                **meta,
                "dataset": "overall",
                "source_file": path.name,
                **summarize_split_rows(all_split_rows),
            }
        )
    return rows


def build_rows(
    input_dir: Path,
    *,
    pattern: str,
    base_bits: float,
    original_bits: float,
    outlier_extra_bits: float,
    include_overall: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(input_dir.glob(pattern)):
        rows.extend(
            rows_for_file(
                path,
                base_bits=base_bits,
                original_bits=original_bits,
                outlier_extra_bits=outlier_extra_bits,
                include_overall=include_overall,
            )
        )
    return rows


def format_float(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.9f}"
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "model",
        "experiment",
        "steps",
        "dataset",
        "count",
        "key_compression_ratio",
        "value_compression_ratio",
        "kv_compression_ratio",
        "key_compression_ratio_percent",
        "value_compression_ratio_percent",
        "kv_compression_ratio_percent",
        "key_compressed_memory_gb",
        "value_compressed_memory_gb",
        "kv_compressed_memory_gb",
        "key_original_memory_gb",
        "value_original_memory_gb",
        "kv_original_memory_gb",
        "keys_outlier_ratio",
        "values_outlier_ratio",
        "source_file",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output["key_compression_ratio_percent"] = 100 * float(row["key_compression_ratio"])
            output["value_compression_ratio_percent"] = 100 * float(row["value_compression_ratio"])
            output["kv_compression_ratio_percent"] = 100 * float(row["kv_compression_ratio"])
            writer.writerow({key: format_float(output.get(key, "")) for key in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Approximate K/V compression ratios for all main_results *_info.json files "
            "and export a CSV table."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--pattern", default="*_info.json")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--no-overall", action="store_true", help="Do not add an overall row per result file.")
    parser.add_argument("--base-bits", type=float, default=11.0, help="Bits per non-outlier BF16 value.")
    parser.add_argument("--original-bits", type=float, default=16.0, help="Original BF16 bits per value.")
    parser.add_argument("--outlier-extra-bits", type=float, default=8.0, help="Extra bits for an outlier exponent.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_rows(
        args.input_dir,
        pattern=args.pattern,
        base_bits=args.base_bits,
        original_bits=args.original_bits,
        outlier_extra_bits=args.outlier_extra_bits,
        include_overall=not args.no_overall,
    )
    write_csv(args.output_csv, rows)
    print(f"Processed {len(rows)} rows from {args.input_dir}")
    print(f"Saved CSV to {args.output_csv}")


if __name__ == "__main__":
    main()
