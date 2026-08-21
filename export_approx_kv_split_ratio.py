import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def split_sample_memory(
    sample: dict[str, Any],
    *,
    base_bits: float,
    original_bits: float,
    outlier_extra_bits: float,
) -> dict[str, float] | None:
    total_compressed = sample.get("kv_compressed_memory")
    total_original = sample.get("kv_original_memory")
    key_outlier_ratio = sample.get("keys_outlier_ratios")
    value_outlier_ratio = sample.get("values_outlier_ratios")

    if (
        total_compressed is None
        or total_original is None
        or key_outlier_ratio is None
        or value_outlier_ratio is None
    ):
        return None

    total_compressed = float(total_compressed)
    total_original = float(total_original)
    key_outlier_ratio = float(key_outlier_ratio)
    value_outlier_ratio = float(value_outlier_ratio)
    if total_original <= 0:
        return None

    key_weight = base_bits / original_bits + outlier_extra_bits / original_bits * key_outlier_ratio
    value_weight = base_bits / original_bits + outlier_extra_bits / original_bits * value_outlier_ratio
    weight_sum = key_weight + value_weight
    if weight_sum <= 0:
        return None

    key_compressed = total_compressed * key_weight / weight_sum
    value_compressed = total_compressed - key_compressed
    key_original = total_original / 2.0
    value_original = total_original / 2.0

    return {
        "key_compressed_memory": key_compressed,
        "value_compressed_memory": value_compressed,
        "key_original_memory": key_original,
        "value_original_memory": value_original,
        "key_compression_ratio": key_compressed / key_original,
        "value_compression_ratio": value_compressed / value_original,
        "kv_compressed_memory": total_compressed,
        "kv_original_memory": total_original,
        "kv_compression_ratio": total_compressed / total_original,
        "keys_outlier_ratios": key_outlier_ratio,
        "values_outlier_ratios": value_outlier_ratio,
    }


def summarize_dataset(
    dataset: str,
    samples: list[Any],
    *,
    base_bits: float,
    original_bits: float,
    outlier_extra_bits: float,
) -> dict[str, Any]:
    split_rows = [
        split_sample_memory(
            sample,
            base_bits=base_bits,
            original_bits=original_bits,
            outlier_extra_bits=outlier_extra_bits,
        )
        for sample in samples
        if isinstance(sample, dict)
    ]
    split_rows = [row for row in split_rows if row is not None]

    return {
        "dataset": dataset,
        "count": len(split_rows),
        "key_compressed_memory": safe_mean([row["key_compressed_memory"] for row in split_rows]),
        "value_compressed_memory": safe_mean([row["value_compressed_memory"] for row in split_rows]),
        "key_original_memory": safe_mean([row["key_original_memory"] for row in split_rows]),
        "value_original_memory": safe_mean([row["value_original_memory"] for row in split_rows]),
        "key_compression_ratio": safe_mean([row["key_compression_ratio"] for row in split_rows]),
        "value_compression_ratio": safe_mean([row["value_compression_ratio"] for row in split_rows]),
        "kv_compressed_memory": safe_mean([row["kv_compressed_memory"] for row in split_rows]),
        "kv_original_memory": safe_mean([row["kv_original_memory"] for row in split_rows]),
        "kv_compression_ratio": safe_mean([row["kv_compression_ratio"] for row in split_rows]),
        "keys_outlier_ratios": safe_mean([row["keys_outlier_ratios"] for row in split_rows]),
        "values_outlier_ratios": safe_mean([row["values_outlier_ratios"] for row in split_rows]),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row["count"] > 0]
    return {
        "dataset": "overall",
        "count": sum(row["count"] for row in valid),
        "key_compressed_memory": safe_mean([row["key_compressed_memory"] for row in valid]),
        "value_compressed_memory": safe_mean([row["value_compressed_memory"] for row in valid]),
        "key_original_memory": safe_mean([row["key_original_memory"] for row in valid]),
        "value_original_memory": safe_mean([row["value_original_memory"] for row in valid]),
        "key_compression_ratio": safe_mean([row["key_compression_ratio"] for row in valid]),
        "value_compression_ratio": safe_mean([row["value_compression_ratio"] for row in valid]),
        "kv_compressed_memory": safe_mean([row["kv_compressed_memory"] for row in valid]),
        "kv_original_memory": safe_mean([row["kv_original_memory"] for row in valid]),
        "kv_compression_ratio": safe_mean([row["kv_compression_ratio"] for row in valid]),
        "keys_outlier_ratios": safe_mean([row["keys_outlier_ratios"] for row in valid]),
        "values_outlier_ratios": safe_mean([row["values_outlier_ratios"] for row in valid]),
    }


def build_rows(
    data: dict[str, Any],
    *,
    dataset_name: str | None,
    base_bits: float,
    original_bits: float,
    outlier_extra_bits: float,
) -> list[dict[str, Any]]:
    rows = []
    for dataset, samples in data.items():
        if dataset_name is not None and dataset != dataset_name:
            continue
        if not isinstance(samples, list):
            continue
        rows.append(
            summarize_dataset(
                dataset,
                samples,
                base_bits=base_bits,
                original_bits=original_bits,
                outlier_extra_bits=outlier_extra_bits,
            )
        )
    rows.sort(key=lambda row: row["dataset"])
    if len(rows) > 1:
        rows.append(summarize_rows(rows))
    return rows


def format_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Dataset | N | K comp. GB | V comp. GB | K ratio | V ratio | KV ratio | K outlier | V outlier |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        dataset = f"**{row['dataset']}**" if row["dataset"] == "overall" else row["dataset"]
        lines.append(
            "| "
            f"{dataset} | "
            f"{row['count']} | "
            f"{row['key_compressed_memory']:.6f} | "
            f"{row['value_compressed_memory']:.6f} | "
            f"{100 * row['key_compression_ratio']:.2f}% | "
            f"{100 * row['value_compression_ratio']:.2f}% | "
            f"{100 * row['kv_compression_ratio']:.2f}% | "
            f"{100 * row['keys_outlier_ratios']:.2f}% | "
            f"{100 * row['values_outlier_ratios']:.2f}% |"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Approximate K/V compressed memory and compression ratios from exported "
            "KV-cache status JSON. The split assumes K/V original memory is equal "
            "and allocates total compressed memory according to key/value outlier ratios."
        )
    )
    parser.add_argument("json_path", help="Path to a *_info.json result file.")
    parser.add_argument("--dataset", default=None, help="Optional dataset name to export, e.g. multi_news.")
    parser.add_argument("--output-json", default=None, help="Optional path to save the JSON summary.")
    parser.add_argument("--output-md", default=None, help="Optional path to save the Markdown table.")
    parser.add_argument("--base-bits", type=float, default=11.0, help="Bits per non-outlier BF16 value.")
    parser.add_argument("--original-bits", type=float, default=16.0, help="Original BF16 bits per value.")
    parser.add_argument("--outlier-extra-bits", type=float, default=8.0, help="Extra bits for an outlier exponent.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    json_path = Path(args.json_path)
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a dict at top level in {json_path}")

    rows = build_rows(
        data,
        dataset_name=args.dataset,
        base_bits=args.base_bits,
        original_bits=args.original_bits,
        outlier_extra_bits=args.outlier_extra_bits,
    )
    markdown = format_markdown(rows)
    print(markdown)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source": str(json_path),
            "approximation": {
                "base_bits": args.base_bits,
                "original_bits": args.original_bits,
                "outlier_extra_bits": args.outlier_extra_bits,
                "key_weight": "base_bits/original_bits + outlier_extra_bits/original_bits * keys_outlier_ratios",
                "value_weight": "base_bits/original_bits + outlier_extra_bits/original_bits * values_outlier_ratios",
            },
            "rows": rows,
        }
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"Saved JSON report to {output_path}")

    if args.output_md:
        output_path = Path(args.output_md)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown + "\n", encoding="utf-8")
        print(f"Saved Markdown table to {output_path}")


if __name__ == "__main__":
    main()
