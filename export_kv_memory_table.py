import argparse
import json
import os
from typing import Any


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _collect_dataset_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for dataset_name, samples in data.items():
        if not isinstance(samples, list):
            continue

        compressed_values = []
        original_values = []
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            compressed = sample.get("kv_compressed_memory")
            original = sample.get("kv_original_memory")
            if compressed is None or original is None:
                continue
            compressed_values.append(float(compressed))
            original_values.append(float(original))

        mean_compressed = _safe_mean(compressed_values)
        mean_original = _safe_mean(original_values)
        ratio = (mean_compressed / mean_original) if mean_original > 0 else 0.0
        rows.append(
            {
                "dataset": dataset_name,
                "kv_compressed_memory": mean_compressed,
                "kv_original_memory": mean_original,
                "compressed_ratio": ratio,
            }
        )
    return rows


def _print_table(rows: list[dict[str, Any]]) -> None:
    dataset_names = [row["dataset"] for row in rows]
    table = [
        ["datasets"] + dataset_names,
        ["kv_compressed_memory"] + [f"{row['kv_compressed_memory']:.9f}" for row in rows],
        ["kv_original_memory"] + [f"{row['kv_original_memory']:.9f}" for row in rows],
        ["compressed ratio"] + [f"{row['compressed_ratio']:.9f}" for row in rows],
    ]

    col_widths = [max(len(r[idx]) for r in table) for idx in range(len(table[0]))]
    for row in table:
        print(" | ".join(cell.ljust(col_widths[idx]) for idx, cell in enumerate(row)))


def parse_args():
    parser = argparse.ArgumentParser(description="Export KV memory summary table from a result JSON file.")
    parser.add_argument(
        "json_path",
        help="Path to the JSON file, e.g. kv_short_BLOCK_SIZE8_inputs_DeepSeek-V2-Lite.json",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to save the aggregated dataset rows as JSON.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = _collect_dataset_rows(data)
    _print_table(rows)

    if args.output_json:
        output_dir = os.path.dirname(args.output_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        print(f"Saved JSON report to {args.output_json}")


if __name__ == "__main__":
    main()
