import argparse
import json
from pathlib import Path
from statistics import mean, pstdev, pvariance
from typing import Any


MODEL_FILES = {
    "Qwen3-4B": "qwen3_4B_config_nvcomp_ans.json",
    "Mistral-7B": "mistral_7B_config_nvcomp_ans.json",
    "Qwen1.5-MoE": "qwen1p5_moe_config_nvcomp_ans.json",
    "Huihui-MoE": "huihui_moe_config_nvcomp_ans.json",
}


def default_input_dir() -> Path:
    script_path = Path(__file__).resolve()
    candidates = [
        Path.cwd() / "configs" / "kv_longbench_lossless_comp",
        script_path.parent / "configs" / "kv_longbench_lossless_comp",
        script_path.parents[2] / "configs" / "kv_longbench_lossless_comp",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path.cwd()


def flatten_records(data: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if isinstance(data, dict):
        for value in data.values():
            records.extend(flatten_records(value))
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                records.append(item)
            else:
                records.extend(flatten_records(item))
    return records


def select_dataset_records(data: Any, dataset_name: str | None) -> list[dict[str, Any]]:
    if dataset_name is None:
        return flatten_records(data)
    if not isinstance(data, dict):
        return []
    return flatten_records(data.get(dataset_name, []))


def summarize_values(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "variance": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    return {
        "count": len(values),
        "mean": mean(values),
        "variance": pvariance(values),
        "std": pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def summarize_file(path: Path, dataset_name: str | None = None) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    records = select_dataset_records(data, dataset_name)
    compressed = [float(record["kv_compressed_memory"]) for record in records if "kv_compressed_memory" in record]
    original = [float(record["kv_original_memory"]) for record in records if "kv_original_memory" in record]
    ratios = [
        float(record["kv_compression_ratio"])
        for record in records
        if "kv_compression_ratio" in record
    ]

    by_dataset: dict[str, list[float]] = {}
    for record in records:
        if "kv_compressed_memory" not in record:
            continue
        dataset = str(record.get("dataset", "unknown"))
        by_dataset.setdefault(dataset, []).append(float(record["kv_compressed_memory"]))

    return {
        "path": str(path),
        "kv_compressed_memory": summarize_values(compressed),
        "kv_original_memory": summarize_values(original),
        "kv_compression_ratio": summarize_values(ratios),
        "datasets": {
            dataset: summarize_values(values)
            for dataset, values in sorted(by_dataset.items())
        },
    }


def build_summary(input_dir: Path, dataset_name: str | None = None) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for model, file_name in MODEL_FILES.items():
        path = input_dir / file_name
        if not path.exists():
            models[model] = {"missing": True, "path": str(path)}
            continue
        result = summarize_file(path, dataset_name=dataset_name)
        result["missing"] = False
        models[model] = result
    return {
        "input_dir": str(input_dir),
        "dataset": dataset_name or "all",
        "method": "NeuZip/nvcomp_ans",
        "models": models,
    }


def markdown_table(summary: dict[str, Any]) -> str:
    rows = [
        [
            "Model",
            "N",
            "Mean Comp. Mem. (GB)",
            "Variance",
            "Std.",
            "Mean Orig. Mem. (GB)",
            "Mean Ratio (%)",
        ]
    ]
    for model in MODEL_FILES:
        item = summary["models"][model]
        if item.get("missing"):
            rows.append([model, "-", "-", "-", "-", "-", "-"])
            continue
        comp = item["kv_compressed_memory"]
        orig = item["kv_original_memory"]
        ratio = item["kv_compression_ratio"]
        rows.append([
            model,
            str(comp["count"]),
            f"{comp['mean']:.4f}",
            f"{comp['variance']:.6f}",
            f"{comp['std']:.4f}",
            f"{orig['mean']:.4f}",
            f"{ratio['mean'] * 100:.2f}",
        ])

    lines = [
        "# NeuZip KV Compressed Memory",
        "",
        "Memory is reported in GB from `kv_compressed_memory` in `*_config_nvcomp_ans.json`.",
        f"Dataset: `{summary['dataset']}`.",
        "",
        "| " + " | ".join(rows[0]) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(rows[0]) - 1)) + " |",
    ]
    for row in rows[1:]:
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description="Export NeuZip/nvcomp KV compressed-memory summary.")
    parser.add_argument("--input-dir", type=Path, default=default_input_dir())
    parser.add_argument("--dataset", default=None, help="Optional dataset key to summarize, e.g. multi_news.")
    parser.add_argument("--output-prefix", type=Path, default=Path("nvcomp_kv_memory_table"))
    return parser.parse_args()


def main():
    args = parse_args()
    summary = build_summary(args.input_dir, dataset_name=args.dataset)
    md = markdown_table(summary)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    md_path = args.output_prefix.with_suffix(".md")
    json_path = args.output_prefix.with_suffix(".json")
    md_path.write_text(md, encoding="utf-8")
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(md)
    print(f"Saved Markdown table to {md_path}")
    print(f"Saved JSON summary to {json_path}")


if __name__ == "__main__":
    main()
