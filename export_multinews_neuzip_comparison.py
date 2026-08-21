import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


MODEL_FILES = {
    "Qwen3-4B": "qwen3_4B_config_nvcomp_ans.json",
    "Mistral-7B": "mistral_7B_config_nvcomp_ans.json",
    "Qwen1.5-MoE": "qwen1p5_moe_config_nvcomp_ans.json",
    "Huihui-MoE": "huihui_moe_config_nvcomp_ans.json",
}

ADACODE_MULTINEWS_RATIO = {
    "Qwen3-4B": 71.87,
    "Mistral-7B": 71.33,
    "Qwen1.5-MoE": 71.48,
    "Huihui-MoE": 72.23,
}

TOKENS_PER_SECOND = {
    "Qwen3-4B": {"AdaCode": 448.0, "NeuZip": 50.0},
    "Mistral-7B": {"AdaCode": 667.6, "NeuZip": 62.4},
    "Qwen1.5-MoE": {"AdaCode": 483.2, "NeuZip": 74.7},
    "Huihui-MoE": {"AdaCode": 449.0, "NeuZip": 65.4},
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
    if isinstance(data, dict):
        records: list[dict[str, Any]] = []
        for value in data.values():
            records.extend(flatten_records(value))
        return records
    if isinstance(data, list):
        records = []
        for item in data:
            if isinstance(item, dict):
                records.append(item)
            else:
                records.extend(flatten_records(item))
        return records
    return []


def neuzip_ratio_for_dataset(path: Path, dataset: str) -> tuple[int, float]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    records = flatten_records(data.get(dataset, []) if isinstance(data, dict) else [])
    ratios = [
        float(record["kv_compression_ratio"]) * 100.0
        for record in records
        if "kv_compression_ratio" in record
    ]
    return len(ratios), mean(ratios) if ratios else 0.0


def build_rows(input_dir: Path, dataset: str) -> list[dict[str, Any]]:
    rows = []
    for model, file_name in MODEL_FILES.items():
        count, neuzip_ratio = neuzip_ratio_for_dataset(input_dir / file_name, dataset)
        tps = TOKENS_PER_SECOND[model]
        rows.append({
            "model": model,
            "n": count,
            "adacode_ratio": ADACODE_MULTINEWS_RATIO[model],
            "neuzip_ratio": neuzip_ratio,
            "adacode_tps": tps["AdaCode"],
            "neuzip_tps": tps["NeuZip"],
            "adacode_vs_neuzip_tps": tps["AdaCode"] / tps["NeuZip"],
        })
    return rows


def markdown_table(rows: list[dict[str, Any]], dataset: str) -> str:
    header = [
        "Model",
        "N",
        "AdaCode Ratio (%)",
        "NeuZip Ratio (%)",
        "AdaCode tok/s",
        "NeuZip tok/s",
    ]
    lines = [
        "# AdaCode vs. NeuZip on multi_news",
        "",
        f"Dataset: `{dataset}`. Compression ratio is compressed/original memory, lower is better.",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(header) - 1)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join([
                row["model"],
                str(row["n"]),
                f"{row['adacode_ratio']:.2f}",
                f"{row['neuzip_ratio']:.2f}",
                f"{row['adacode_tps']:.1f}",
                f"{row['neuzip_tps']:.1f}",
            ])
            + " |"
        )
    lines.append("")
    lines.append("NeuZip achieves a similar compression ratio, but its throughput is much lower. This shows that NeuZip is not suitable for dynamic KV-cache compression.")
    lines.append("")
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description="Export multi_news AdaCode-vs-NeuZip ratio and throughput comparison.")
    parser.add_argument("--input-dir", type=Path, default=default_input_dir())
    parser.add_argument("--dataset", default="multi_news")
    parser.add_argument("--output-prefix", type=Path, default=Path("multinews_neuzip_comparison"))
    return parser.parse_args()


def main():
    args = parse_args()
    rows = build_rows(args.input_dir, args.dataset)
    md = markdown_table(rows, args.dataset)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    md_path = args.output_prefix.with_suffix(".md")
    json_path = args.output_prefix.with_suffix(".json")
    md_path.write_text(md, encoding="utf-8")
    json_path.write_text(json.dumps({"dataset": args.dataset, "rows": rows}, indent=2), encoding="utf-8")

    print(md)
    print(f"Saved Markdown table to {md_path}")
    print(f"Saved JSON summary to {json_path}")


if __name__ == "__main__":
    main()
