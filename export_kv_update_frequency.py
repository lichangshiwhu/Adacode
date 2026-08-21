import argparse
import json
from pathlib import Path
from typing import Any


BASE_PATHS = [
    "./configs/kv_test_Qwen3-4B_False_True_adjuster_True_steps",
    "./configs/kv_test_Mistral-7B-Instruct-v0.3_False_True_adjuster_True_steps",
    "./configs/kv_test_Qwen1.5-MoE-A2.7B-Chat_False_True_adjuster_True_steps",
    "./configs/kv_test_Huihui-MoE-1.2B-A0.6B_False_True_adjuster_True_steps",
]

STEPS = [32, 64, 128, 256, 512, 1024]

DEFAULT_DATASETS = [
    "riddlebench",
    "simplemath",
    "livecodebench",
    "mbpp",
    "aime24",
    "aime25",
]


def default_config_dir() -> Path:
    script_path = Path(__file__).resolve()
    candidates = [
        Path.cwd() / "configs",
        script_path.parent / "configs",
        script_path.parents[2] / "configs",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path.cwd() / "configs"


def model_name_from_base_path(base_path: str) -> str:
    name = Path(base_path).name
    prefix = "kv_test_"
    suffix = "_False_True_adjuster_True_steps"
    if name.startswith(prefix):
        name = name[len(prefix):]
    if name.endswith(suffix):
        name = name[:-len(suffix)]
    return name


def resolve_info_path(config_dir: Path, base_path: str, step: int) -> Path:
    base_name = Path(base_path).name
    return config_dir / f"{base_name}{step}_info.json"


def collect_records(
    data: Any,
    datasets: list[str] | None = None,
    dataset_limits: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    records: list[dict[str, Any]] = []
    dataset_counts: dict[str, int] = {}

    if isinstance(data, dict) and datasets is not None:
        selected_data = {}
        for dataset in datasets:
            value = data.get(dataset, [])
            if isinstance(value, list) and dataset_limits is not None and dataset in dataset_limits:
                value = value[:dataset_limits[dataset]]
            selected_data[dataset] = value
    else:
        selected_data = data

    if isinstance(selected_data, dict):
        for dataset, value in selected_data.items():
            if isinstance(value, list):
                dataset_counts[dataset] = len(value)

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            if "keys_update_times" in node or "values_update_times" in node:
                records.append(node)
            else:
                for value in node.values():
                    visit(value)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(selected_data)
    return records, dataset_counts


def summarize_file(
    path: Path,
    datasets: list[str] | None = None,
    dataset_limits: dict[str, int] | None = None,
) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    records, dataset_counts = collect_records(data, datasets=datasets, dataset_limits=dataset_limits)

    total_keys = sum(int(record.get("keys_update_times", 0)) for record in records)
    total_values = sum(int(record.get("values_update_times", 0)) for record in records)
    total_kv = total_keys + total_values
    total_key_proposals = sum(int(record.get("keys_update_proposals", 0)) for record in records)
    total_value_proposals = sum(int(record.get("values_update_proposals", 0)) for record in records)
    total_kv_proposals = total_key_proposals + total_value_proposals
    total_new_tokens = sum(float(record.get("total_new_tokens", 0.0)) for record in records)
    total_adjust_recompresses = sum(int(record.get("adjust_recompresses", 0)) for record in records)
    num_records = len(records)
    effective_steps = sorted({
        int(record["kv_adjust_update_steps"])
        for record in records
        if "kv_adjust_update_steps" in record
    })

    return {
        "path": str(path),
        "dataset_counts": dataset_counts,
        "effective_update_steps": effective_steps,
        "num_records": num_records,
        "keys_update_times": total_keys,
        "values_update_times": total_values,
        "kv_update_times": total_kv,
        "keys_update_proposals": total_key_proposals,
        "values_update_proposals": total_value_proposals,
        "kv_update_proposals": total_kv_proposals,
        "avg_kv_update_times_per_record": 0.0 if num_records == 0 else total_kv / num_records,
        "avg_kv_update_proposals_per_record": 0.0 if num_records == 0 else total_kv_proposals / num_records,
        "kv_update_times_per_1k_new_tokens": 0.0 if total_new_tokens == 0 else total_kv * 1000.0 / total_new_tokens,
        "kv_update_proposals_per_1k_new_tokens": 0.0 if total_new_tokens == 0 else total_kv_proposals * 1000.0 / total_new_tokens,
        "total_new_tokens": total_new_tokens,
        "adjust_recompresses": total_adjust_recompresses,
        "avg_adjust_recompresses_per_record": 0.0 if num_records == 0 else total_adjust_recompresses / num_records,
    }


def infer_prefix_limits(config_dir: Path, datasets: list[str] | None) -> dict[str, dict[str, int]]:
    if datasets is None:
        return {}

    limits_by_model: dict[str, dict[str, int]] = {}
    for base_path in BASE_PATHS:
        model = model_name_from_base_path(base_path)
        limits: dict[str, int] = {}
        for dataset in datasets:
            counts = []
            for step in STEPS:
                path = resolve_info_path(config_dir, base_path, step)
                if not path.exists():
                    continue
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                value = data.get(dataset, []) if isinstance(data, dict) else []
                if isinstance(value, list):
                    counts.append(len(value))
            if counts:
                limits[dataset] = min(counts)
        limits_by_model[model] = limits
    return limits_by_model


def build_summary(config_dir: Path, datasets: list[str] | None, align_prefix: bool) -> dict[str, Any]:
    limits_by_model = infer_prefix_limits(config_dir, datasets) if align_prefix else {}
    rows = []
    for base_path in BASE_PATHS:
        model = model_name_from_base_path(base_path)
        dataset_limits = limits_by_model.get(model)
        for step in STEPS:
            path = resolve_info_path(config_dir, base_path, step)
            if path.exists():
                summary = summarize_file(path, datasets=datasets, dataset_limits=dataset_limits)
                summary["model"] = model
                summary["step"] = step
                summary["missing"] = False
            else:
                summary = {
                    "model": model,
                    "step": step,
                    "path": str(path),
                    "missing": True,
                    "dataset_counts": {},
                    "effective_update_steps": [],
                    "num_records": 0,
                    "keys_update_times": 0,
                    "values_update_times": 0,
                    "kv_update_times": 0,
                    "keys_update_proposals": 0,
                    "values_update_proposals": 0,
                    "kv_update_proposals": 0,
                    "avg_kv_update_times_per_record": 0.0,
                    "avg_kv_update_proposals_per_record": 0.0,
                    "kv_update_times_per_1k_new_tokens": 0.0,
                    "kv_update_proposals_per_1k_new_tokens": 0.0,
                    "total_new_tokens": 0.0,
                    "adjust_recompresses": 0,
                    "avg_adjust_recompresses_per_record": 0.0,
                }
            rows.append(summary)
    return {
        "config_dir": str(config_dir),
        "datasets": datasets if datasets is not None else "all",
        "align_prefix": align_prefix,
        "prefix_limits": limits_by_model,
        "steps": STEPS,
        "rows": rows,
    }


def pivot_table(rows: list[dict[str, Any]], value_key: str) -> list[list[str]]:
    models = [model_name_from_base_path(base_path) for base_path in BASE_PATHS]
    table = [["Model"] + [str(step) for step in STEPS]]
    by_key = {(row["model"], row["step"]): row for row in rows}
    for model in models:
        table_row = [model]
        for step in STEPS:
            row = by_key[(model, step)]
            if row.get("missing"):
                table_row.append("-")
                continue
            value = row[value_key]
            if isinstance(value, float):
                table_row.append(f"{value:.3f}")
            elif isinstance(value, list):
                table_row.append(",".join(str(item) for item in value))
            else:
                table_row.append(str(value))
        table.append(table_row)
    return table


def markdown_table(table: list[list[str]]) -> str:
    header = table[0]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(header) - 1)) + " |",
    ]
    for row in table[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_outputs(summary: dict[str, Any], output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    tables = {
        "effective_update_steps": pivot_table(summary["rows"], "effective_update_steps"),
        "num_records": pivot_table(summary["rows"], "num_records"),
        "total_kv_update_times": pivot_table(summary["rows"], "kv_update_times"),
        "total_kv_update_proposals": pivot_table(summary["rows"], "kv_update_proposals"),
        "adjust_recompresses": pivot_table(summary["rows"], "adjust_recompresses"),
        "avg_kv_update_times_per_record": pivot_table(summary["rows"], "avg_kv_update_times_per_record"),
        "avg_kv_update_proposals_per_record": pivot_table(summary["rows"], "avg_kv_update_proposals_per_record"),
        "avg_adjust_recompresses_per_record": pivot_table(summary["rows"], "avg_adjust_recompresses_per_record"),
        "kv_update_times_per_1k_new_tokens": pivot_table(summary["rows"], "kv_update_times_per_1k_new_tokens"),
        "kv_update_proposals_per_1k_new_tokens": pivot_table(summary["rows"], "kv_update_proposals_per_1k_new_tokens"),
    }

    md_parts = [
        "# KV Update Frequency",
        "",
        "KV update count is `keys_update_times + values_update_times`.",
        f"Datasets included: `{summary['datasets']}`.",
        f"Prefix-aligned records: `{summary['align_prefix']}`.",
        "",
        "## Effective Update Steps Recorded in Logs",
        "",
        markdown_table(tables["effective_update_steps"]),
        "",
        "## Number of Records",
        "",
        markdown_table(tables["num_records"]),
        "",
        "## Total KV Update Times",
        "",
        markdown_table(tables["total_kv_update_times"]),
        "",
        "## Total KV Update Proposals",
        "",
        markdown_table(tables["total_kv_update_proposals"]),
        "",
        "## Total Adjust Recompresses",
        "",
        markdown_table(tables["adjust_recompresses"]),
        "",
        "## Average KV Update Times per Record",
        "",
        markdown_table(tables["avg_kv_update_times_per_record"]),
        "",
        "## Average KV Update Proposals per Record",
        "",
        markdown_table(tables["avg_kv_update_proposals_per_record"]),
        "",
        "## Average Adjust Recompresses per Record",
        "",
        markdown_table(tables["avg_adjust_recompresses_per_record"]),
        "",
        "## KV Update Times per 1K New Tokens",
        "",
        markdown_table(tables["kv_update_times_per_1k_new_tokens"]),
        "",
        "## KV Update Proposals per 1K New Tokens",
        "",
        markdown_table(tables["kv_update_proposals_per_1k_new_tokens"]),
        "",
    ]

    md_path = output_prefix.with_suffix(".md")
    json_path = output_prefix.with_suffix(".json")
    md_path.write_text("\n".join(md_parts), encoding="utf-8")
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n".join(md_parts))
    print(f"Saved Markdown table to {md_path}")
    print(f"Saved JSON summary to {json_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Export KV update-frequency tables from *_info.json logs.")
    parser.add_argument("--config-dir", type=Path, default=default_config_dir(), help="Directory containing *_info.json logs.")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=DEFAULT_DATASETS,
        help="Datasets to include. Defaults to the short-task datasets common to all step logs. Use --datasets all to include every dataset in each file.",
    )
    parser.add_argument(
        "--no-align-prefix",
        action="store_true",
        help="Do not truncate each dataset to the common prefix length across steps.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("kv_update_frequency"),
        help="Output path prefix. .md and .json files will be written.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    datasets = None if args.datasets == ["all"] else args.datasets
    summary = build_summary(args.config_dir, datasets=datasets, align_prefix=not args.no_align_prefix)
    write_outputs(summary, args.output_prefix)


if __name__ == "__main__":
    main()
