import argparse
import json
from pathlib import Path
from typing import Any


NAME_MAP = {
    'keys': 'latent',
    'values': 'rope',
}


def rename_mla_cache_names(obj: Any) -> Any:
    if isinstance(obj, dict):
        renamed = {}
        for key, value in obj.items():
            new_key = key
            for old_name, new_name in NAME_MAP.items():
                new_key = new_key.replace(old_name, new_name)
            renamed[new_key] = rename_mla_cache_names(value)
        return renamed
    if isinstance(obj, list):
        return [rename_mla_cache_names(item) for item in obj]
    return obj


def _default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f'{input_path.stem}_mla_names{input_path.suffix}')


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Rename old DeepSeek MLA cache statistic fields from keys/values to latent/rope.'
    )
    parser.add_argument('input', type=Path, help='Input JSON file exported by export_kvcache_status.py.')
    parser.add_argument(
        '-o',
        '--output',
        type=Path,
        default=None,
        help='Output JSON file. Defaults to <input_stem>_mla_names.json.',
    )
    parser.add_argument(
        '--in-place',
        action='store_true',
        help='Overwrite the input JSON file instead of writing a new file.',
    )
    parser.add_argument(
        '--indent',
        type=int,
        default=None,
        help='Pretty-print JSON with this indentation. Defaults to compact JSON.',
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    input_path = args.input
    if not input_path.exists():
        raise FileNotFoundError(f'Input JSON does not exist: {input_path}')
    if args.in_place and args.output is not None:
        raise ValueError('Use either --in-place or --output, not both.')

    output_path = input_path if args.in_place else (args.output or _default_output_path(input_path))

    with input_path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    renamed = rename_mla_cache_names(data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(renamed, f, indent=args.indent)

    print(f'Wrote renamed MLA cache stats to {output_path}')


if __name__ == '__main__':
    main()
