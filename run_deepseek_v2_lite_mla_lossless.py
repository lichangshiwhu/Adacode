import argparse
import os
import subprocess
import sys
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Run main_lossless_comp.py for DeepSeek-V2-Lite with MLA Float11 cache.',
        add_help=True,
    )
    parser.add_argument('--model_config_name', default=_env('MODEL_CONFIG_NAME', 'deepseek_v2_lite_config'))
    parser.add_argument('--kv_cache_impl', default=_env('KV_CACHE_IMPL', 'mla_float11'))
    parser.add_argument('--mla_cache_mode', default=_env('MLA_CACHE_MODE', 'auto'))
    parser.add_argument('--kv_layout', default=_env('KV_LAYOUT', 'segmented_tiled'))
    parser.add_argument('--block_size', default=_env('BLOCK_SIZE', '128'))
    parser.add_argument('--batch_size', default=_env('BATCH_SIZE', '1'))
    parser.add_argument('--num_seqence', default=_env('NUM_SEQUENCE', '4'))
    parser.add_argument('--max_length', default=_env('MAX_LENGTH', '4096'))
    parser.add_argument('--max_new_tokens', default=_env('MAX_NEW_TOKENS', '128'))
    parser.add_argument('--output_path', default=None)
    parser.add_argument('--compress_rope', action='store_true', default=_env('MLA_COMPRESS_ROPE', '0') == '1')
    parser.add_argument('--use_synthetic_data', action='store_true', default=_env('USE_SYNTHETIC_DATA', '0') == '1')
    parser.add_argument('--synthetic_seq_length', default=_env('SYNTHETIC_SEQ_LENGTH', '2048'))
    parser.add_argument('--float11_need_adjust', action='store_true', default=_env('FLOAT11_NEED_ADJUST', '0') == '1')
    parser.add_argument('--float11_update_steps', default=_env('FLOAT11_UPDATE_STEPS', '64'))
    return parser


def main() -> int:
    root_dir = Path(__file__).resolve().parents[1]
    kv_dir = root_dir / 'kv_compression'
    main_script = kv_dir / 'main_lossless_comp.py'

    parser = build_arg_parser()
    args, extra_args = parser.parse_known_args()

    output_path = args.output_path
    if output_path is None:
        output_path = str(root_dir / 'kv_lossless_mla_float11_DeepSeek-V2-Lite.json')

    cmd = [
        sys.executable,
        str(main_script),
        '--model_config_name', args.model_config_name,
        '--kv_cache_impl', args.kv_cache_impl,
        '--mla_cache_mode', args.mla_cache_mode,
        '--kv_layout', args.kv_layout,
        '--block_size', str(args.block_size),
        '--batch_size', str(args.batch_size),
        '--num_seqence', str(args.num_seqence),
        '--max_length', str(args.max_length),
        '--max_new_tokens', str(args.max_new_tokens),
        '--output_path', output_path,
    ]

    if not args.compress_rope:
        cmd.append('--mla_disable_rope_compression')
    if args.use_synthetic_data:
        cmd.extend(['--use_synthetic_data', '--synthetic_seq_length', str(args.synthetic_seq_length)])
    if args.float11_need_adjust:
        cmd.extend(['--float11_need_adjust', '--float11_update_steps', str(args.float11_update_steps)])
    cmd.extend(extra_args)

    env = os.environ.copy()
    pythonpath_parts = [str(root_dir), str(kv_dir)]
    if env.get('PYTHONPATH'):
        pythonpath_parts.append(env['PYTHONPATH'])
    env['PYTHONPATH'] = os.pathsep.join(pythonpath_parts)

    print('Running:', ' '.join(cmd), flush=True)
    return subprocess.run(cmd, env=env).returncode


if __name__ == '__main__':
    raise SystemExit(main())
