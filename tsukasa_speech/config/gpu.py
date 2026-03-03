#!/usr/bin/env python3
"""
Detect GPU VRAM and recommend a training config tier.

Returns one of: low, mid, high  (printed to stdout)

Thresholds (based on largest single GPU):
  - low  : < 24 GB   (~16GB class: RTX 4060Ti 16GB, RTX 3080, etc.)
  - mid  : 24-48 GB  (RTX 3090, RTX 4090, A5000, V100 32GB, etc.)
  - high : >= 48 GB  (A100, A40, H100, multi-GPU aggregated, etc.)

Usage:
  python detect_gpu.py              # prints tier name
  python detect_gpu.py --json       # prints full JSON report
  python detect_gpu.py --config     # prints recommended config path

Can also be used as a library:
  from tsukasa_speech.config.gpu import detect_gpu_tier
  tier, info = detect_gpu_tier()
"""
import argparse
import json
import os
import sys

try:
    import yaml
except ImportError:
    yaml = None

# VRAM thresholds in MB
TIER_THRESHOLDS = {
    'low': 24 * 1024,     # < 24 GB
    'mid': 48 * 1024,     # 24-48 GB
    # 'high': >= 36 GB
}

TIER_CONFIGS = {
    'low': 'Configs/config_low_vram.yml',
    'mid': 'Configs/config_mid_vram.yml',
    'high': 'Configs/config_high_vram.yml',
}

def _project_root():
    """Project root directory (two levels up from tsukasa_speech/config/)."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_tier_config(tier):
    """Load YAML config for the given tier. Returns None if unavailable."""
    if yaml is None:
        return None
    path = os.path.join(_project_root(), TIER_CONFIGS[tier])
    if not os.path.isfile(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def get_tier_description(tier):
    """Build tier description from config (batch_size, max_len, SLM)."""
    vranges = {'low': '~16GB', 'mid': '24-32GB', 'high': '32GB+'}
    cfg = _load_tier_config(tier)
    if cfg is None:
        return f"{vranges[tier]} (batch_size=?, max_len=?, SLM unknown)"

    batch_size = cfg.get('batch_size', '?')
    max_len = cfg.get('max_len', '?')
    loss = cfg.get('loss_params') or {}
    lambda_slm = loss.get('lambda_slm', 0)
    slm_status = 'SLM enabled' if lambda_slm and float(lambda_slm) != 0 else 'SLM disabled'

    return f"{vranges[tier]} (batch_size={batch_size}, max_len={max_len}, {slm_status})"


def detect_gpu_tier():
    """Detect GPU VRAM and return (tier_name, info_dict).

    Uses the largest single GPU's VRAM to determine the tier.
    Falls back to 'low' if no GPU is found.
    """
    info = {
        'cuda_available': False,
        'gpu_count': 0,
        'gpus': [],
        'max_vram_mb': 0,
        'total_vram_mb': 0,
        'tier': 'low',
        'config': TIER_CONFIGS['low'],
    }

    try:
        import torch
        if not torch.cuda.is_available():
            info['error'] = 'CUDA not available'
            return 'low', info

        info['cuda_available'] = True
        n_gpus = torch.cuda.device_count()
        info['gpu_count'] = n_gpus

        max_vram = 0
        total_vram = 0
        for i in range(n_gpus):
            props = torch.cuda.get_device_properties(i)
            vram_mb = props.total_memory / (1024 ** 2)
            gpu_info = {
                'index': i,
                'name': props.name,
                'vram_mb': int(vram_mb),
            }
            info['gpus'].append(gpu_info)
            max_vram = max(max_vram, vram_mb)
            total_vram += vram_mb

        info['max_vram_mb'] = int(max_vram)
        info['total_vram_mb'] = int(total_vram)

        # Determine tier based on max single GPU VRAM
        if max_vram >= TIER_THRESHOLDS['mid']:
            tier = 'high'
        elif max_vram >= TIER_THRESHOLDS['low']:
            tier = 'mid'
        else:
            tier = 'low'

        info['tier'] = tier
        info['config'] = TIER_CONFIGS[tier]

        return tier, info

    except ImportError:
        info['error'] = 'PyTorch not installed'
        return 'low', info
    except Exception as e:
        info['error'] = str(e)
        return 'low', info


def main():
    parser = argparse.ArgumentParser(description='Detect GPU and recommend config tier')
    parser.add_argument('--json', action='store_true', help='Output full JSON report')
    parser.add_argument('--config', action='store_true', help='Output recommended config path')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print detailed info to stderr')
    args = parser.parse_args()

    tier, info = detect_gpu_tier()

    if args.verbose or (not args.json and not args.config):
        # Print human-readable summary to stderr so stdout stays clean for piping
        print(f'GPU Detection:', file=sys.stderr)
        if info['gpus']:
            for gpu in info['gpus']:
                print(f'  GPU {gpu["index"]}: {gpu["name"]} ({gpu["vram_mb"]} MB)', file=sys.stderr)
        else:
            print(f'  No GPU detected: {info.get("error", "unknown")}', file=sys.stderr)
        print(f'  Max VRAM:  {info["max_vram_mb"]} MB', file=sys.stderr)
        print(f'  Tier:      {tier} - {get_tier_description(tier)}', file=sys.stderr)
        print(f'  Config:    {info["config"]}', file=sys.stderr)

    if args.json:
        print(json.dumps(info, indent=2))
    elif args.config:
        print(info['config'])
    else:
        print(tier)


if __name__ == '__main__':
    main()
