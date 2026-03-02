#!/usr/bin/env python3
"""
Merge base config with user overrides from Data/run_config.yaml.

Base configs (baked into Docker image) contain all invariant parameters.
User's run_config.yaml provides only the values they want to override.

Usage:
  python merge_config.py --base Configs/config_mid_vram.yml \
    --run-config Data/run_config.yaml \
    --output /tmp/config_stage1.yml \
    --stage 1 \
    --num-speakers 2

  (Reference/example base configs are in Configs/reference/)
"""
import argparse
import copy
import os
import sys

import yaml


def deep_merge(base, override):
    """Recursively merge override dict into base dict. Override wins on conflicts."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


# Mapping from run_config.yaml stage keys to base config keys
STAGE_KEY_MAP = {
    1: {
        'max_steps': 'max_steps_1st',
    },
    2: {},
}


def build_stage_overrides(run_config, stage):
    """Extract stage-specific overrides from run_config and map to base config keys."""
    stage_key = f'stage{stage}'
    stage_section = run_config.get(stage_key, {})
    if not stage_section:
        return {}

    key_map = STAGE_KEY_MAP.get(stage, {})
    overrides = {}
    for k, v in stage_section.items():
        mapped_key = key_map.get(k, k)
        overrides[mapped_key] = v
    return overrides


def main():
    parser = argparse.ArgumentParser(description="Merge base config with user overrides")
    parser.add_argument("--base", required=True, help="Path to base config YAML")
    parser.add_argument("--run-config", required=True, help="Path to Data/run_config.yaml")
    parser.add_argument("--output", required=True, help="Output path for merged config")
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2], help="Training stage")
    parser.add_argument("--num-speakers", type=int, default=1, help="Number of detected speakers")
    parser.add_argument("--data-dir", default=None,
                        help="Data directory (e.g. Data/my_model). "
                             "When set, overrides train_data, val_data, and log_dir paths.")
    args = parser.parse_args()

    # Load base config
    with open(args.base, 'r', encoding='utf-8') as f:
        base = yaml.safe_load(f)

    # Load user run_config (optional file)
    run_config = {}
    if os.path.exists(args.run_config):
        with open(args.run_config, 'r', encoding='utf-8') as f:
            run_config = yaml.safe_load(f) or {}

    # Build overrides from stage-specific section
    stage_overrides = build_stage_overrides(run_config, args.stage)

    # Build top-level overrides (exclude meta keys and stage sections)
    skip_keys = {'stage', 'stage1', 'stage2', 'val_ratio', 'gpu_tier', 'max_duration'}
    top_overrides = {k: v for k, v in run_config.items() if k not in skip_keys}

    # Merge: base <- top_overrides <- stage_overrides
    merged = deep_merge(base, top_overrides)
    merged = deep_merge(merged, stage_overrides)

    # Override data paths if --data-dir is specified
    if args.data_dir:
        merged.setdefault('data_params', {})['train_data'] = os.path.join(args.data_dir, 'train_list.txt')
        merged.setdefault('data_params', {})['val_data'] = os.path.join(args.data_dir, 'val_list.txt')
        merged['log_dir'] = os.path.join(args.data_dir, 'output')

    # Auto-set multispeaker based on detected speakers
    if args.num_speakers > 1:
        merged.setdefault('model_params', {})['multispeaker'] = True

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    with open(args.output, 'w', encoding='utf-8') as f:
        yaml.dump(merged, f, default_flow_style=False, allow_unicode=True)

    print(f"Config merged: {args.base} + {args.run_config} -> {args.output} (stage {args.stage}, {args.num_speakers} speaker(s))")


if __name__ == "__main__":
    main()
