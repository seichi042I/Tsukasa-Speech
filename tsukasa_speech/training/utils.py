"""Training utility functions shared between Stage 1 and Stage 2.

Provides checkpoint management and inference config helpers used by both
train_first.py (stage1) and finetune_accelerate.py (stage2).
"""

import os
import os.path as osp
import glob
import yaml


INFERENCE_CONFIG_KEYS = [
    'ASR_path', 'ASR_config', 'F0_path', 'PLBERT_dir',
    'preprocess_params', 'model_params', 'data_params',
]


def save_inference_config(config, log_dir):
    """Save a minimal inference config to log_dir/inference/config.yml."""
    inference_dir = osp.join(log_dir, 'inference')
    os.makedirs(inference_dir, exist_ok=True)
    inference_cfg = {k: config[k] for k in INFERENCE_CONFIG_KEYS if k in config}
    out_path = osp.join(inference_dir, 'config.yml')
    with open(out_path, 'w', encoding='utf-8') as f:
        yaml.dump(inference_cfg, f, default_flow_style=False, allow_unicode=True)
    return out_path


def find_latest_checkpoint(log_dir, prefix):
    """Find the latest checkpoint file matching {prefix}_NNNNN.pth pattern."""
    pattern = osp.join(log_dir, f'{prefix}_*.pth')
    files = glob.glob(pattern)
    if not files:
        return None
    def extract_number(f):
        basename = osp.basename(f)
        num_str = basename.replace(f'{prefix}_', '').replace('.pth', '')
        try:
            return int(num_str)
        except ValueError:
            return -1
    files = [(f, extract_number(f)) for f in files]
    files = [(f, n) for f, n in files if n >= 0]
    if not files:
        return None
    files.sort(key=lambda x: x[1])
    return files[-1][0]


def cleanup_checkpoints(log_dir, prefix, keep_latest=3, keep_every=None):
    """Remove old checkpoints, keeping the latest N and optionally every Kth.

    Args:
        log_dir: Directory containing checkpoints.
        prefix: Checkpoint filename prefix (e.g. 'checkpoint_1st', 'Sana_Finetune_').
        keep_latest: Number of most recent checkpoints to keep.
        keep_every: If set, also keep every Kth checkpoint (by index in sorted order).
    """
    pattern = osp.join(log_dir, f'{prefix}_*.pth')
    files = glob.glob(pattern)
    if not files:
        return

    def extract_number(f):
        basename = osp.basename(f)
        num_str = basename.replace(f'{prefix}_', '').replace('.pth', '')
        try:
            return int(num_str)
        except ValueError:
            return -1

    files = [(f, extract_number(f)) for f in files]
    files = [(f, n) for f, n in files if n >= 0]
    files.sort(key=lambda x: x[1])

    if len(files) <= keep_latest:
        return

    latest_files = set(f for f, _ in files[-keep_latest:])

    for idx, (filepath, number) in enumerate(files):
        if filepath in latest_files:
            continue
        if keep_every and idx % keep_every == 0:
            continue
        print(f'Removing old checkpoint: {osp.basename(filepath)}')
        os.remove(filepath)
