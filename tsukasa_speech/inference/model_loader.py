# coding: utf-8
"""Model discovery, checkpoint loading, and config normalization."""

import glob
import os
import os.path as osp
from collections import OrderedDict

import torch
import yaml

from tsukasa_speech.utils.common import recursive_munch
from tsukasa_speech.models.builder import build_model, load_ASR_models, load_F0_models
from tsukasa_speech.utils.plbert.util import load_plbert


def find_latest_checkpoint(directory, prefix='inference'):
    """Find the latest checkpoint matching {prefix}_NNNNN.pth in directory."""
    pattern = osp.join(directory, f'{prefix}_*.pth')
    files = glob.glob(pattern)
    if not files:
        return None
    def extract_number(f):
        num_str = osp.basename(f).replace(f'{prefix}_', '').replace('.pth', '')
        try:
            return int(num_str)
        except ValueError:
            return -1
    numbered = [(f, extract_number(f)) for f in files]
    numbered = [(f, n) for f, n in numbered if n >= 0]
    if not numbered:
        return None
    numbered.sort(key=lambda x: x[1])
    return numbered[-1][0]


def _find_any_checkpoint(directory):
    """Find any .pth file in directory (excluding optimizer/discriminator files)."""
    if not osp.isdir(directory):
        return None
    skip_prefixes = ('optimizer', 'discriminator', 'msd', 'mpd', 'wd')
    candidates = []
    for f in sorted(os.listdir(directory)):
        if not f.endswith('.pth'):
            continue
        base = f.lower().replace('.pth', '')
        if any(base.startswith(s) for s in skip_prefixes):
            continue
        candidates.append(osp.join(directory, f))
    return candidates[-1] if candidates else None


def _find_any_config(directory):
    """Find any .yml config file in directory."""
    if not osp.isdir(directory):
        return None
    for f in sorted(os.listdir(directory)):
        if f.endswith('.yml') or f.endswith('.yaml'):
            return osp.join(directory, f)
    return None


def normalize_config(config):
    """Fix auxiliary model paths to use correct relative paths."""
    defaults = {
        'F0_path': 'Utils/JDC/bst.t7',
        'ASR_config': 'Utils/ASR/config.yml',
        'ASR_path': 'Utils/ASR/bst_00080.pth',
        'PLBERT_dir': 'Utils/PLBERT/',
    }
    for key, default in defaults.items():
        val = config.get(key, '')
        if not val or not osp.exists(val):
            config[key] = default
    config.setdefault('preprocess_params', {}).setdefault('sr', 24000)
    return config


def resolve_model_dir(model_dir):
    """Auto-discover config, checkpoint, and style_db from a model directory.

    Supports multiple layouts:
      - Finetuned: {model_dir}/inference/config.yml + inference_*.pth
      - Legacy: {model_dir}/inference_config.yml + inference_*.pth
      - Base model: {model_dir}/*.yml + *.pth (e.g. Models/Style_Tsukasa_v02)

    Returns:
        (config_path, checkpoint_path, style_db_path or None)
    """
    inference_dir = osp.join(model_dir, 'inference')

    # Config
    config_path = osp.join(inference_dir, 'config.yml')
    if not osp.exists(config_path):
        for fallback in ['inference_config.yml', 'config_stage2.yml']:
            p = osp.join(model_dir, fallback)
            if osp.exists(p):
                config_path = p
                break
    if not osp.exists(config_path):
        # Base model: any .yml in the directory
        found = _find_any_config(model_dir)
        if found:
            config_path = found
    if not osp.exists(config_path):
        raise FileNotFoundError(
            f'No config found in {model_dir}. '
            f'Expected {inference_dir}/config.yml or any .yml in {model_dir}/')

    # Checkpoint
    checkpoint_path = find_latest_checkpoint(inference_dir, 'inference')
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint(model_dir, 'inference')
    if checkpoint_path is None:
        # Base model: any .pth in the directory
        checkpoint_path = _find_any_checkpoint(model_dir)
    if checkpoint_path is None:
        raise FileNotFoundError(
            f'No checkpoint found in {model_dir}. '
            f'Expected inference_*.pth or any .pth in {model_dir}/')

    # Style DB (optional)
    style_db_path = None
    for search_dir in [inference_dir, model_dir]:
        for name in ['style_db_compact.pt', 'style_db.pt']:
            p = osp.join(search_dir, name)
            if osp.exists(p):
                style_db_path = p
                break
        if style_db_path:
            break

    return config_path, checkpoint_path, style_db_path


def load_inference_model(config, checkpoint_path, device):
    """Load model from an inference checkpoint (lightweight, no optimizer/discriminator)."""
    model_params = recursive_munch(config['model_params'])

    # Load auxiliary models
    text_aligner = load_ASR_models(config['ASR_path'], config['ASR_config'])
    pitch_extractor = load_F0_models(config['F0_path'])
    plbert = load_plbert(config['PLBERT_dir'])

    model = build_model(model_params, text_aligner, pitch_extractor, plbert)

    # Load checkpoint
    state = torch.load(checkpoint_path, map_location='cpu')
    params = state['net']

    inference_keys = [
        'bert', 'bert_encoder', 'text_encoder', 'predictor',
        'predictor_encoder', 'style_encoder', 'decoder', 'diffusion',
    ]
    for key in inference_keys:
        if key in params:
            try:
                model[key].load_state_dict(params[key], strict=True)
            except RuntimeError:
                new_state_dict = OrderedDict()
                for (k_m, v_m), (k_c, v_c) in zip(
                    model[key].state_dict().items(), params[key].items()
                ):
                    new_state_dict[k_m] = v_c
                model[key].load_state_dict(new_state_dict, strict=True)
            print(f'Loaded {key}')

    for key in model:
        model[key].to(device)
        model[key].eval()

    return model, model_params
