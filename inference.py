"""Simple TTS inference script for Tsukasa Speech.

Usage (simple — just specify model directory):
    python inference.py \
        --model-dir Data/output/MidVRAM \
        --text "こんにちは、元気ですか"

Usage (with options):
    python inference.py \
        --model-dir Data/output/MidVRAM \
        --text "こんにちは、元気ですか" \
        --speaker 1 --diffusion-steps 10 --output output.wav

Usage (with reference audio):
    python inference.py \
        --model-dir Data/output/MidVRAM \
        --reference Data/speaker1/wav/sample.wav \
        --text "こんにちは、元気ですか"

Usage (with style DB text-similarity search):
    python inference.py \
        --model-dir Data/output/MidVRAM \
        --use-style-db-search \
        --text "こんにちは、元気ですか"

Legacy usage (explicit paths):
    python inference.py \
        --config Configs/config.yml \
        --checkpoint path/to/inference_5000.pth \
        --text "こんにちは、元気ですか"
"""

import argparse
import glob
import os
import os.path as osp
import re

import yaml
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import librosa
import soundfile as sf

from munch import Munch
from meldataset import TextCleaner, preprocess
from Utils.phonemize.mixed_phon import smart_phonemize
from Utils.PLBERT.util import load_plbert
from models import build_model, load_ASR_models, load_F0_models
from utils import length_to_mask, log_norm, recursive_munch
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule


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
                from collections import OrderedDict
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


def compute_ref_style(model, ref_audio_path, sr=24000, device='cuda'):
    """Extract style embedding from a reference audio file."""
    wave, orig_sr = librosa.load(ref_audio_path, sr=sr)
    mel = preprocess(wave).to(device)  # [1, 80, T]

    with torch.no_grad():
        ref_ss = model.style_encoder(mel.unsqueeze(1))       # acoustic style
        ref_sp = model.predictor_encoder(mel.unsqueeze(1))    # prosodic style

    return ref_ss, ref_sp


@torch.no_grad()
def lookup_style_from_db(model, text, style_db_path, speaker_id, device='cuda'):
    """Look up style vectors from a precomputed style DB using text similarity.

    Computes a BERT embedding for the input text, finds the most
    text-similar entries for the target speaker, and randomly picks one
    from the top 10.
    """
    text_cleaner = TextCleaner()

    # Phonemize and tokenize
    phonemized = smart_phonemize(text)
    tokens = text_cleaner(phonemized)
    tokens = [0] + tokens + [0]
    tokens = torch.LongTensor(tokens).unsqueeze(0).to(device)
    input_lengths = torch.LongTensor([tokens.shape[1]]).to(device)
    text_mask = length_to_mask(input_lengths).to(device)

    # BERT embedding for input text
    bert_out = model.bert(tokens, attention_mask=(~text_mask).int())  # (1, T, 768)
    query_embed = bert_out.mean(dim=1)  # (1, 768)

    # Load style DB and filter to target speaker
    db = torch.load(style_db_path, map_location=device)
    speaker_mask = db['speaker_ids'] == speaker_id
    if speaker_mask.sum() == 0:
        available = db['speaker_ids'].unique().tolist()
        raise ValueError(
            f'Speaker {speaker_id} not found in style DB. '
            f'Available speakers: {available}'
        )

    speaker_bert = db['bert_embeds'][speaker_mask].to(device)  # (M, 768)
    speaker_ss = db['style_ss'][speaker_mask].to(device)       # (M, 128)
    speaker_sp = db['style_sp'][speaker_mask].to(device)       # (M, 128)

    # Cosine similarity
    sims = F.cosine_similarity(query_embed, speaker_bert, dim=1)  # (M,)

    # Compact DB: pick the single best match; Full DB: sample from top 10
    db_type = db.get('db_type', 'full')
    if db_type == 'compact':
        pick = sims.argmax().item()
    else:
        k = min(10, len(sims))
        top_indices = sims.topk(k).indices
        pick = top_indices[torch.randint(len(top_indices), (1,)).item()].item()

    ref_ss = speaker_ss[pick].unsqueeze(0)  # (1, 128)
    ref_sp = speaker_sp[pick].unsqueeze(0)  # (1, 128)

    print(f'Selected style entry (similarity: {sims[pick].item():.4f}, db_type: {db_type})')
    return ref_ss, ref_sp


def load_repr_style(style_db_path, speaker_id, device='cuda'):
    """Load per-speaker representative style vectors from the style DB.

    The representative vector is the sample nearest to the centroid of the
    largest cluster (by K-means), avoiding the instability of averaged vectors.

    Returns:
        ref_ss: (1, 128) representative acoustic style
        ref_sp: (1, 128) representative prosodic style
    """
    db = torch.load(style_db_path, map_location=device)

    if 'repr_style_ss' not in db:
        raise ValueError(
            'Style DB does not contain representative vectors. '
            'Please rebuild with: python precompute_styles.py --model-dir <dir>'
        )

    if speaker_id not in db['repr_style_ss']:
        available = list(db['repr_style_ss'].keys())
        raise ValueError(
            f'Speaker {speaker_id} not found in style DB. '
            f'Available speakers: {available}'
        )

    ref_ss = db['repr_style_ss'][speaker_id].unsqueeze(0).to(device)
    ref_sp = db['repr_style_sp'][speaker_id].unsqueeze(0).to(device)
    print(f'Using representative style for speaker {speaker_id}')
    return ref_ss, ref_sp


@torch.no_grad()
def synthesize(model, model_params, text, ref_ss, ref_sp, device='cuda',
               diffusion_steps=5, sr=24000):
    """Run TTS pipeline: text -> waveform."""
    text_cleaner = TextCleaner()

    # Phonemize
    phonemized = smart_phonemize(text)
    print(f'Phonemized: {phonemized}')

    # Text to token indices
    tokens = text_cleaner(phonemized)
    tokens.insert(0, 0)   # start token
    tokens.append(0)       # end token
    tokens = torch.LongTensor(tokens).unsqueeze(0).to(device)
    input_lengths = torch.LongTensor([tokens.shape[1]]).to(device)
    text_mask = length_to_mask(input_lengths).to(device)

    # Text encoding
    t_en = model.text_encoder(tokens, input_lengths, text_mask)

    # BERT encoding
    bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
    d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

    # Step 1: Predict durations (dummy alignment — only durations needed here)
    s_dur = ref_sp  # prosodic style from reference
    T = tokens.shape[1]
    dummy_aln = torch.zeros(1, T, T, device=device)
    d, _ = model.predictor(d_en, s_dur, input_lengths,
                           alignment=dummy_aln, m=text_mask)

    # Build alignment from predicted durations
    # duration_proj outputs logits; apply sigmoid before summing (matches training code)
    d_rounded = torch.clamp(torch.round(torch.sigmoid(d).sum(axis=-1)), min=1)
    d_int = d_rounded.long().squeeze(0)

    total_mel_len = d_int.sum().item()
    alignment = torch.zeros(1, T, int(total_mel_len), device=device)
    asr_aligned = torch.zeros(1, t_en.shape[1], int(total_mel_len), device=device)
    pos = 0
    for j in range(d_int.shape[0]):
        dur = d_int[j].item()
        alignment[0, j, pos:pos+dur] = 1.0
        asr_aligned[0, :, pos:pos+dur] = t_en[0, :, j:j+1].expand(-1, dur)
        pos += dur

    # Step 2: Re-run predictor with real alignment to get aligned DurationEncoder
    # features (p_en, 640-dim) needed for F0 prediction
    _, p_en = model.predictor(d_en, s_dur, input_lengths,
                              alignment=alignment, m=text_mask)

    # Step 3: Predict F0 and energy norm
    F0_fake, N_fake = model.predictor(texts=p_en, style=s_dur, f0=True)

    # Diffusion-based style refinement (optional)
    multispeaker = model_params.multispeaker
    if diffusion_steps > 0:
        sampler = DiffusionSampler(
            model.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
            clamp=False,
        )
        ref = torch.cat([ref_ss, ref_sp], dim=1)
        noise = torch.randn(1, 1, ref_ss.shape[-1] + ref_sp.shape[-1], device=device)

        if multispeaker:
            s_preds = sampler(
                noise=noise,
                embedding=bert_dur,
                embedding_scale=1,
                features=ref,
                embedding_mask_proba=0.1,
                num_steps=diffusion_steps,
            ).squeeze(1)
        else:
            s_preds = sampler(
                noise=noise,
                embedding=bert_dur,
                embedding_scale=1,
                embedding_mask_proba=0.1,
                num_steps=diffusion_steps,
            ).squeeze(1)

        # Split predicted style into acoustic + prosodic
        style_dim = ref_ss.shape[-1]
        s_acoustic = s_preds[:, :style_dim]
    else:
        s_acoustic = ref_ss

    # Decode to waveform
    y_rec = model.decoder(asr_aligned, F0_fake, N_fake, s_acoustic)

    wav = y_rec.cpu().numpy().squeeze()
    return wav


def main():
    parser = argparse.ArgumentParser(description='Tsukasa Speech TTS Inference')
    parser.add_argument('--model-dir', '-d', default=None,
                        help='Model directory (e.g. Data/output/MidVRAM). '
                             'Auto-discovers config, checkpoint, and style DB.')
    parser.add_argument('--config', '-c', default=None, help='Path to config YAML (override)')
    parser.add_argument('--checkpoint', '-m', default=None, help='Path to inference checkpoint (override)')
    parser.add_argument('--reference', '-r', default=None, help='Path to reference speaker audio (.wav)')
    parser.add_argument('--speaker', '-s', type=int, default=0, help='Speaker ID (default: 0)')
    parser.add_argument('--style-db', default=None, help='Path to precomputed style DB (override)')
    parser.add_argument('--use-style-db-search', action='store_true',
                        help='Enable text-similarity search in style DB (default: use representative style)')
    parser.add_argument('--text', '-t', required=True, help='Text to synthesize')
    parser.add_argument('--output', '-o', default='output.wav', help='Output WAV path')
    parser.add_argument('--diffusion-steps', type=int, default=5, help='Number of diffusion steps (0 to skip)')
    parser.add_argument('--device', default='cuda', help='Device (cuda or cpu)')
    args = parser.parse_args()

    # Resolve paths: --model-dir auto-discovers, explicit flags override
    config_path = args.config
    checkpoint_path = args.checkpoint
    style_db_path = args.style_db

    if args.model_dir is not None:
        auto_config, auto_ckpt, auto_style_db = resolve_model_dir(args.model_dir)
        if config_path is None:
            config_path = auto_config
        if checkpoint_path is None:
            checkpoint_path = auto_ckpt
        if style_db_path is None:
            style_db_path = auto_style_db

    if config_path is None or checkpoint_path is None:
        parser.error('Either --model-dir or both --config and --checkpoint are required')

    config = yaml.safe_load(open(config_path))
    sr = config['preprocess_params'].get('sr', 24000)
    device = args.device

    print(f'Config:     {config_path}')
    print(f'Checkpoint: {checkpoint_path}')
    if style_db_path:
        print(f'Style DB:   {style_db_path}')
    print(f'Loading model...')
    model, model_params = load_inference_model(config, checkpoint_path, device)

    if args.reference is not None:
        print(f'Extracting style from {args.reference}...')
        ref_ss, ref_sp = compute_ref_style(model, args.reference, sr=sr, device=device)
    elif args.use_style_db_search:
        if style_db_path is None:
            parser.error('--use-style-db-search requires a style DB (--style-db or in --model-dir)')
        print(f'Looking up style for speaker {args.speaker} (text-similarity search)...')
        ref_ss, ref_sp = lookup_style_from_db(
            model, args.text, style_db_path, args.speaker, device=device,
        )
    elif style_db_path is not None:
        print(f'Loading representative style for speaker {args.speaker}...')
        ref_ss, ref_sp = load_repr_style(style_db_path, args.speaker, device=device)
    else:
        parser.error('No style source available. Provide --reference, --style-db, '
                     'or ensure style_db.pt exists in --model-dir')

    print(f'Synthesizing: "{args.text}"')
    wav = synthesize(
        model, model_params, args.text,
        ref_ss, ref_sp,
        device=device,
        diffusion_steps=args.diffusion_steps,
        sr=sr,
    )

    sf.write(args.output, wav, sr)
    print(f'Saved to {args.output} ({len(wav) / sr:.2f}s, {sr}Hz)')


if __name__ == '__main__':
    main()
