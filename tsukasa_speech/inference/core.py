# coding: utf-8
"""Core TTS inference: prosody prediction, waveform synthesis, and CLI entry point."""

import argparse
import os.path as osp

import numpy as np
import torch
import yaml
import soundfile as sf

from tsukasa_speech.data.text import TextCleaner
from tsukasa_speech.utils.common import length_to_mask, log_norm
from tsukasa_speech.utils.phonemize.mixed_phon import smart_phonemize
from tsukasa_speech.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

from tsukasa_speech.inference.model_loader import (
    load_inference_model,
    resolve_model_dir,
    normalize_config,
)
from tsukasa_speech.inference.style import (
    compute_ref_style,
    lookup_style_from_db,
    load_repr_style,
)


@torch.no_grad()
def predict_prosody(model, model_params, text, ref_ss, ref_sp, device='cuda',
                    diffusion_steps=5, sr=24000, style_strength=1.0):
    """Run TTS pipeline up to F0/energy prediction (before decoding).

    Returns a dict containing intermediate tensors needed for waveform synthesis.
    """
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

    # Dampen prosodic style toward zero for flatter prosody.
    s_dur = ref_sp * style_strength
    T = tokens.shape[1]
    dummy_aln = torch.zeros(1, T, T, device=device)
    d, _ = model.predictor(d_en, s_dur, input_lengths,
                           alignment=dummy_aln, m=text_mask)

    # Build alignment from predicted durations
    d_rounded = torch.clamp(torch.round(torch.sigmoid(d).sum(axis=-1)), min=1)
    d_int = d_rounded.long().squeeze(0)

    # Fix boundary tokens
    d_int[0] = 10
    d_int[-1] = 0

    # Cap per-token duration
    real = d_int[1:-1]
    if real.numel() > 0:
        dur_cap = int(torch.clamp(real.float().median() * 3, min=0, max=50).item())
        d_int[1:-1] = torch.clamp(real, max=dur_cap)

    total_mel_len = d_int.sum().item()
    alignment = torch.zeros(1, T, int(total_mel_len), device=device)
    asr_aligned = torch.zeros(1, t_en.shape[1], int(total_mel_len), device=device)
    pos = 0
    for j in range(d_int.shape[0]):
        dur = d_int[j].item()
        alignment[0, j, pos:pos+dur] = 1.0
        asr_aligned[0, :, pos:pos+dur] = t_en[0, :, j:j+1].expand(-1, dur)
        pos += dur

    # Re-run predictor with real alignment for F0 prediction
    _, p_en = model.predictor(d_en, s_dur, input_lengths,
                              alignment=alignment, m=text_mask)

    # Predict F0 and energy norm
    F0_fake, N_fake = model.predictor(texts=p_en, style=s_dur, f0=True)

    # Diffusion-based style refinement
    multispeaker = model_params.multispeaker
    if diffusion_steps > 0:
        sampler = DiffusionSampler(
            model.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
            clamp=False,
        )
        ref = torch.cat([ref_ss, s_dur], dim=1)
        noise = torch.randn(1, 1, ref_ss.shape[-1] + s_dur.shape[-1], device=device)

        if multispeaker:
            s_preds = sampler(
                noise=noise, embedding=bert_dur, embedding_scale=1,
                features=ref, embedding_mask_proba=0.1, num_steps=diffusion_steps,
            ).squeeze(1)
        else:
            s_preds = sampler(
                noise=noise, embedding=bert_dur, embedding_scale=1,
                embedding_mask_proba=0.1, num_steps=diffusion_steps,
            ).squeeze(1)

        style_dim = ref_ss.shape[-1]
        s_acoustic = s_preds[:, :style_dim]
    else:
        s_acoustic = ref_ss

    # Build phoneme boundary info for the editor
    # d_int values are downsampled by 2x from mel frames, so multiply by 2
    # to align with the F0/N arrays which are at full mel-frame resolution.
    phoneme_chars = list(phonemized)
    phonemes = []
    pos = d_int[0].item() * 2  # skip start token duration
    for k, ch in enumerate(phoneme_chars):
        if k + 1 < len(d_int):
            dur = d_int[k + 1].item() * 2
            phonemes.append({'label': ch, 'start_frame': pos, 'end_frame': pos + dur})
            pos += dur

    return {
        'F0': F0_fake,
        'N': N_fake,
        'F0_original': F0_fake.clone(),
        'N_original': N_fake.clone(),
        'asr_aligned': asr_aligned,
        's_acoustic': s_acoustic,
        'd_int': d_int,
        'phonemes': phonemes,
        'phonemized': phonemized,
        'sr': sr,
    }


@torch.no_grad()
def synthesize_from_prosody(model, prosody_state):
    """Decode waveform from pre-computed prosody state."""
    F0 = prosody_state['F0']
    N = prosody_state['N']
    asr_aligned = prosody_state['asr_aligned']
    s_acoustic = prosody_state['s_acoustic']
    sr = prosody_state['sr']

    y_rec = model.decoder(asr_aligned, F0, N, s_acoustic)
    wav = y_rec.cpu().numpy().squeeze()

    # Trim trailing silence / low-energy artifacts.
    frame_len = int(sr * 0.02)
    energy_threshold = 0.01
    end = len(wav)
    while end > frame_len:
        frame = wav[end - frame_len:end]
        if np.abs(frame).mean() > energy_threshold:
            break
        end -= frame_len
    if end < len(wav):
        fade_samples = int(sr * 0.005)
        end = min(end + fade_samples, len(wav))
        wav = wav[:end]
        wav[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)

    return wav


@torch.no_grad()
def synthesize(model, model_params, text, ref_ss, ref_sp, device='cuda',
               diffusion_steps=5, sr=24000, style_strength=1.0):
    """Run TTS pipeline: text -> waveform (backward-compatible wrapper)."""
    state = predict_prosody(model, model_params, text, ref_ss, ref_sp,
                            device=device, diffusion_steps=diffusion_steps,
                            sr=sr, style_strength=style_strength)
    return synthesize_from_prosody(model, state)


def create_f0_plot(prosody_states, hop_length=300):
    """Create a Plotly figure showing F0 curves for all sentences.

    Args:
        prosody_states: list of dicts from predict_prosody()
        hop_length: hop length in samples (default 300 for 24kHz / 80fps)

    Returns:
        plotly Figure
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    frame_offset = 0

    for i, state in enumerate(prosody_states):
        sr = state['sr']
        f0 = state['F0'].squeeze().cpu().numpy()
        n_frames = len(f0)

        # Time axis in seconds
        times = (np.arange(n_frames) + frame_offset) * hop_length / sr

        # Original F0 (thin grey dashed)
        if 'F0_original' in state:
            f0_orig = state['F0_original'].squeeze().cpu().numpy()
            has_diff = not np.allclose(f0, f0_orig, atol=1e-6)
            if has_diff:
                fig.add_trace(go.Scatter(
                    x=times, y=f0_orig,
                    mode='lines',
                    line=dict(color='rgba(150,150,150,0.5)', width=1, dash='dash'),
                    name=f'\u6587{i+1} (\u539f\u672c)',
                    showlegend=True,
                ))

        # Current F0 (solid blue)
        fig.add_trace(go.Scatter(
            x=times, y=f0,
            mode='lines',
            line=dict(color='#1f77b4', width=2),
            name=f'\u6587{i+1}',
        ))

        # Sentence boundary
        if i > 0:
            boundary_time = frame_offset * hop_length / sr
            fig.add_vline(
                x=boundary_time,
                line_dash='dash',
                line_color='rgba(100,100,100,0.4)',
                annotation_text=f'\u6587{i+1}',
                annotation_position='top',
            )

        frame_offset += n_frames

    fig.update_layout(
        xaxis_title='\u6642\u9593 (\u79d2)',
        yaxis_title='F0',
        height=250,
        margin=dict(l=50, r=20, t=30, b=40),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    return fig


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
    parser.add_argument('--style-strength', type=float, default=1.0,
                        help='Prosodic style strength 0.0\u20131.0 (0=flat, 1=full style)')
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
        style_strength=args.style_strength,
    )

    sf.write(args.output, wav, sr)
    print(f'Saved to {args.output} ({len(wav) / sr:.2f}s, {sr}Hz)')


if __name__ == '__main__':
    main()
