"""Simple TTS inference script for Tsukasa Speech.

Usage (with reference audio):
    python inference.py \
        --config Configs/config_high_vram.yml \
        --checkpoint Data/output/HighVRAM/inference_5000.pth \
        --reference Data/speaker1/wav/sample.wav \
        --text "こんにちは、元気ですか" \
        --output output.wav \
        [--diffusion-steps 5]

Usage (reference-free with style DB):
    python inference.py \
        --config Configs/config_high_vram.yml \
        --checkpoint Data/output/HighVRAM/inference_5000.pth \
        --speaker 0 --style-db Data/output/HighVRAM/style_db.pt \
        --text "こんにちは、元気ですか" \
        --output output.wav \
        [--diffusion-steps 5]
"""

import argparse
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

    # Cosine similarity -> top 10 -> random pick
    sims = F.cosine_similarity(query_embed, speaker_bert, dim=1)  # (M,)
    k = min(10, len(sims))
    top_indices = sims.topk(k).indices
    pick = top_indices[torch.randint(len(top_indices), (1,)).item()].item()

    ref_ss = speaker_ss[pick].unsqueeze(0)  # (1, 128)
    ref_sp = speaker_sp[pick].unsqueeze(0)  # (1, 128)

    print(f'Selected style entry (similarity: {sims[pick].item():.4f})')
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
    d_rounded = torch.clamp(torch.round(d.sum(axis=-1)), min=1)
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
    parser.add_argument('--config', '-c', required=True, help='Path to config YAML')
    parser.add_argument('--checkpoint', '-m', required=True, help='Path to inference checkpoint (.pth)')
    parser.add_argument('--reference', '-r', default=None, help='Path to reference speaker audio (.wav)')
    parser.add_argument('--speaker', '-s', type=int, default=None, help='Speaker ID for reference-free inference')
    parser.add_argument('--style-db', default=None, help='Path to precomputed style DB (.pt)')
    parser.add_argument('--text', '-t', required=True, help='Text to synthesize')
    parser.add_argument('--output', '-o', default='output.wav', help='Output WAV path')
    parser.add_argument('--diffusion-steps', type=int, default=5, help='Number of diffusion steps (0 to skip)')
    parser.add_argument('--device', default='cuda', help='Device (cuda or cpu)')
    args = parser.parse_args()

    # Validate: need either --reference or (--speaker + --style-db)
    if args.reference is None and (args.speaker is None or args.style_db is None):
        parser.error('Either --reference or both --speaker and --style-db are required')

    config = yaml.safe_load(open(args.config))
    sr = config['preprocess_params'].get('sr', 24000)
    device = args.device

    print(f'Loading model from {args.checkpoint}...')
    model, model_params = load_inference_model(config, args.checkpoint, device)

    if args.reference is not None:
        print(f'Extracting style from {args.reference}...')
        ref_ss, ref_sp = compute_ref_style(model, args.reference, sr=sr, device=device)
    else:
        print(f'Looking up style for speaker {args.speaker} from {args.style_db}...')
        ref_ss, ref_sp = lookup_style_from_db(
            model, args.text, args.style_db, args.speaker, device=device,
        )

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
