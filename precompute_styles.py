"""Precompute a text-aware style database for reference-free inference.

Iterates over training data and extracts per-utterance:
  - BERT text embedding (mean-pooled, 768-dim)
  - Acoustic style vector (128-dim)
  - Prosodic style vector (128-dim)
  - Speaker ID

Usage:
    python precompute_styles.py \
        --config Configs/config_high_vram.yml \
        --checkpoint Data/output/HighVRAM/inference_5000.pth \
        --output Data/output/HighVRAM/style_db.pt \
        [--batch-size 16] [--device cuda]
"""

import argparse
import yaml
import numpy as np
import torch
import librosa
import soundfile as sf
import os

from meldataset import TextCleaner, preprocess
from utils import length_to_mask

# StyleEncoder has 4x half-downsample then 5x5 conv → mel needs >= 80 frames
MIN_MEL_FRAMES = 80


@torch.no_grad()
def build_style_db(model, train_data_path, root_path='', sr=24000,
                   batch_size=16, device='cuda'):
    """Build a style DB dict from a trained model and training data list.

    Args:
        model: Munch of loaded model modules (already on device, eval mode).
        train_data_path: Path to train_list.txt (wave_path|phonemized_text|speaker_id).
        root_path: Root directory prepended to wave paths.
        sr: Sample rate.
        batch_size: Batch size for BERT forward passes.
        device: torch device string.

    Returns:
        dict with keys 'bert_embeds', 'style_ss', 'style_sp', 'speaker_ids'.
    """
    with open(train_data_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [l.strip() for l in f if l.strip()]

    entries = []
    for line in lines:
        parts = line.split('|')
        if len(parts) >= 2:
            wave_path = parts[0]
            text = parts[1]
            speaker_id = int(parts[2]) if len(parts) >= 3 else 0
            entries.append((wave_path, text, speaker_id))

    print(f'[style_db] Processing {len(entries)} utterances...')

    text_cleaner = TextCleaner()

    all_bert = []
    all_ss = []
    all_sp = []
    all_spk = []

    for start in range(0, len(entries), batch_size):
        batch_entries = entries[start:start + batch_size]

        # Tokenize texts and load audio for the batch
        token_lists = []
        mel_list = []
        spk_ids = []

        for wave_path, text, speaker_id in batch_entries:
            tokens = text_cleaner(text)
            if len(tokens) == 0:
                continue
            tokens = [0] + tokens + [0]

            full_path = os.path.join(root_path, wave_path) if root_path else wave_path
            try:
                wave, _ = librosa.load(full_path, sr=sr)
            except Exception as e:
                print(f'  Skipping {wave_path}: {e}')
                continue

            # Match training data pipeline: add 5000-sample silence padding
            wave = np.concatenate([np.zeros(5000), wave, np.zeros(5000)])

            mel = preprocess(wave)  # (1, 80, T)
            if mel.shape[-1] < MIN_MEL_FRAMES:
                continue
            token_lists.append(tokens)
            mel_list.append(mel)
            spk_ids.append(speaker_id)

        if not token_lists:
            continue

        # Pad tokens for batched BERT
        max_tok_len = max(len(t) for t in token_lists)
        tokens_padded = torch.zeros(len(token_lists), max_tok_len, dtype=torch.long, device=device)
        input_lengths = torch.zeros(len(token_lists), dtype=torch.long, device=device)
        for i, tok in enumerate(token_lists):
            tokens_padded[i, :len(tok)] = torch.LongTensor(tok)
            input_lengths[i] = len(tok)

        text_mask = length_to_mask(input_lengths).to(device)

        bert_out = model.bert(tokens_padded, attention_mask=(~text_mask).int())
        mask_expand = (~text_mask).unsqueeze(-1).float()
        bert_mean = (bert_out * mask_expand).sum(dim=1) / mask_expand.sum(dim=1)

        for i in range(len(mel_list)):
            mel = mel_list[i].to(device)
            ss = model.style_encoder(mel.unsqueeze(1))
            sp = model.predictor_encoder(mel.unsqueeze(1))

            all_bert.append(bert_mean[i].cpu())
            all_ss.append(ss.squeeze(0).cpu())
            all_sp.append(sp.squeeze(0).cpu())
            all_spk.append(spk_ids[i])

        processed = min(start + batch_size, len(entries))
        if processed % (batch_size * 10) == 0 or processed == len(entries):
            print(f'  [style_db] {processed}/{len(entries)} utterances processed')

    print(f'[style_db] Collected {len(all_bert)} style entries')

    return {
        'bert_embeds': torch.stack(all_bert),
        'style_ss': torch.stack(all_ss),
        'style_sp': torch.stack(all_sp),
        'speaker_ids': torch.tensor(all_spk),
    }


def save_style_db(db, output_path):
    """Save a style DB dict and print summary."""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    torch.save(db, output_path)
    print(f'[style_db] Saved to {output_path}')
    print(f'  bert_embeds: {db["bert_embeds"].shape}')
    print(f'  style_ss:    {db["style_ss"].shape}')
    print(f'  style_sp:    {db["style_sp"].shape}')
    print(f'  speaker_ids: {db["speaker_ids"].shape}')


def main():
    parser = argparse.ArgumentParser(description='Precompute style database for reference-free inference')
    parser.add_argument('--config', '-c', required=True, help='Path to config YAML')
    parser.add_argument('--checkpoint', '-m', required=True, help='Path to inference checkpoint (.pth)')
    parser.add_argument('--output', '-o', required=True, help='Output .pt path for style DB')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for processing')
    parser.add_argument('--device', default='cuda', help='Device (cuda or cpu)')
    args = parser.parse_args()

    from inference import load_inference_model

    config = yaml.safe_load(open(args.config))
    device = args.device
    sr = config['preprocess_params'].get('sr', 24000)
    data_params = config['data_params']

    print(f'Loading model from {args.checkpoint}...')
    model, _ = load_inference_model(config, args.checkpoint, device)

    db = build_style_db(
        model,
        train_data_path=data_params['train_data'],
        root_path=data_params.get('root_path', ''),
        sr=sr,
        batch_size=args.batch_size,
        device=device,
    )
    save_style_db(db, args.output)


if __name__ == '__main__':
    main()
