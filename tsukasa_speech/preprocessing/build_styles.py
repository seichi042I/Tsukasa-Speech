"""Precompute a text-aware style database for reference-free inference.

Iterates over training data and extracts per-utterance:
  - BERT text embedding (mean-pooled, 768-dim)
  - Acoustic style vector (128-dim)
  - Prosodic style vector (128-dim)
  - Speaker ID

Usage (simple — auto-discovers config and checkpoint):
    python precompute_styles.py --model-dir Data/output/MidVRAM

Usage (explicit paths):
    python precompute_styles.py \
        --config Configs/config.yml \
        --checkpoint Data/output/MidVRAM/inference/inference_5000.pth \
        --output Data/output/MidVRAM/inference/style_db.pt \
        [--batch-size 16] [--device cuda]
"""

import argparse
import yaml
import numpy as np
import torch
import librosa
import soundfile as sf
import os

from tsukasa_speech.data.text import TextCleaner
from tsukasa_speech.data.mel import preprocess
from tsukasa_speech.utils.common import length_to_mask
from tsukasa_speech.preprocessing.compact_styles import kmeans

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

    style_ss = torch.stack(all_ss)
    style_sp = torch.stack(all_sp)
    speaker_ids = torch.tensor(all_spk)

    # Compute per-speaker representative style vectors via clustering:
    # cluster style vectors, find the largest cluster, pick the sample
    # nearest to that cluster's centroid as the representative.
    unique_speakers = speaker_ids.unique().sort().values
    repr_style_ss = {}
    repr_style_sp = {}
    num_repr_clusters = 16

    for spk in unique_speakers:
        spk_val = spk.item()
        mask = speaker_ids == spk_val
        spk_ss = style_ss[mask]
        spk_sp = style_sp[mask]
        n_samples = spk_ss.shape[0]

        if n_samples <= 1:
            repr_style_ss[spk_val] = spk_ss[0]
            repr_style_sp[spk_val] = spk_sp[0]
            print(f'  [style_db] Speaker {spk_val}: representative from {n_samples} entry')
            continue

        k = min(num_repr_clusters, n_samples)
        style_concat = torch.cat([spk_ss, spk_sp], dim=1)
        centroids, assignments = kmeans(style_concat, k)

        # Find the cluster with the most samples
        cluster_counts = torch.bincount(assignments, minlength=k)
        largest_cluster = cluster_counts.argmax().item()

        # Find the sample nearest to the centroid of the largest cluster
        cluster_mask = assignments == largest_cluster
        cluster_indices = cluster_mask.nonzero(as_tuple=True)[0]
        cluster_points = style_concat[cluster_indices]
        centroid = centroids[largest_cluster]
        dists = (cluster_points - centroid.unsqueeze(0)).norm(dim=1)
        nearest_idx = cluster_indices[dists.argmin().item()].item()

        repr_style_ss[spk_val] = spk_ss[nearest_idx]
        repr_style_sp[spk_val] = spk_sp[nearest_idx]
        print(f'  [style_db] Speaker {spk_val}: representative from cluster {largest_cluster} '
              f'({cluster_counts[largest_cluster].item()}/{n_samples} samples)')

    return {
        'db_type': 'full',
        'bert_embeds': torch.stack(all_bert),
        'style_ss': style_ss,
        'style_sp': style_sp,
        'speaker_ids': speaker_ids,
        'repr_style_ss': repr_style_ss,
        'repr_style_sp': repr_style_sp,
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
    print(f'  repr_style_ss: {list(db["repr_style_ss"].keys())} speakers')
    print(f'  repr_style_sp: {list(db["repr_style_sp"].keys())} speakers')


def main():
    parser = argparse.ArgumentParser(description='Precompute style database for reference-free inference')
    parser.add_argument('--model-dir', '-d', default=None,
                        help='Model directory (e.g. Data/output/MidVRAM). '
                             'Auto-discovers config and checkpoint, saves to inference/style_db.pt')
    parser.add_argument('--config', '-c', default=None, help='Path to config YAML (override)')
    parser.add_argument('--checkpoint', '-m', default=None, help='Path to inference checkpoint (override)')
    parser.add_argument('--output', '-o', default=None, help='Output .pt path for style DB (override)')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for processing')
    parser.add_argument('--device', default='cuda', help='Device (cuda or cpu)')
    args = parser.parse_args()

    from tsukasa_speech.inference.model_loader import load_inference_model, resolve_model_dir

    config_path = args.config
    checkpoint_path = args.checkpoint
    output_path = args.output

    if args.model_dir is not None:
        auto_config, auto_ckpt, _ = resolve_model_dir(args.model_dir)
        if config_path is None:
            config_path = auto_config
        if checkpoint_path is None:
            checkpoint_path = auto_ckpt
        if output_path is None:
            output_path = os.path.join(args.model_dir, 'inference', 'style_db.pt')

    if config_path is None or checkpoint_path is None:
        parser.error('Either --model-dir or both --config and --checkpoint are required')
    if output_path is None:
        parser.error('Either --model-dir or --output is required')

    config = yaml.safe_load(open(config_path))
    device = args.device
    sr = config['preprocess_params'].get('sr', 24000)
    data_params = config['data_params']

    print(f'Loading model from {checkpoint_path}...')
    model, _ = load_inference_model(config, checkpoint_path, device)

    db = build_style_db(
        model,
        train_data_path=data_params['train_data'],
        root_path=data_params.get('root_path', ''),
        sr=sr,
        batch_size=args.batch_size,
        device=device,
    )
    save_style_db(db, output_path)


if __name__ == '__main__':
    main()
