"""Compact a full style DB into a lightweight per-speaker store.

For each speaker, keeps:
  - num_centroids (default 16) "average" entries (K-means cluster representatives)
  - num_outliers  (default 32) "outlier" entries (most distant from cluster centers)
  = 48 entries per speaker total

The output has the same dict format as the full style_db.pt, so inference.py
works with it unchanged.

Usage:
    python compact_style_db.py \
        --input Data/output/HighVRAM/style_db.pt \
        --output Data/output/HighVRAM/style_db_compact.pt \
        [--num-centroids 16] [--num-outliers 32]
"""

import argparse
import os
import torch


def kmeans(data, k, max_iters=100, seed=42):
    """K-means clustering with k-means++ initialization (pure torch).

    Args:
        data: (N, D) tensor
        k: number of clusters
        max_iters: maximum iterations

    Returns:
        centroids: (k, D)
        assignments: (N,) cluster index for each point
    """
    torch.manual_seed(seed)
    N, D = data.shape
    k = min(k, N)

    # K-means++ init
    idx = torch.randint(N, (1,)).item()
    centroid_list = [data[idx]]
    for _ in range(k - 1):
        stacked = torch.stack(centroid_list)  # (c, D)
        dists = torch.cdist(data, stacked)    # (N, c)
        min_dists = dists.min(dim=1).values   # (N,)
        probs = min_dists / min_dists.sum()
        chosen = torch.multinomial(probs, 1).item()
        centroid_list.append(data[chosen])

    centroids = torch.stack(centroid_list)  # (k, D)

    for _ in range(max_iters):
        dists = torch.cdist(data, centroids)       # (N, k)
        assignments = dists.argmin(dim=1)           # (N,)

        new_centroids = torch.zeros_like(centroids)
        for i in range(k):
            mask = assignments == i
            if mask.sum() > 0:
                new_centroids[i] = data[mask].mean(dim=0)
            else:
                new_centroids[i] = centroids[i]

        if torch.allclose(centroids, new_centroids, atol=1e-6):
            break
        centroids = new_centroids

    # Final assignment
    dists = torch.cdist(data, centroids)
    assignments = dists.argmin(dim=1)
    return centroids, assignments


def select_entries_for_speaker(bert, ss, sp, num_centroids, num_outliers):
    """Select representative + outlier entries for one speaker.

    Clusters on concatenated style vectors (ss||sp), then:
    1. For each centroid picks the nearest real data point (representative)
    2. Among remaining points, picks those most distant from their centroid (outliers)

    Returns indices into the speaker's arrays.
    """
    N = ss.shape[0]
    total = num_centroids + num_outliers

    # If speaker has fewer entries than requested, keep all
    if N <= total:
        return torch.arange(N)

    # Cluster on style vectors (acoustic + prosodic)
    style_concat = torch.cat([ss, sp], dim=1)  # (N, 256)
    centroids, assignments = kmeans(style_concat, num_centroids)

    # 1. Find nearest real point to each centroid
    dists_to_centroids = torch.cdist(style_concat, centroids)  # (N, k)
    representative_indices = []
    used = set()
    for c in range(centroids.shape[0]):
        # Sort by distance to this centroid, pick closest unused
        order = dists_to_centroids[:, c].argsort()
        for idx in order:
            idx_val = idx.item()
            if idx_val not in used:
                representative_indices.append(idx_val)
                used.add(idx_val)
                break

    # 2. Among remaining points, pick the num_outliers most distant from their centroid
    remaining_mask = torch.ones(N, dtype=torch.bool)
    for idx in representative_indices:
        remaining_mask[idx] = False

    remaining_indices = remaining_mask.nonzero(as_tuple=True)[0]  # indices into original
    if len(remaining_indices) > 0:
        # Distance of each remaining point to its assigned centroid
        remaining_assignments = assignments[remaining_indices]
        remaining_styles = style_concat[remaining_indices]
        assigned_centroids = centroids[remaining_assignments]  # (R, 256)
        point_dists = (remaining_styles - assigned_centroids).norm(dim=1)  # (R,)

        n_outliers = min(num_outliers, len(remaining_indices))
        top_outlier_positions = point_dists.topk(n_outliers).indices
        outlier_indices = remaining_indices[top_outlier_positions].tolist()
    else:
        outlier_indices = []

    selected = torch.tensor(representative_indices + outlier_indices, dtype=torch.long)
    return selected


def compact_style_db(db, num_centroids=16, num_outliers=32):
    """Compact a full style DB dict into a lightweight per-speaker store.

    Args:
        db: Full style DB dict (bert_embeds, style_ss, style_sp, speaker_ids).
        num_centroids: Number of centroid (average) entries per speaker.
        num_outliers: Number of outlier entries per speaker.

    Returns:
        Compact style DB dict (same format, fewer entries).
    """
    bert_embeds = db['bert_embeds']
    style_ss = db['style_ss']
    style_sp = db['style_sp']
    speaker_ids = db['speaker_ids']

    print(f'[compact] Full DB: {bert_embeds.shape[0]} entries')

    unique_speakers = speaker_ids.unique().sort().values
    print(f'[compact] Speakers: {unique_speakers.tolist()}')

    compact_bert = []
    compact_ss = []
    compact_sp = []
    compact_spk = []

    for spk in unique_speakers:
        spk_val = spk.item()
        mask = speaker_ids == spk_val
        spk_bert = bert_embeds[mask]
        spk_ss = style_ss[mask]
        spk_sp = style_sp[mask]
        n_total = spk_bert.shape[0]

        selected = select_entries_for_speaker(
            spk_bert, spk_ss, spk_sp,
            num_centroids, num_outliers,
        )

        compact_bert.append(spk_bert[selected])
        compact_ss.append(spk_ss[selected])
        compact_sp.append(spk_sp[selected])
        compact_spk.append(torch.full((len(selected),), spk_val, dtype=torch.long))

        print(f'  [compact] Speaker {spk_val}: {n_total} -> {len(selected)} entries')

    compact = {
        'bert_embeds': torch.cat(compact_bert),
        'style_ss': torch.cat(compact_ss),
        'style_sp': torch.cat(compact_sp),
        'speaker_ids': torch.cat(compact_spk),
    }

    n_compact = compact['bert_embeds'].shape[0]
    n_full = bert_embeds.shape[0]
    print(f'[compact] Result: {n_compact} entries (was {n_full}, {n_compact/n_full*100:.1f}%)')
    return compact


def main():
    parser = argparse.ArgumentParser(
        description='Compact a full style DB into a lightweight per-speaker store')
    parser.add_argument('--input', '-i', required=True, help='Path to full style_db.pt')
    parser.add_argument('--output', '-o', required=True, help='Output path for compact DB')
    parser.add_argument('--num-centroids', type=int, default=16,
                        help='Number of centroid (average) entries per speaker')
    parser.add_argument('--num-outliers', type=int, default=32,
                        help='Number of outlier entries per speaker')
    args = parser.parse_args()

    print(f'Loading {args.input}...')
    db = torch.load(args.input, map_location='cpu')

    compact = compact_style_db(db, args.num_centroids, args.num_outliers)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    torch.save(compact, args.output)
    print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()
