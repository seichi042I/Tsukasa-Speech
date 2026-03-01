#!/usr/bin/env python3
"""
Preprocess transcript_utf8.txt into pipe-separated IPA phonemized format
for Tsukasa Speech (StyleTTS2) training.

Supports multi-speaker auto-discovery from directory structure:
  Data/
      speaker_name1/
          wav/
              audio_001.wav
          transcript_utf8.txt
      speaker_name2/
          wav/
              audio_001.wav
          transcript_utf8.txt

Input format (colon-separated, per speaker transcript_utf8.txt):
  IORI_0001.wav:月の宝…:ツキノタカラ

Output format (pipe-separated, unified train/val lists):
  Data/speaker_name/wav/IORI_0001.wav|tsɯki no takaɽa|0

Usage:
  python preprocess_data.py [--data-dir Data] [--val-ratio 0.1] [--n_jobs 4]
  python preprocess_data.py --data-dir Data --n_jobs 8 --max-duration 15.0
  python preprocess_data.py --data-dir Data --cache-wavs --cache-dir /tmp/wave_cache
"""
import os
import sys
import random
import argparse
import time
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Speaker discovery
# ---------------------------------------------------------------------------

def discover_speakers(data_dir):
    """Auto-discover speaker directories containing wav/ and transcript_utf8.txt."""
    speakers = []
    for name in sorted(os.listdir(data_dir)):
        speaker_path = os.path.join(data_dir, name)
        if not os.path.isdir(speaker_path):
            continue
        wav_dir = os.path.join(speaker_path, 'wav')
        transcript = os.path.join(speaker_path, 'transcript_utf8.txt')
        if os.path.isdir(wav_dir) and os.path.isfile(transcript):
            speakers.append(name)
        elif os.path.isdir(wav_dir):
            print(f"WARNING: {name}/ has wav/ but no transcript_utf8.txt, skipping")
    return speakers


# ---------------------------------------------------------------------------
# Phonemization (parallelizable)
# ---------------------------------------------------------------------------

def _phonemize_single(item):
    """Phonemize a single (wav_rel_path, japanese_text, speaker_id) tuple.

    Returns (entry_string, None) on success or (None, skip_reason) on failure.
    This function is designed to be called from a multiprocessing Pool.
    """
    wav_rel_path, japanese_text, speaker_id, filename = item

    if len(japanese_text) < 2:
        return None, f"Text too short: {filename} -> '{japanese_text}'"

    # Skip non-verbal sounds enclosed in parentheses, e.g. (泣き声), (溜息)
    import re
    if re.fullmatch(r'[（(].+?[)）]', japanese_text.strip()):
        return None, f"Non-verbal sound: {filename} -> '{japanese_text}'"

    try:
        from Utils.phonemize.mixed_phon import smart_phonemize
        ipa_text = smart_phonemize(japanese_text)
        if not ipa_text or len(ipa_text.strip()) < 2:
            return None, f"Empty phonemization: {filename} -> '{japanese_text}'"
    except Exception as e:
        return None, f"ERROR phonemizing {filename}: {e}"

    # Skip entries where IPA text contains parenthesized non-verbal sounds
    if '(' in ipa_text or ')' in ipa_text or '（' in ipa_text or '）' in ipa_text:
        return None, f"Non-verbal sound in IPA: {filename} -> '{ipa_text}'"

    entry = f"{wav_rel_path}|{ipa_text}|{speaker_id}"
    return entry, None


def collect_items_for_speaker(data_dir, speaker_name, speaker_id, max_duration=None, sr=24000):
    """Read a speaker's transcript and return items for parallel phonemization.

    If max_duration is set, WAV files longer than max_duration seconds are skipped.
    """
    transcript_path = os.path.join(data_dir, speaker_name, 'transcript_utf8.txt')
    wav_dir = os.path.join(data_dir, speaker_name, 'wav')

    items = []
    skipped = []

    with open(transcript_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            if line.startswith('処理開始'):
                continue

            parts = line.split(':')
            if len(parts) < 2:
                skipped.append(f"Malformed line {line_num}: {line}")
                continue

            filename = parts[0].strip()
            japanese_text = parts[1].strip()

            wav_path = os.path.join(wav_dir, filename)
            if not os.path.exists(wav_path):
                skipped.append(f"WAV not found: {wav_path}")
                continue

            # Duration filter
            if max_duration is not None:
                try:
                    info = sf.info(wav_path)
                    duration = info.duration
                    if duration > max_duration:
                        skipped.append(f"Too long ({duration:.1f}s > {max_duration}s): {filename}")
                        continue
                except Exception as e:
                    skipped.append(f"Cannot read audio info for {filename}: {e}")
                    continue

            rel_path = os.path.join(data_dir, speaker_name, 'wav', filename)
            items.append((rel_path, japanese_text, speaker_id, filename))

    return items, skipped


# ---------------------------------------------------------------------------
# WAV cache: resample to 24kHz + pad + save as .npy (parallelizable)
# ---------------------------------------------------------------------------

def _cache_single_wav(item, cache_dir, target_sr=24000):
    """Cache a single WAV file as a numpy array.

    Returns (wav_path, True, None) on success/skip, or (wav_path, False, error_msg) on failure.
    """
    wav_path = item

    cache_key = wav_path.replace(os.sep, '_').replace('/', '_')
    cache_file = os.path.join(cache_dir, cache_key + '.npy')

    if os.path.exists(cache_file):
        return wav_path, True, None  # already cached

    try:
        import librosa
        wave, sr = sf.read(wav_path)
        if wave.ndim > 1:
            wave = wave[:, 0]
        if sr != target_sr:
            wave = librosa.resample(wave, orig_sr=sr, target_sr=target_sr)
        wave = np.concatenate([np.zeros([5000]), wave, np.zeros([5000])], axis=0)
        np.save(cache_file, wave)
        return wav_path, True, None
    except Exception as e:
        return wav_path, False, str(e)


# ---------------------------------------------------------------------------
# Mel spectrogram pre-extraction (parallelizable)
# ---------------------------------------------------------------------------

def _extract_mel_single(item, cache_dir, mel_cache_dir, target_sr=24000):
    """Pre-extract mel spectrogram for a single WAV file.

    Reads from wav cache if available, otherwise from raw WAV.
    Returns (wav_path, True, None) on success or (wav_path, False, error_msg) on failure.
    """
    import torch
    import torchaudio

    wav_path = item
    mel_key = wav_path.replace(os.sep, '_').replace('/', '_')
    mel_file = os.path.join(mel_cache_dir, mel_key + '.mel.pt')

    if os.path.exists(mel_file):
        return wav_path, True, None

    try:
        # Try loading from wav cache first
        wav_cache_key = wav_path.replace(os.sep, '_').replace('/', '_')
        wav_cache_file = os.path.join(cache_dir, wav_cache_key + '.npy')

        if os.path.exists(wav_cache_file):
            wave = np.load(wav_cache_file)
        else:
            import librosa
            wave, sr = sf.read(wav_path)
            if wave.ndim > 1:
                wave = wave[:, 0]
            if sr != target_sr:
                wave = librosa.resample(wave, orig_sr=sr, target_sr=target_sr)
            wave = np.concatenate([np.zeros([5000]), wave, np.zeros([5000])], axis=0)

        to_mel = torchaudio.transforms.MelSpectrogram(
            n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
        wave_tensor = torch.from_numpy(wave).float()
        mel_tensor = to_mel(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - (-4)) / 4

        torch.save(mel_tensor, mel_file)
        return wav_path, True, None
    except Exception as e:
        return wav_path, False, str(e)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess transcripts for multi-speaker training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic preprocessing with 4 parallel workers
  python preprocess_data.py --data-dir Data

  # Use 8 workers, filter out audio longer than 15s
  python preprocess_data.py --data-dir Data --n_jobs 8 --max-duration 15.0

  # Also pre-cache WAV files (resample to 24kHz)
  python preprocess_data.py --data-dir Data --cache-wavs --cache-dir /tmp/wave_cache

  # Full pipeline: phonemize + cache WAVs + pre-extract mel spectrograms
  python preprocess_data.py --data-dir Data --cache-wavs --extract-mels --n_jobs 8
""")
    parser.add_argument("--data-dir", default="Data",
                        help="Root data directory (default: Data)")
    parser.add_argument("--train-output", default=None,
                        help="Output train list path (default: DATA_DIR/train_list.txt)")
    parser.add_argument("--val-output", default=None,
                        help="Output val list path (default: DATA_DIR/val_list.txt)")
    parser.add_argument("--val-ratio", type=float, default=0.0,
                        help="Validation split ratio (default: 0.0, all data used for training)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--n_jobs", type=int, default=4,
                        help="Number of parallel processes (default: 4)")
    parser.add_argument("--max-duration", type=float, default=None,
                        help="Skip audio files longer than this duration in seconds (default: no limit)")
    parser.add_argument("--cache-wavs", action="store_true",
                        help="Pre-cache WAV files as resampled numpy arrays")
    parser.add_argument("--cache-dir", default=None,
                        help="Directory for WAV cache (default: DATA_DIR/wave_cache)")
    parser.add_argument("--extract-mels", action="store_true",
                        help="Pre-extract mel spectrograms (requires --cache-wavs or raw WAVs)")
    parser.add_argument("--mel-cache-dir", default=None,
                        help="Directory for mel cache (default: DATA_DIR/mel_cache)")
    args = parser.parse_args()

    if args.train_output is None:
        args.train_output = os.path.join(args.data_dir, 'train_list.txt')
    if args.val_output is None:
        args.val_output = os.path.join(args.data_dir, 'val_list.txt')
    if args.cache_dir is None:
        args.cache_dir = os.path.join(args.data_dir, 'wave_cache')
    if args.mel_cache_dir is None:
        args.mel_cache_dir = os.path.join(args.data_dir, 'mel_cache')

    n_jobs = min(args.n_jobs, cpu_count())
    random.seed(args.seed)

    print(f"=== Preprocessing Pipeline ===")
    print(f"  Data directory:   {args.data_dir}")
    print(f"  Parallel workers: {n_jobs}")
    if args.max_duration:
        print(f"  Max duration:     {args.max_duration}s")
    print()

    # ---- Step 1: Discover speakers ----
    speakers = discover_speakers(args.data_dir)
    if not speakers:
        print(f"ERROR: No speaker directories found in {args.data_dir}/")
        print(f"Expected structure: {args.data_dir}/speaker_name/wav/*.wav + transcript_utf8.txt")
        sys.exit(1)

    print(f"Discovered {len(speakers)} speaker(s): {', '.join(speakers)}")

    # ---- Step 2: Collect items (serial, fast I/O only) ----
    all_items = []
    total_skipped = 0

    for speaker_id, speaker_name in enumerate(speakers):
        print(f"\nCollecting speaker {speaker_id}: {speaker_name}")
        items, skipped = collect_items_for_speaker(
            args.data_dir, speaker_name, speaker_id,
            max_duration=args.max_duration)
        all_items.extend(items)
        total_skipped += len(skipped)
        print(f"  Candidates: {len(items)}, Skipped: {len(skipped)}")
        for s in skipped[:5]:
            print(f"    - {s}")
        if len(skipped) > 5:
            print(f"    ... and {len(skipped) - 5} more")

    if not all_items:
        print("ERROR: No valid entries found after filtering.")
        sys.exit(1)

    # ---- Step 3: Parallel phonemization ----
    print(f"\n=== Phonemizing {len(all_items)} entries with {n_jobs} workers ===")
    t0 = time.time()

    if n_jobs <= 1:
        results = [_phonemize_single(item) for item in all_items]
    else:
        with Pool(n_jobs) as pool:
            results = pool.map(_phonemize_single, all_items, chunksize=max(1, len(all_items) // (n_jobs * 4)))

    all_entries = []
    phonemize_skipped = 0
    for entry, skip_reason in results:
        if entry is not None:
            all_entries.append(entry)
        else:
            phonemize_skipped += 1

    elapsed = time.time() - t0
    total_skipped += phonemize_skipped
    print(f"  Phonemized: {len(all_entries)}, Skipped: {phonemize_skipped}")
    print(f"  Time: {elapsed:.1f}s ({len(all_items) / max(elapsed, 0.1):.0f} items/s)")

    if not all_entries:
        print("ERROR: No valid entries after phonemization.")
        sys.exit(1)

    # ---- Step 4: Shuffle and split ----
    random.shuffle(all_entries)
    split_idx = max(1, int(len(all_entries) * (1 - args.val_ratio)))
    train_entries = all_entries[:split_idx]
    val_entries = all_entries[split_idx:]

    with open(args.train_output, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_entries) + '\n')

    with open(args.val_output, 'w', encoding='utf-8') as f:
        if val_entries:
            f.write('\n'.join(val_entries) + '\n')
        else:
            f.write('')

    print(f"\n--- Preprocessing Summary ---")
    print(f"Total speakers:      {len(speakers)}")
    print(f"Total valid entries:  {len(all_entries)}")
    print(f"Total skipped:       {total_skipped}")
    print(f"Train entries: {len(train_entries)} -> {args.train_output}")
    print(f"Val entries:   {len(val_entries)} -> {args.val_output}")

    print(f"\n--- Sample Entries ---")
    for entry in all_entries[:5]:
        print(f"  {entry}")

    # ---- Step 5 (optional): Pre-cache WAV files ----
    if args.cache_wavs:
        os.makedirs(args.cache_dir, exist_ok=True)
        wav_paths = sorted(set(e.split('|')[0] for e in all_entries))
        print(f"\n=== Caching {len(wav_paths)} WAV files to {args.cache_dir} ({n_jobs} workers) ===")
        t0 = time.time()

        cache_fn = partial(_cache_single_wav, cache_dir=args.cache_dir)
        if n_jobs <= 1:
            cache_results = [cache_fn(p) for p in wav_paths]
        else:
            with Pool(n_jobs) as pool:
                cache_results = pool.map(cache_fn, wav_paths, chunksize=max(1, len(wav_paths) // (n_jobs * 4)))

        cached_ok = sum(1 for _, ok, _ in cache_results if ok)
        cached_fail = sum(1 for _, ok, _ in cache_results if not ok)
        elapsed = time.time() - t0
        print(f"  Cached: {cached_ok}, Failed: {cached_fail}, Time: {elapsed:.1f}s")
        for path, ok, err in cache_results:
            if not ok:
                print(f"    FAILED: {path}: {err}")

    # ---- Step 6 (optional): Pre-extract mel spectrograms ----
    if args.extract_mels:
        os.makedirs(args.mel_cache_dir, exist_ok=True)
        wav_cache_dir = args.cache_dir if args.cache_wavs else os.path.join(args.data_dir, 'wave_cache')
        wav_paths = sorted(set(e.split('|')[0] for e in all_entries))
        print(f"\n=== Extracting mel spectrograms for {len(wav_paths)} files ({n_jobs} workers) ===")
        t0 = time.time()

        mel_fn = partial(_extract_mel_single, cache_dir=wav_cache_dir, mel_cache_dir=args.mel_cache_dir)
        if n_jobs <= 1:
            mel_results = [mel_fn(p) for p in wav_paths]
        else:
            with Pool(n_jobs) as pool:
                mel_results = pool.map(mel_fn, wav_paths, chunksize=max(1, len(wav_paths) // (n_jobs * 4)))

        mel_ok = sum(1 for _, ok, _ in mel_results if ok)
        mel_fail = sum(1 for _, ok, _ in mel_results if not ok)
        elapsed = time.time() - t0
        print(f"  Extracted: {mel_ok}, Failed: {mel_fail}, Time: {elapsed:.1f}s")
        for path, ok, err in mel_results:
            if not ok:
                print(f"    FAILED: {path}: {err}")

    print(f"\n=== Done ===")


if __name__ == "__main__":
    main()
