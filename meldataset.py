#coding: utf-8
import os
import os.path as osp
import json
import math
import time
import random
import numpy as np
import random
import soundfile as sf
import librosa
from collections import defaultdict

import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Sampler

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import pandas as pd

_pad = "$"
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

dicts = {}
for i in range(len((symbols))):
    dicts[symbols[i]] = i

class TextCleaner:
    def __init__(self, dummy=None):
        self.word_index_dictionary = dicts
        self._warned = set()
    def __call__(self, text):
        indexes = []
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError:
                if char not in self._warned:
                    self._warned.add(char)
                    logger.warning("Unknown symbol '%s' (U+%04X) in: %s", char, ord(char), text)
        return indexes

np.random.seed(1)
random.seed(1)
SPECT_PARAMS = {
    "n_fft": 2048,
    "win_length": 1200,
    "hop_length": 300
}
MEL_PARAMS = {
    "n_mels": 80,
}

to_mel = torchaudio.transforms.MelSpectrogram(
    
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4

def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

class FilePathDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_list,
                 root_path,
                 sr=24000,
                 data_augmentation=False,
                 validation=False,
                 OOD_data="Data/OOD_texts.txt",
                 min_length=50,
                 cache_dir=None,
                 ):

        spect_params = SPECT_PARAMS
        mel_params = MEL_PARAMS

        _data_list = [l.strip().split('|') for l in data_list]
        # Filter out entries containing characters outside the valid symbol set
        _valid_symbols = set(dicts.keys())
        _orig_len = len(_data_list)
        _data_list = [data for data in _data_list if len(data) >= 2 and all(c in _valid_symbols for c in data[1])]
        _filtered = _orig_len - len(_data_list)
        if _filtered > 0:
            logger.warning("Filtered %d/%d entries with invalid symbols", _filtered, _orig_len)
        self.data_list = [data if len(data) == 3 else (*data, 0) for data in _data_list]
        self.text_cleaner = TextCleaner()
        self.sr = sr

        self.df = pd.DataFrame(self.data_list)

        self.to_melspec = torchaudio.transforms.MelSpectrogram(**MEL_PARAMS)

        self.mean, self.std = -4, 4
        self.data_augmentation = data_augmentation and (not validation)
        self.max_mel_length = 192

        self.min_length = min_length
        with open(OOD_data, 'r', encoding='utf-8') as f:
            tl = f.readlines()
        idx = 1 if '.wav' in tl[0].split('|')[0] else 0
        self.ptexts = [t.split('|')[idx] for t in tl]

        self.root_path = root_path

        self.cache_dir = cache_dir or os.environ.get('TSUKASA_CACHE_DIR')
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            logger.info("Wave cache enabled: %s", self.cache_dir)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):        
        data = self.data_list[idx]
        path = data[0]
        
        wave, text_tensor, speaker_id = self._load_tensor(data)
        
        mel_tensor = preprocess(wave).squeeze()
        
        acoustic_feature = mel_tensor.squeeze()
        length_feature = acoustic_feature.size(1)
        acoustic_feature = acoustic_feature[:, :(length_feature - length_feature % 2)]
        
        # get reference sample
        ref_data = (self.df[self.df[2] == str(speaker_id)]).sample(n=1).iloc[0].tolist()
        ref_mel_tensor, ref_label = self._load_data(ref_data[:3])
        
        # get OOD text
        
        ps = ""
        
        while len(ps) < self.min_length:
            rand_idx = np.random.randint(0, len(self.ptexts) - 1)
            ps = self.ptexts[rand_idx]
            
            text = self.text_cleaner(ps)
            text.insert(0, 0)
            text.append(0)

            ref_text = torch.LongTensor(text)
        
        return speaker_id, acoustic_feature, text_tensor, ref_text, ref_mel_tensor, ref_label, path, wave

    def _load_tensor(self, data):
        wave_path, text, speaker_id = data
        speaker_id = int(speaker_id)
        wave = self._load_wave(wave_path)

        text = self.text_cleaner(text)
        text.insert(0, 0)
        text.append(0)
        text = torch.LongTensor(text)

        return wave, text, speaker_id

    def _load_wave(self, wave_path):
        if self.cache_dir:
            cache_key = wave_path.replace(os.sep, '_').replace('/', '_')
            cache_file = osp.join(self.cache_dir, cache_key + '.npy')
            if osp.exists(cache_file):
                return np.load(cache_file)
            wave = self._read_wav(wave_path)
            np.save(cache_file, wave)
            return wave
        return self._read_wav(wave_path)

    def _read_wav(self, wave_path):
        wave, sr = sf.read(osp.join(self.root_path, wave_path))
        if wave.shape[-1] == 2:
            wave = wave[:, 0].squeeze()
        if sr != 24000:
            wave = librosa.resample(wave, orig_sr=sr, target_sr=24000)
            print(wave_path, sr)
        wave = np.concatenate([np.zeros([5000]), wave, np.zeros([5000])], axis=0)
        return wave

    def _load_data(self, data):
        wave, text_tensor, speaker_id = self._load_tensor(data)
        mel_tensor = preprocess(wave).squeeze()

        mel_length = mel_tensor.size(1)
        if mel_length > self.max_mel_length:
            random_start = np.random.randint(0, mel_length - self.max_mel_length)
            mel_tensor = mel_tensor[:, random_start:random_start + self.max_mel_length]

        return mel_tensor, speaker_id


class Collater(object):
    """
    Args:
      adaptive_batch_size (bool): if true, decrease batch size when long data comes.
    """

    def __init__(self, return_wave=False):
        self.text_pad_index = 0
        self.min_mel_length = 192
        self.max_mel_length = 192
        self.return_wave = return_wave
        

    def __call__(self, batch):
        # batch[0] = wave, mel, text, f0, speakerid
        batch_size = len(batch)

        # sort by mel length
        lengths = [b[1].shape[1] for b in batch]
        batch_indexes = np.argsort(lengths)[::-1]
        batch = [batch[bid] for bid in batch_indexes]

        nmels = batch[0][1].size(0)
        max_mel_length = max([b[1].shape[1] for b in batch])
        max_text_length = max([b[2].shape[0] for b in batch])
        max_rtext_length = max([b[3].shape[0] for b in batch])

        labels = torch.zeros((batch_size)).long()
        mels = torch.zeros((batch_size, nmels, max_mel_length)).float()
        texts = torch.zeros((batch_size, max_text_length)).long()
        ref_texts = torch.zeros((batch_size, max_rtext_length)).long()

        input_lengths = torch.zeros(batch_size).long()
        ref_lengths = torch.zeros(batch_size).long()
        output_lengths = torch.zeros(batch_size).long()
        ref_mels = torch.zeros((batch_size, nmels, self.max_mel_length)).float()
        ref_labels = torch.zeros((batch_size)).long()
        paths = ['' for _ in range(batch_size)]
        waves = [None for _ in range(batch_size)]
        
        for bid, (label, mel, text, ref_text, ref_mel, ref_label, path, wave) in enumerate(batch):
            mel_size = mel.size(1)
            text_size = text.size(0)
            rtext_size = ref_text.size(0)
            labels[bid] = label
            mels[bid, :, :mel_size] = mel
            texts[bid, :text_size] = text
            ref_texts[bid, :rtext_size] = ref_text
            input_lengths[bid] = text_size
            ref_lengths[bid] = rtext_size
            output_lengths[bid] = mel_size
            paths[bid] = path
            ref_mel_size = ref_mel.size(1)
            ref_mels[bid, :, :ref_mel_size] = ref_mel
            
            ref_labels[bid] = ref_label
            waves[bid] = torch.from_numpy(wave).float() if isinstance(wave, np.ndarray) else wave

        return waves, texts, input_lengths, ref_texts, ref_lengths, mels, output_lengths, ref_mels



class SpeakerBalancedBatchSampler(Sampler):
    """Round-robin speaker-balanced batch sampler with per-speaker ring buffers.

    Each speaker maintains a shuffled pool of sample indices. Batches are filled
    by cycling through speakers in round-robin order. When a speaker's pool is
    exhausted, it wraps around (ring buffer) and reshuffles.

    Example with batch_size=2, 5 speakers (A-E):
        [A_01, B_01], [C_01, D_01], [E_01, A_02], [B_02, C_02], ...
    """

    def __init__(self, data_list, batch_size, seed=42, drop_last=True):
        self.batch_size = batch_size
        self.epoch = 0
        self.seed = seed
        self.drop_last = drop_last

        # Group dataset indices by speaker ID
        self.speaker_to_indices = defaultdict(list)
        for idx, item in enumerate(data_list):
            speaker_id = str(item[2])
            self.speaker_to_indices[speaker_id].append(idx)

        self.speaker_ids = sorted(self.speaker_to_indices.keys())
        self.num_speakers = len(self.speaker_ids)
        self.total_samples = len(data_list)

        if self.num_speakers == 0:
            raise ValueError("No speakers found in data_list")

        # Log per-speaker statistics
        max_count = max(len(v) for v in self.speaker_to_indices.values())
        min_count = min(len(v) for v in self.speaker_to_indices.values())
        logger.info(
            "SpeakerBalancedBatchSampler: %d speakers, batch_size=%d, "
            "samples/speaker: min=%d, max=%d, total=%d",
            self.num_speakers, self.batch_size, min_count, max_count, self.total_samples,
        )
        draws_per_speaker = self.total_samples / self.num_speakers
        for sid in self.speaker_ids:
            count = len(self.speaker_to_indices[sid])
            ratio = draws_per_speaker / count
            logger.info("  Speaker %s: %d samples (oversample x%.2f)", sid, count, ratio)

    def set_epoch(self, epoch):
        """Set epoch for reproducible per-epoch shuffling."""
        self.epoch = epoch

    def __len__(self):
        # Keep the same epoch length as standard sampling
        if self.drop_last:
            return self.total_samples // self.batch_size
        return math.ceil(self.total_samples / self.batch_size)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)

        # Build per-speaker shuffled pools
        pools = {}
        positions = {}
        for sid in self.speaker_ids:
            indices = list(self.speaker_to_indices[sid])
            rng.shuffle(indices)
            pools[sid] = indices
            positions[sid] = 0

        speaker_ptr = 0
        num_batches = len(self)

        for _ in range(num_batches):
            batch = []
            for _ in range(self.batch_size):
                sid = self.speaker_ids[speaker_ptr % self.num_speakers]

                # Ring buffer: wrap around and reshuffle when exhausted
                if positions[sid] >= len(pools[sid]):
                    rng.shuffle(pools[sid])
                    positions[sid] = 0

                batch.append(pools[sid][positions[sid]])
                positions[sid] += 1
                speaker_ptr += 1

            yield batch


class LengthBucketBatchSampler(Sampler):
    """Dynamic batch sampler with length-based bucketing.

    Organizes samples into N buckets by mel-spectrogram length (percentile boundaries).
    Each bucket gets a dynamic batch size inversely proportional to its average
    sample length, targeting constant total mel frames per batch.

    Benefits:
    - Consistent VRAM usage across batches (no spikes from all-long batches)
    - Minimal padding waste (similar lengths within each batch)
    - Speaker balance maintained within each bucket via round-robin
    - Batch order shuffled across buckets to avoid length patterns

    Example with num_buckets=4, base_batch_size=4:
        Bucket 0 (short):  avg=200 frames, batch_size=8
        Bucket 1 (medium): avg=400 frames, batch_size=4
        Bucket 2 (long):   avg=600 frames, batch_size=3
        Bucket 3 (longest): avg=1000 frames, batch_size=2
    """

    def __init__(self, data_list, root_path, base_batch_size,
                 num_buckets=4, seed=42, drop_last=True,
                 max_batch_size=None, min_batch_size=2,
                 speaker_balanced=True, length_cache_path=None):
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        self.num_buckets = num_buckets
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.speaker_balanced = speaker_balanced
        self.data_list = data_list

        # Pre-compute mel lengths for all samples
        mel_lengths = self._compute_lengths(data_list, root_path, length_cache_path)
        self.mel_lengths = np.array(mel_lengths)

        # Create buckets by percentile boundaries
        percentiles = np.linspace(0, 100, num_buckets + 1)[1:-1]  # e.g. [25, 50, 75]
        boundaries = np.percentile(self.mel_lengths, percentiles)
        bucket_ids = np.digitize(self.mel_lengths, boundaries)  # 0..num_buckets-1

        self.buckets = [[] for _ in range(num_buckets)]
        for idx, bid in enumerate(bucket_ids):
            self.buckets[bid].append(idx)

        # Remove empty buckets (can happen with degenerate length distributions)
        self.buckets = [b for b in self.buckets if len(b) > 0]
        self.num_buckets = len(self.buckets)

        # Target tokens = base_batch_size × median length
        median_length = float(np.median(self.mel_lengths))
        self.target_tokens = base_batch_size * median_length

        # Dynamic batch size per bucket
        self.bucket_batch_sizes = []
        for bucket in self.buckets:
            avg_len = float(np.mean(self.mel_lengths[bucket]))
            bs = max(self.min_batch_size, int(round(self.target_tokens / avg_len)))
            if self.max_batch_size:
                bs = min(bs, self.max_batch_size)
            self.bucket_batch_sizes.append(bs)

        # Per-bucket speaker grouping for round-robin
        if speaker_balanced:
            self.bucket_speaker_map = []
            for bucket in self.buckets:
                smap = defaultdict(list)
                for idx in bucket:
                    sid = str(data_list[idx][2])
                    smap[sid].append(idx)
                self.bucket_speaker_map.append(smap)

        # Compute total batches
        self._total_batches = 0
        for bucket, bs in zip(self.buckets, self.bucket_batch_sizes):
            if drop_last:
                self._total_batches += len(bucket) // bs
            else:
                self._total_batches += math.ceil(len(bucket) / bs)

        # Log statistics
        total_samples = len(data_list)
        logger.info(
            "LengthBucketBatchSampler: %d samples -> %d buckets, "
            "target_tokens=%d, total_batches=%d",
            total_samples, self.num_buckets,
            int(self.target_tokens), self._total_batches)
        for b in range(self.num_buckets):
            lengths = self.mel_lengths[self.buckets[b]]
            est_tokens = int(np.mean(lengths) * self.bucket_batch_sizes[b])
            logger.info(
                "  Bucket %d: %d samples (%.1f%%), mel_len=[%d..%d] avg=%.0f, "
                "batch_size=%d, est_tokens/batch=%d",
                b, len(self.buckets[b]),
                100 * len(self.buckets[b]) / total_samples,
                int(lengths.min()), int(lengths.max()), np.mean(lengths),
                self.bucket_batch_sizes[b], est_tokens)

    @staticmethod
    def _compute_lengths(data_list, root_path, cache_path):
        """Pre-compute mel frame counts using audio file headers (fast, no decoding)."""
        if cache_path and osp.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    cached = json.load(f)
                if len(cached) == len(data_list):
                    logger.info("Loaded mel length cache (%d entries) from %s",
                                len(cached), cache_path)
                    return cached
                logger.warning("Length cache size mismatch (%d vs %d), recomputing",
                              len(cached), len(data_list))
            except (json.JSONDecodeError, IOError):
                pass

        logger.info("Scanning audio lengths for %d files...", len(data_list))
        hop_length = 300
        padding_samples = 10000  # 5000 start + 5000 end silence
        mel_lengths = []

        for idx, item in enumerate(data_list):
            wav_path = item[0]
            full_path = osp.join(root_path, wav_path) if root_path else wav_path
            try:
                info = sf.info(full_path)
                frames = info.frames
                if info.samplerate != 24000:
                    frames = int(frames * 24000 / info.samplerate)
                mel_len = (frames + padding_samples) // hop_length
                mel_len -= mel_len % 2  # even length
            except Exception as e:
                if idx < 5:
                    logger.warning("Cannot read %s: %s", wav_path, e)
                mel_len = 500
            mel_lengths.append(mel_len)

            if (idx + 1) % 50000 == 0:
                logger.info("  ...scanned %d / %d", idx + 1, len(data_list))

        logger.info("Length scan complete: min=%d, max=%d, median=%.0f mel frames",
                     min(mel_lengths), max(mel_lengths), np.median(mel_lengths))

        if cache_path:
            try:
                cache_dir = osp.dirname(cache_path)
                if cache_dir:
                    os.makedirs(cache_dir, exist_ok=True)
                with open(cache_path, 'w') as f:
                    json.dump(mel_lengths, f)
                logger.info("Saved mel length cache to %s", cache_path)
            except IOError as e:
                logger.warning("Failed to save length cache: %s", e)

        return mel_lengths

    def set_epoch(self, epoch):
        """Set epoch for reproducible per-epoch shuffling."""
        self.epoch = epoch

    def __len__(self):
        return self._total_batches

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        all_batches = []

        for b in range(self.num_buckets):
            bs = self.bucket_batch_sizes[b]

            if self.speaker_balanced:
                # Round-robin through speakers within this bucket
                speaker_map = self.bucket_speaker_map[b]
                speaker_ids = list(speaker_map.keys())
                rng.shuffle(speaker_ids)

                pools = {}
                positions = {}
                for sid in speaker_ids:
                    pool = list(speaker_map[sid])
                    rng.shuffle(pool)
                    pools[sid] = pool
                    positions[sid] = 0

                ordered = []
                ptr = 0
                for _ in range(len(self.buckets[b])):
                    sid = speaker_ids[ptr % len(speaker_ids)]
                    if positions[sid] >= len(pools[sid]):
                        rng.shuffle(pools[sid])
                        positions[sid] = 0
                    ordered.append(pools[sid][positions[sid]])
                    positions[sid] += 1
                    ptr += 1
            else:
                ordered = list(self.buckets[b])
                rng.shuffle(ordered)

            # Create batches from ordered samples
            for start in range(0, len(ordered), bs):
                batch = ordered[start:start + bs]
                if self.drop_last and len(batch) < bs:
                    break
                all_batches.append(batch)

        # Shuffle batch order across all buckets
        rng.shuffle(all_batches)

        for batch in all_batches:
            yield batch


def build_dataloader(path_list,
                     root_path,
                     validation=False,
                     OOD_data="Data/OOD_texts.txt",
                     min_length=50,
                     batch_size=4,
                     num_workers=1,
                     device='cpu',
                     collate_config={},
                     dataset_config={},
                     persistent_workers=False,
                     prefetch_factor=None,
                     speaker_balanced=False,
                     length_bucket=False,
                     num_buckets=4,
                     max_batch_size=None,
                     min_batch_size=2,
                     length_cache_path=None):

    dataset = FilePathDataset(path_list, root_path, OOD_data=OOD_data, min_length=min_length, validation=validation, **dataset_config)
    collate_fn = Collater(**collate_config)

    pw = persistent_workers if num_workers > 0 else False
    pf = prefetch_factor if num_workers > 0 else None

    if length_bucket and not validation:
        batch_sampler = LengthBucketBatchSampler(
            dataset.data_list, root_path,
            base_batch_size=batch_size,
            num_buckets=num_buckets,
            speaker_balanced=speaker_balanced,
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size,
            length_cache_path=length_cache_path,
            drop_last=True)
        data_loader = DataLoader(dataset,
                                 batch_sampler=batch_sampler,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn,
                                 pin_memory=(device != 'cpu'),
                                 persistent_workers=pw,
                                 prefetch_factor=pf)
    elif speaker_balanced and not validation:
        batch_sampler = SpeakerBalancedBatchSampler(
            dataset.data_list, batch_size=batch_size, drop_last=True)
        data_loader = DataLoader(dataset,
                                 batch_sampler=batch_sampler,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn,
                                 pin_memory=(device != 'cpu'),
                                 persistent_workers=pw,
                                 prefetch_factor=pf)
    else:
        data_loader = DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=(not validation),
                                 num_workers=num_workers,
                                 drop_last=True,
                                 collate_fn=collate_fn,
                                 pin_memory=(device != 'cpu'),
                                 persistent_workers=pw,
                                 prefetch_factor=pf)

    return data_loader

