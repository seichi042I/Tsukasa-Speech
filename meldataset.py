#coding: utf-8
import os
import os.path as osp
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
    def __call__(self, text):
        indexes = []
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError:
                print(text)
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
                     speaker_balanced=False):

    dataset = FilePathDataset(path_list, root_path, OOD_data=OOD_data, min_length=min_length, validation=validation, **dataset_config)
    collate_fn = Collater(**collate_config)

    pw = persistent_workers if num_workers > 0 else False
    pf = prefetch_factor if num_workers > 0 else None

    if speaker_balanced and not validation:
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

