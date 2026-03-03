# coding: utf-8
"""Style embedding extraction: reference audio, style DB lookup, representative styles."""

import torch
import torch.nn.functional as F
import librosa

from tsukasa_speech.data.text import TextCleaner
from tsukasa_speech.data.mel import preprocess
from tsukasa_speech.utils.common import length_to_mask
from tsukasa_speech.utils.phonemize.mixed_phon import smart_phonemize


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
