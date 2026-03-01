---
thumbnail: https://i.postimg.cc/y6gT18Tn/Untitled-design-1.png
license: cc-by-nc-4.0
language:
- ja
pipeline_tag: text-to-speech
tags:

- 'StyleTTS'
- 'Japanese'
- 'Diffusion'
- 'Prompt'
- 'TTS'
- 'TexttoSpeech'
- 'speech'
- 'StyleTTS2'
- 'LLM'
- 'anime'
- 'voice'

---

<div style="text-align:center;">
  <img src="https://i.postimg.cc/y6gT18Tn/Untitled-design-1.png" alt="Logo" style="width:300px; height:auto;">
</div>

# Tsukasa 司 Speech — RunPod Training Pipeline

This repository is a **self-training/fine-tuning pipeline** for Tsukasa Speech, packaged as a Docker image for use on [RunPod](https://runpod.io) or any CUDA-capable host.

日本語版 README は [README_JP.md](README_JP.md)。

Original model card / demo: [Respair/Tsukasa_Speech](https://huggingface.co/Respair/Tsukasa_Speech)

---

## What is Tsukasa Speech?

A Japanese TTS model built on [StyleTTS 2](https://github.com/yl4579/StyleTTS2) with the following additions:

- mLSTM layers (xLSTM) instead of standard LSTM in the text/prosody encoders
- Whisper Large v2 encoder as the SLM discriminator (instead of WavLM)
- Retrained PL-BERT, F0 pitch extractor, and text aligner from scratch
- ISTFTNet decoder at 24 kHz
- Promptable style transfer
- Smart mixed Japanese/Romaji phonemization

---

## Quick Start (RunPod)

### 1. Prepare data

Place training data on your RunPod network volume:

```
/runpod-volume/Data/
    speaker_name/
        wav/
            XXXX_0001.wav
            XXXX_0002.wav
            ...
        transcript_utf8.txt
```

`transcript_utf8.txt` format (colon-separated):
```
XXXX_0001.wav:月の宝…:ツキノタカラ
XXXX_0002.wav:空を飛びたいな:ソラヲトビタイナ
```
Fields: `filename:japanese_text:reading` (the reading column is optional and can be omitted).

### 2. Build and push the Docker image

```bash
docker build -t your-dockerhub/tsukasa-speech .
docker push your-dockerhub/tsukasa-speech
```

### 3. Launch on RunPod

In the RunPod UI:

| Setting | Value |
|---|---|
| Container Image | `your-dockerhub/tsukasa-speech` |
| Volume Mount | `/runpod-volume` (network volume with your `Data/` directory) |
| `MODEL_REPO` env | HuggingFace repo ID for model weights (e.g. `Respair/Tsukasa_Speech`) |
| `HF_TOKEN` env | (optional) for private repos |

On startup the container will automatically:
1. Download missing model weights from HuggingFace
2. Detect GPU VRAM and select the appropriate config tier
3. Phonemize transcripts and build `train_list.txt` / `val_list.txt`
4. Pre-warm the wave cache
5. Run Stage 1 → Stage 2 training

### 4. Control training via environment variables

| Variable | Default | Description |
|---|---|---|
| `STAGE` | `all` | `1` = Stage 1 only, `2` = Stage 2 only, `all` = both, `shell` = debug shell |
| `GPU_TIER_OVERRIDE` | (auto-detect) | `low` / `mid` / `high` — override automatic GPU detection |
| `N_JOBS` | `4` | Parallel workers for preprocessing |
| `DATA_DIR` | `Data` | Path to data directory |

Or place a `Data/run_config.yaml` on your volume:

```yaml
stage: all          # 1 | 2 | all
val_ratio: 0.1      # validation split fraction
max_duration: 15.0  # skip audio files longer than N seconds

# per-stage overrides (optional)
stage1:
  epochs: 100
stage2:
  epochs: 50
```

---

## GPU Tiers

The container auto-detects your GPU and selects a config:

| Tier | GPU VRAM | Config | batch_size | max_len | SLM |
|---|---|---|---|---|---|
| `low` | < 20 GB | `config_low_vram.yml` | 2 | 200 | disabled |
| `mid` | 20–36 GB | `config_mid_vram.yml` | 4 | 600 | enabled |
| `high` | ≥ 36 GB | `config_high_vram.yml` | 8 | 800 | enabled |

Override with `GPU_TIER_OVERRIDE=mid` (or via `run_config.yaml`).

---

## Model Weights

The following files must be present at startup (downloaded automatically from `MODEL_REPO`):

| File | Size | Description |
|---|---|---|
| `Models/Style_Tsukasa_v02/Top_ckpt_24khz.pth` | ~2.0 GB | Pretrained Tsukasa checkpoint |
| `Utils/ASR/bst_00080.pth` | ~91 MB | Text aligner (ASR) |
| `Utils/JDC/bst.t7` | ~21 MB | F0 pitch extractor |
| `Utils/PLBERT/step_1050000.t7` | ~1.8 GB | PL-BERT |

To skip downloading, mount them directly:
```bash
docker run ... -v /path/to/Models:/app/Models -v /path/to/Utils:/app/Utils
```

---

## Local Development

```bash
# Full pipeline (auto GPU detect → preprocess → train)
docker compose up train

# Stage 1 only
docker compose up stage1

# Stage 2 only
docker compose up stage2

# Debug shell
docker compose run shell
# Inside container: run 'train' to start training
```

---

## Repository Structure

```
.
├── train.sh                  # Main training pipeline script
├── entrypoint.sh             # Container entrypoint
├── Dockerfile
├── docker-compose.yml        # Local development
│
├── train_first.py            # Stage 1: acoustic pre-training
├── finetune_accelerate.py    # Stage 2: joint fine-tuning
├── preprocess_data.py        # Phonemization + data split
├── detect_gpu.py             # GPU VRAM detection → config tier
├── merge_config.py           # Merge base config + user overrides
├── download_models.py        # HuggingFace model weight download
│
├── models.py                 # Model architecture
├── meldataset.py             # DataLoader
├── losses.py                 # Loss functions
├── optimizers.py             # Optimizer builder
├── utils.py                  # Utilities
│
├── Configs/
│   ├── config_low_vram.yml   # ~16 GB GPU
│   ├── config_mid_vram.yml   # 24–32 GB GPU
│   ├── config_high_vram.yml  # 32 GB+ GPU
│   └── reference/            # Reference configs (not used by pipeline)
│       ├── base_stage1.yml
│       └── base_stage2.yml
│
├── OOD_LargeScale_.csv       # OOD text data for training
├── Utils/                    # ASR, JDC, PLBERT, phonemizer
└── Modules/                  # Diffusion, SLM adversarial loss
```

---

## References

- [yl4579/StyleTTS2](https://github.com/yl4579/StyleTTS2)
- [NX-AI/xlstm](https://github.com/NX-AI/xlstm)
- [archinetai/audio-diffusion-pytorch](https://github.com/archinetai/audio-diffusion-pytorch)
- [jik876/hifi-gan](https://github.com/jik876/hifi-gan)
- [rishikksh20/iSTFTNet-pytorch](https://github.com/rishikksh20/iSTFTNet-pytorch)
- [ShoukanLabs/VoPho](https://github.com/ShoukanLabs/VoPho)

```bibtex
@article{xlstm,
  title={xLSTM: Extended Long Short-Term Memory},
  author={Beck, Maximilian and P{\"o}ppel, Korbinian and Spanring, Markus and Auer, Andreas and Prudnikova, Oleksandra and Kopp, Michael and Klambauer, G{\"u}nter and Brandstetter, Johannes and Hochreiter, Sepp},
  journal={arXiv preprint arXiv:2405.04517},
  year={2024}
}
```
