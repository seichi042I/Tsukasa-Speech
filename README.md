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

# Tsukasa 司 Speech

Japanese TTS model with training pipeline, Gradio UI, and Docker support.

日本語版 README は [README_JP.md](README_JP.md)。

Original model / demo: [Respair/Tsukasa_Speech](https://huggingface.co/Respair/Tsukasa_Speech)

---

## What is Tsukasa Speech?

A Japanese text-to-speech model built on [StyleTTS 2](https://github.com/yl4579/StyleTTS2) with:

- **mLSTM (xLSTM)** layers instead of standard LSTM in the text/prosody encoders
- **Whisper Large v2** encoder as the SLM discriminator (replaces WavLM)
- **PL-BERT, F0 extractor, and text aligner** retrained from scratch for Japanese
- **ISTFTNet** decoder at 24 kHz
- **Promptable style transfer** via reference audio or style database lookup
- **Smart mixed phonemizer** supporting Japanese + Romaji input

---

## Features

### Gradio UI

Two-tab web interface (`http://localhost:7860`):

- **Inference tab** — text-to-speech synthesis with reference audio, style DB lookup, pitch editor, and sentence splitting
- **Training tab** — data preprocessing, Stage 1 / Stage 2 training, and progress monitoring from the browser

### Training Pipeline

Two-stage training with automatic GPU detection:

1. **Stage 1** — acoustic pre-training (text encoder, style encoder, decoder)
2. **Stage 2** — joint fine-tuning with SLM adversarial loss and diffusion decoder

### Style System

- **Reference audio** — extract style from any WAV file
- **Style database** — precomputed per-speaker style vectors for text-similarity lookup
- **Pitch editor** — interactive F0 curve editing before final synthesis

---

## Quick Start

### Docker (recommended)

```bash
docker compose up
```

Open `http://localhost:7860` for the Gradio UI. Place your training data in `./Data/` (auto-mounted).

To provide model weights, either:
- Mount local files: place `Models/` and `Utils/` weight files in the repo root
- Set `MODEL_REPO` env var for automatic HuggingFace download

### RunPod

```bash
# Build and push
docker build -t your-dockerhub/tsukasa-speech .
docker push your-dockerhub/tsukasa-speech
```

In the RunPod UI:

| Setting | Value |
|---|---|
| Container Image | `your-dockerhub/tsukasa-speech` |
| Volume Mount | `/runpod-volume` (network volume with `Data/` directory) |
| HTTP Port | `7860` (Gradio UI), `6006` (TensorBoard) |
| `MODEL_REPO` env | HuggingFace repo ID (e.g. `Respair/Tsukasa_Speech`) |
| `HF_TOKEN` env | (optional) for private repos |

### Local (conda / venv)

```bash
pip install -r requirements.txt
python -m tsukasa_speech.app.main
```

Open `http://127.0.0.1:7860`.

---

## Data Format

```
Data/
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

Fields: `filename:japanese_text:reading` (the reading column is optional).

---

## GPU Tiers

Auto-detected at startup based on the largest single GPU's VRAM:

| Tier | VRAM | batch_size | max_len | SLM |
|---|---|---|---|---|
| `low` | < 24 GB | 2 | 400 | disabled |
| `mid` | 24 – 48 GB | 2 | 800 | enabled |
| `high` | >= 48 GB | 8 | 1600 | enabled |

Override with `GPU_TIER_OVERRIDE=mid` env var.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `STAGE` | — | `1` / `2` / `all` / `shell` (debug shell) |
| `GPU_TIER_OVERRIDE` | (auto-detect) | `low` / `mid` / `high` |
| `DATA_DIR` | `Data` | Path to data directory |
| `N_JOBS` | `4` | Parallel workers for preprocessing |
| `MODEL_REPO` | — | HuggingFace repo ID for model weight download |
| `HF_TOKEN` | — | HuggingFace token for private repos |
| `GRADIO_PORT` | `7860` | Port for the Gradio UI |
| `TSUKASA_CACHE_DIR` | `/tmp/wave_cache` | WAV cache directory |

You can also place a `Data/run_config.yaml` file:

```yaml
stage: all
val_ratio: 0.1
max_duration: 15.0

stage1:
  epochs: 100
stage2:
  epochs: 50
```

---

## CLI Reference

| Purpose | Command |
|---|---|
| Gradio UI | `python -m tsukasa_speech.app.main` |
| Training pipeline | `python -m tsukasa_speech.training --data-dir Data --stage all` |
| Preprocessing | `python -m tsukasa_speech.preprocessing.phonemize_data --data-dir Data` |
| Model download | `python -m tsukasa_speech.utils.download` |
| GPU detection | `python -m tsukasa_speech.config.gpu` |

---

## Repository Structure

```
.
├── tsukasa_speech/             # Main Python package
│   ├── app/                    # Gradio UI (inference + training tabs)
│   ├── config/                 # GPU detection, config merge
│   ├── data/                   # Text processing, mel spectrograms, DataLoader
│   ├── diffusion/              # Diffusion model, sampler
│   ├── inference/              # Model loader, style extraction, TTS core
│   ├── models/                 # Model architecture, builder
│   ├── preprocessing/          # Phonemize, style DB construction
│   ├── training/               # Two-stage training pipeline
│   ├── utils/                  # ASR, JDC, PLBERT, phonemize
│   └── vocoder/                # ISTFTNet, HiFi-GAN
├── Configs/                    # GPU tier YAML configs
│   ├── config_low_vram.yml
│   ├── config_mid_vram.yml
│   └── config_high_vram.yml
├── Utils/                      # Model weight files only
├── train_first.py              # Stage 1 shim (for accelerate launch)
├── finetune_accelerate.py      # Stage 2 shim (for accelerate launch)
├── train.sh                    # Headless training script
├── entrypoint.sh               # Docker entrypoint (launches Gradio UI)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Model Weights

Downloaded automatically from `MODEL_REPO` at first startup:

| File | Size | Description |
|---|---|---|
| `Models/Style_Tsukasa_v02/Top_ckpt_24khz.pth` | ~2.0 GB | Pretrained Tsukasa checkpoint |
| `Utils/ASR/bst_00080.pth` | ~91 MB | Text aligner (ASR) |
| `Utils/JDC/bst.t7` | ~21 MB | F0 pitch extractor |
| `Utils/PLBERT/step_1050000.t7` | ~1.8 GB | PL-BERT |

To skip download, mount them directly:

```bash
docker run ... -v /path/to/Models:/app/Models -v /path/to/Utils:/app/Utils
```

---

## Python API

```python
from tsukasa_speech.inference.model_loader import load_inference_model
from tsukasa_speech.inference.style import compute_ref_style
from tsukasa_speech.inference.core import synthesize

# Load model
model, model_params = load_inference_model("Models/Style_Tsukasa_v02")

# Extract style from reference audio
ref_ss, ref_sp = compute_ref_style(model, "reference.wav")

# Synthesize
wav = synthesize(model, model_params, "こんにちは", ref_ss, ref_sp)
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
