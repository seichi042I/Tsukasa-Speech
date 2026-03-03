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

日本語 TTS モデル。学習パイプライン・Gradio UI・Docker 対応。

English README: [README.md](README.md)

オリジナルモデル / デモ: [Respair/Tsukasa_Speech](https://huggingface.co/Respair/Tsukasa_Speech)

---

## Tsukasa Speech とは

[StyleTTS 2](https://github.com/yl4579/StyleTTS2) をベースにした日本語テキスト音声合成モデルです:

- テキスト・プロソディエンコーダーに **mLSTM (xLSTM)** レイヤーを採用
- SLM discriminator に **Whisper Large v2** エンコーダーを使用（WavLM を置換）
- **PL-BERT、F0 抽出器、テキストアライナー**を日本語向けにゼロから再学習
- **ISTFTNet** デコーダー（24 kHz）
- リファレンス音声またはスタイル DB による**プロンプトスタイル転送**
- 日本語・ローマ字混在入力対応の**スマートフォネマイザー**

---

## 機能

### Gradio UI

2 タブ構成の Web インターフェース（`http://localhost:7860`）:

- **音声合成タブ** — リファレンス音声、スタイル DB ルックアップ、ピッチエディター、文分割による音声合成
- **学習タブ** — データ前処理、Stage 1 / Stage 2 学習、進捗監視をブラウザから操作

### 学習パイプライン

GPU 自動検出による 2 段階学習:

1. **Stage 1** — 音響事前学習（テキストエンコーダー、スタイルエンコーダー、デコーダー）
2. **Stage 2** — SLM Adversarial Loss と拡散デコーダーによるジョイントファインチューニング

### スタイルシステム

- **リファレンス音声** — 任意の WAV ファイルからスタイル抽出
- **スタイル DB** — 話者ごとに事前計算されたスタイルベクトルによるテキスト類似度ルックアップ
- **ピッチエディター** — 最終合成前の F0 カーブインタラクティブ編集

---

## クイックスタート

### Docker（推奨）

```bash
docker compose up
```

`http://localhost:7860` で Gradio UI にアクセス。学習データは `./Data/` に配置（自動マウント）。

モデル重みの提供方法:
- ローカルマウント: リポジトリルートに `Models/` と `Utils/` の重みファイルを配置
- `MODEL_REPO` 環境変数を設定して HuggingFace から自動ダウンロード

### RunPod

```bash
# ビルド & プッシュ
docker build -t your-dockerhub/tsukasa-speech .
docker push your-dockerhub/tsukasa-speech
```

RunPod UI での設定:

| 設定 | 値 |
|---|---|
| Container Image | `your-dockerhub/tsukasa-speech` |
| Volume Mount | `/runpod-volume`（`Data/` ディレクトリを含むネットワークボリューム） |
| HTTP Port | `7860`（Gradio UI）、`6006`（TensorBoard） |
| `MODEL_REPO` 環境変数 | HuggingFace リポジトリ ID（例: `Respair/Tsukasa_Speech`） |
| `HF_TOKEN` 環境変数 | （任意）プライベートリポジトリ用 |

### ローカル（conda / venv）

```bash
pip install -r requirements.txt
python -m tsukasa_speech.app.main
```

`http://127.0.0.1:7860` を開きます。

---

## データフォーマット

```
Data/
    話者名/
        wav/
            XXXX_0001.wav
            XXXX_0002.wav
            ...
        transcript_utf8.txt
```

`transcript_utf8.txt` のフォーマット（コロン区切り）:

```
XXXX_0001.wav:月の宝…:ツキノタカラ
XXXX_0002.wav:空を飛びたいな:ソラヲトビタイナ
```

フィールド: `ファイル名:日本語テキスト:読み仮名`（読み仮名は省略可）。

---

## GPU Tier

起動時に最大 GPU の VRAM に基づいて自動検出:

| Tier | VRAM | batch_size | max_len | SLM |
|---|---|---|---|---|
| `low` | < 24 GB | 2 | 400 | 無効 |
| `mid` | 24 – 48 GB | 2 | 800 | 有効 |
| `high` | >= 48 GB | 8 | 1600 | 有効 |

`GPU_TIER_OVERRIDE=mid` 環境変数で上書き可能。

---

## 環境変数

| 変数名 | デフォルト | 説明 |
|---|---|---|
| `STAGE` | — | `1` / `2` / `all` / `shell`（デバッグシェル） |
| `GPU_TIER_OVERRIDE` | （自動検出） | `low` / `mid` / `high` |
| `DATA_DIR` | `Data` | データディレクトリのパス |
| `N_JOBS` | `4` | 前処理の並列ワーカー数 |
| `MODEL_REPO` | — | モデル重み自動 DL 用 HuggingFace リポジトリ ID |
| `HF_TOKEN` | — | プライベートリポジトリ用 HuggingFace トークン |
| `GRADIO_PORT` | `7860` | Gradio UI のポート |
| `TSUKASA_CACHE_DIR` | `/tmp/wave_cache` | WAV キャッシュディレクトリ |

`Data/run_config.yaml` ファイルでも設定可能:

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

## CLI リファレンス

| 用途 | コマンド |
|---|---|
| Gradio UI | `python -m tsukasa_speech.app.main` |
| 学習パイプライン | `python -m tsukasa_speech.training --data-dir Data --stage all` |
| 前処理 | `python -m tsukasa_speech.preprocessing.phonemize_data --data-dir Data` |
| モデルダウンロード | `python -m tsukasa_speech.utils.download` |
| GPU 検出 | `python -m tsukasa_speech.config.gpu` |

---

## リポジトリ構成

```
.
├── tsukasa_speech/             # メイン Python パッケージ
│   ├── app/                    # Gradio UI（推論 + 学習タブ）
│   ├── config/                 # GPU 検出、設定マージ
│   ├── data/                   # テキスト処理、メルスペクトログラム、DataLoader
│   ├── diffusion/              # 拡散モデル、サンプラー
│   ├── inference/              # モデルローダー、スタイル抽出、TTS コア
│   ├── models/                 # モデルアーキテクチャ、ビルダー
│   ├── preprocessing/          # フォネマイズ、スタイル DB 構築
│   ├── training/               # 2 段階学習パイプライン
│   ├── utils/                  # ASR, JDC, PLBERT, phonemize
│   └── vocoder/                # ISTFTNet, HiFi-GAN
├── Configs/                    # GPU Tier 別 YAML 設定
│   ├── config_low_vram.yml
│   ├── config_mid_vram.yml
│   └── config_high_vram.yml
├── Utils/                      # モデル重みファイルのみ
├── train_first.py              # Stage 1 shim（accelerate launch 用）
├── finetune_accelerate.py      # Stage 2 shim（accelerate launch 用）
├── train.sh                    # ヘッドレス学習スクリプト
├── entrypoint.sh               # Docker エントリーポイント（Gradio UI 起動）
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## モデル重み

`MODEL_REPO` 設定時は初回起動で自動ダウンロード:

| ファイル | サイズ | 説明 |
|---|---|---|
| `Models/Style_Tsukasa_v02/Top_ckpt_24khz.pth` | 約 2.0 GB | 事前学習済み Tsukasa チェックポイント |
| `Utils/ASR/bst_00080.pth` | 約 91 MB | テキストアライナー（ASR） |
| `Utils/JDC/bst.t7` | 約 21 MB | F0 ピッチ抽出器 |
| `Utils/PLBERT/step_1050000.t7` | 約 1.8 GB | PL-BERT |

ダウンロードをスキップするにはボリュームマウント:

```bash
docker run ... -v /path/to/Models:/app/Models -v /path/to/Utils:/app/Utils
```

---

## Python API

```python
from tsukasa_speech.inference.model_loader import load_inference_model
from tsukasa_speech.inference.style import compute_ref_style
from tsukasa_speech.inference.core import synthesize

# モデル読み込み
model, model_params = load_inference_model("Models/Style_Tsukasa_v02")

# リファレンス音声からスタイル抽出
ref_ss, ref_sp = compute_ref_style(model, "reference.wav")

# 音声合成
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
