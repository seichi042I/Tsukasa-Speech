---
thumbnail: https://i.postimg.cc/y6gT18Tn/Untitled-design-1.png
license: cc-by-nc-4.0
language:
- ja
pipeline_tag: text-to-speech
tags:
- '#StyleTTS'
- '#Japanese'
- Diffusion
- Prompt
- '#TTS'
- '#TexttoSpeech'
- '#speech'
- '#StyleTTS2'
---

<div style="text-align:center;">
  <img src="https://i.postimg.cc/y6gT18Tn/Untitled-design-1.png" alt="Logo" style="width:300px; height:auto;">
</div>

# Tsukasa 司 Speech — RunPod 学習パイプライン

このリポジトリは Tsukasa Speech の **自己学習・ファインチューニング用パイプライン** です。[RunPod](https://runpod.io) または CUDA 対応ホストで動作する Docker イメージとして提供されます。

英語版 README は [README.md](README.md)。

オリジナルモデル / デモ: [Respair/Tsukasa_Speech](https://huggingface.co/Respair/Tsukasa_Speech)

---

## Tsukasa Speech とは

[StyleTTS 2](https://github.com/yl4579/StyleTTS2) をベースにした日本語 TTS モデルで、以下の変更が加えられています:

- テキスト・プロソディエンコーダーに mLSTM (xLSTM) レイヤーを採用
- SLM discriminator に WavLM ではなく Whisper Large v2 エンコーダーを使用
- PL-BERT、F0 ピッチ抽出器、テキストアライナーをゼロから再学習
- 24 kHz ISTFTNet デコーダー
- プロンプトによるスタイル転送
- 日本語・ローマ字混在テキスト対応のフォネマイザー

---

## クイックスタート (RunPod)

### 1. データの準備

RunPod ネットワークボリュームに学習データを配置します:

```
/runpod-volume/Data/
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

### 2. Docker イメージのビルドとプッシュ

```bash
docker build -t your-dockerhub/tsukasa-speech .
docker push your-dockerhub/tsukasa-speech
```

### 3. RunPod で起動

RunPod UI で以下を設定します:

| 設定 | 値 |
|---|---|
| Container Image | `your-dockerhub/tsukasa-speech` |
| Volume Mount | `/runpod-volume`（`Data/` ディレクトリを含むネットワークボリューム） |
| `MODEL_REPO` 環境変数 | モデル重みの HuggingFace リポジトリ ID（例: `Respair/Tsukasa_Speech`） |
| `HF_TOKEN` 環境変数 | （任意）プライベートリポジトリ用のトークン |

コンテナ起動時に自動で以下の処理が実行されます:
1. HuggingFace からモデル重みをダウンロード（未取得の場合）
2. GPU VRAM を検出して最適な設定 Tier を選択
3. トランスクリプトをフォネマイズして `train_list.txt` / `val_list.txt` を生成
4. ウェーブキャッシュのウォームアップ
5. Stage 1 → Stage 2 学習を実行

### 4. 環境変数による制御

| 変数名 | デフォルト | 説明 |
|---|---|---|
| `STAGE` | `all` | `1` = Stage 1 のみ, `2` = Stage 2 のみ, `all` = 両方, `shell` = デバッグシェル |
| `GPU_TIER_OVERRIDE` | （自動検出） | `low` / `mid` / `high` — GPU 検出を上書き |
| `N_JOBS` | `4` | 前処理の並列ワーカー数 |
| `DATA_DIR` | `Data` | データディレクトリのパス |

またはボリューム上に `Data/run_config.yaml` を配置して設定できます:

```yaml
stage: all          # 1 | 2 | all
val_ratio: 0.1      # 検証データの割合
max_duration: 15.0  # この秒数を超える音声ファイルをスキップ

# ステージごとの上書き設定（任意）
stage1:
  epochs: 100
stage2:
  epochs: 50
```

---

## GPU Tier

コンテナが GPU を自動検出してコンフィグを選択します:

| Tier | GPU VRAM | コンフィグ | batch_size | max_len | SLM |
|---|---|---|---|---|---|
| `low` | 20 GB 未満 | `config_low_vram.yml` | 2 | 200 | 無効 |
| `mid` | 20〜36 GB | `config_mid_vram.yml` | 4 | 600 | 有効 |
| `high` | 36 GB 以上 | `config_high_vram.yml` | 8 | 800 | 有効 |

`GPU_TIER_OVERRIDE=mid`（または `run_config.yaml` の `gpu_tier` キー）で上書き可能です。

---

## モデル重み

起動時に以下のファイルが必要です（`MODEL_REPO` 設定時は自動ダウンロード）:

| ファイル | サイズ | 説明 |
|---|---|---|
| `Models/Style_Tsukasa_v02/Top_ckpt_24khz.pth` | 約 2.0 GB | 事前学習済み Tsukasa チェックポイント |
| `Utils/ASR/bst_00080.pth` | 約 91 MB | テキストアライナー（ASR） |
| `Utils/JDC/bst.t7` | 約 21 MB | F0 ピッチ抽出器 |
| `Utils/PLBERT/step_1050000.t7` | 約 1.8 GB | PL-BERT |

ボリュームマウントでダウンロードをスキップすることもできます:
```bash
docker run ... -v /path/to/Models:/app/Models -v /path/to/Utils:/app/Utils
```

---

## ローカル開発

```bash
# フルパイプライン（GPU 自動検出 → 前処理 → 学習）
docker compose up train

# Stage 1 のみ
docker compose up stage1

# Stage 2 のみ
docker compose up stage2

# デバッグシェル
docker compose run shell
# コンテナ内で 'train' コマンドを実行すると学習開始
```

---

## リポジトリ構成

```
.
├── train.sh                  # メイン学習パイプラインスクリプト
├── entrypoint.sh             # コンテナエントリーポイント
├── Dockerfile
├── docker-compose.yml        # ローカル開発用
│
├── train_first.py            # Stage 1: 音響事前学習
├── finetune_accelerate.py    # Stage 2: ジョイントファインチューニング
├── preprocess_data.py        # フォネマイズ + データ分割
├── detect_gpu.py             # GPU VRAM 検出 → コンフィグ Tier 選択
├── merge_config.py           # ベースコンフィグ + ユーザー上書きのマージ
├── download_models.py        # HuggingFace からモデル重みをダウンロード
│
├── models.py                 # モデルアーキテクチャ
├── meldataset.py             # DataLoader
├── losses.py                 # 損失関数
├── optimizers.py             # オプティマイザービルダー
├── utils.py                  # ユーティリティ
│
├── Configs/
│   ├── config_low_vram.yml   # ~16 GB GPU 用
│   ├── config_mid_vram.yml   # 24〜32 GB GPU 用
│   ├── config_high_vram.yml  # 32 GB+ GPU 用
│   └── reference/            # 参照用コンフィグ（パイプラインでは使用しない）
│       ├── base_stage1.yml
│       └── base_stage2.yml
│
├── OOD_LargeScale_.csv       # 学習用 OOD テキストデータ
├── Utils/                    # ASR, JDC, PLBERT, フォネマイザー
└── Modules/                  # 拡散モデル, SLM Adversarial Loss
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
