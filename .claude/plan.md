# 実装計画: シングルマウント + マルチ話者対応

## 概要
クラウド環境の制約（バインドマウント1つのみ）に対応し、`Data/` ディレクトリだけで全ての入出力を制御する。
同時にマルチ話者学習への拡張を行う。

---

## 新しい Data/ ディレクトリ構造

```
Data/
    speaker_name1/
        wav/
            audio_001.wav
            audio_002.wav
        transcript_utf8.txt
    speaker_name2/
        wav/
            audio_001.wav
        transcript_utf8.txt
    run_config.yaml          # ユーザー設定（epochs, batch_size等）
    output/                  # チェックポイント出力（自動生成）
        epoch_1st_00100.pth
        epoch_2nd_00050.pth
        first_stage.pth
        tensorboard/
        train.log
    train_list.txt           # preprocess_data.pyが生成
    val_list.txt             # preprocess_data.pyが生成
```

---

## コンフィグ戦略

### 問題
現在のConfig YAMLには**不変パラメータ**（モデルアーキテクチャ、pretrained パス、loss設定）と
**ユーザー調整パラメータ**（epochs、batch_size、max_len）が混在。
Configs/を外部マウントしたくない。

### 解決策: ベースコンフィグ + ユーザーオーバーライド
1. **ベースコンフィグ** (`Configs/base_stage1.yml`, `Configs/base_stage2.yml`)
   - モデルアーキテクチャ、pretrained パス、loss/optimizer デフォルト値
   - Dockerイメージに内蔵（不変）
2. **ユーザーコンフィグ** (`Data/run_config.yaml`)
   - ユーザーが変更したいパラメータのみ記載
   - 存在しなくてもデフォルト値で動作
3. **マージスクリプト** (`merge_config.py`)
   - ベース + ユーザーオーバーライドを深いマージ
   - `log_dir`, `data_params` のパスを自動設定
   - 結果を `/tmp/config_stage1.yml`, `/tmp/config_stage2.yml` に出力

### run_config.yaml のフォーマット（ユーザーが編集するファイル）
```yaml
# ステージ選択: "1", "2", "all", "shell"
stage: all

# 前処理
val_ratio: 0.1

# Stage 1 (Acoustic Pre-Training)
stage1:
  epochs: 100
  batch_size: 4
  max_len: 400
  save_freq: 2

# Stage 2 (Joint Fine-Tuning)
stage2:
  epochs: 50
  batch_size: 2
  max_len: 200
  save_freq: 1

# 上級者向け: loss_params, optimizer_params 等も
# ベースコンフィグの任意のキーをオーバーライド可能
# 例:
# loss_params:
#   lambda_mel: 10.
#   joint_epoch: 999
```

---

## 変更ファイル一覧

### 1. `Configs/base_stage1.yml` — 新規作成
Stage 1 のベースコンフィグ。config_iori.yml から作成。
- `log_dir: "Data/output"`（Data内に出力）
- `data_params.train_data: "Data/train_list.txt"`
- `data_params.val_data: "Data/val_list.txt"`
- `pretrained_model: "Models/Style_Tsukasa_v02/Top_ckpt_24khz.pth"`（イメージ内）
- `data_params.OOD_data: "OOD_LargeScale_.csv"`（イメージ内）
- その他モデルアーキテクチャ、loss、optimizer パラメータはconfig_iori.yml と同一

### 2. `Configs/base_stage2.yml` — 新規作成
Stage 2 のベースコンフィグ。config_iori_ft.yml から作成。
- `log_dir: "Data/output"`
- 同様のパス変更
- loss_params, optimizer_params は config_iori_ft.yml と同一

### 3. `merge_config.py` — 新規作成
```python
# 機能:
# 1. ベースコンフィグYAML読込
# 2. Data/run_config.yaml読込（存在しない場合はデフォルト使用）
# 3. stage1/stage2 キーの内容をトップレベルにフラット展開してマージ
# 4. multispeakerフラグを自動設定（話者数 > 1 ならtrue）
# 5. マージ済みコンフィグを指定パスに出力

# マージルール:
# - run_config.yaml の stage1.epochs → base.epochs_1st (Stage 1)
#   または base.epochs (Stage 2)
# - run_config.yaml の stage1.batch_size → base.batch_size
# - それ以外のトップレベルキー（loss_params等）はそのまま深いマージ
# - data_params, pretrained paths 等は常にベースの値を使用

# CLI: python merge_config.py --base BASE.yml --run-config DATA/run_config.yaml --output /tmp/config.yml --stage 1|2 --num-speakers N
```

### 4. `preprocess_data.py` — 書き換え（マルチ話者対応）
```
変更内容:
- Data/ 直下のサブディレクトリをスキャンし、wav/ と transcript_utf8.txt を持つものを話者として検出
- ディレクトリ名のソート順で speaker_id (0, 1, 2, ...) を割り当て
- 各話者のトランスクリプトを処理し、パスは Data/speaker_name/wav/filename.wav 形式
- 全話者のエントリを統合して train_list.txt / val_list.txt を出力
- 話者ごとの統計情報をログ出力
- --data-dir引数で Data/ のパスを指定可能（デフォルト: Data）

出力例 (train_list.txt):
  Data/iori/wav/IORI_0001.wav|tsɯki no takaɽa|0
  Data/iori/wav/IORI_0002.wav|ohajoː gozaimasɯ|0
  Data/sana/wav/SANA_0001.wav|koɴniʨiwa|1

エラーハンドリング:
- 話者ディレクトリが0個: エラー終了
- wav/ が空: 警告出してスキップ
- transcript_utf8.txt が無い: 警告出してスキップ
```

### 5. `finetune_accelerate.py` — 編集（1行削除）
- 285行目の `torch.cuda.empty_cache()` を削除

### 6. `meldataset.py` — 変更なし
WAVキャッシュ機構は既に実装済み。
パスの扱い（`root_path` + `wave_path`）も現在のままで動作する。
（root_path="" のまま、wave_path が "Data/speaker/wav/file.wav" となるため）

### 7. `Dockerfile` — 書き換え
```dockerfile
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

# Layer 1: System deps (変更なし)
# Layer 2: pip install (requirements.txt変更時のみ)
# Layer 3: Pretrained models + Utils + Modules + OOD CSV (~4GB、ほぼ不変)
#           + Configs/base_stage1.yml, Configs/base_stage2.yml ← 追加
# Layer 4: *.py + entrypoint.sh (コード変更のみ)

# 変更点:
# - COPY Configs/base_stage1.yml Configs/base_stage2.yml Configs/ を追加
# - Models/IORI_Finetuned のmkdirは不要（Data/output/を使うため）
# - /tmp/wave_cache のmkdirは維持
```

### 8. `docker-compose.yml` — 書き換え
```yaml
x-gpu-common: &gpu-common
  build: .
  shm_size: '8g'
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
  environment: &env-common
    HF_HOME: /root/.cache/huggingface
    TSUKASA_CACHE_DIR: /tmp/wave_cache
  volumes:
    - ./Data:/app/Data              # 唯一のバインドマウント
    - hf_cache:/root/.cache/huggingface  # named volume (ローカル)

services:
  train:
    <<: *gpu-common
    environment:
      <<: *env-common
      CUDA_VISIBLE_DEVICES: 0
      # STAGEはData/run_config.yamlから読む。環境変数での上書きも可能。

  shell:
    <<: *gpu-common
    environment:
      <<: *env-common
      STAGE: shell
    stdin_open: true
    tty: true

volumes:
  hf_cache:
```

**削除されるマウント:**
- `./Configs:/app/Configs:ro` → ベースコンフィグはイメージ内蔵
- `./Models/IORI_Finetuned:/app/Models/IORI_Finetuned` → `Data/output/` に統合

**削除される環境変数:**
- `STAGE`, `CONFIG_STAGE1`, `CONFIG_STAGE2` → `Data/run_config.yaml` で制御
  （STAGE環境変数は run_config.yaml のフォールバックとして残す）

### 9. `entrypoint.sh` — 書き換え
```
パイプライン:
  1. Data/run_config.yaml からSTAGE読み取り（環境変数STAGEでオーバーライド可）
  2. コンフィグマージ: merge_config.py でベース + run_config → /tmp/config_stage{1,2}.yml
  3. 話者自動検出 + 前処理: train_list.txt が無ければ preprocess_data.py 実行
  4. WAVキャッシュ pre-warm（既存ロジック維持）
  5. ステージ実行

コンフィグマージの呼び出し:
  # 話者数をカウント
  NUM_SPEAKERS=$(話者ディレクトリ数)

  python merge_config.py \
    --base Configs/base_stage1.yml \
    --run-config Data/run_config.yaml \
    --output /tmp/config_stage1.yml \
    --stage 1 \
    --num-speakers $NUM_SPEAKERS

  python merge_config.py \
    --base Configs/base_stage2.yml \
    --run-config Data/run_config.yaml \
    --output /tmp/config_stage2.yml \
    --stage 2 \
    --num-speakers $NUM_SPEAKERS

学習実行:
  accelerate launch train_first.py -p /tmp/config_stage1.yml
  accelerate launch finetune_accelerate.py -p /tmp/config_stage2.yml
```

### 10. `.dockerignore` — 書き換え
```
.git
__pycache__
*.pyc
*.pyo

# ユーザーデータ（ランタイムでマウント）
Data/

# 不要ファイル
Inference/
reference_sample_wavs/
app_tsuka.py
*.md
.claude/
.gitignore
.gitattributes
docker-compose.yml
Dockerfile
.dockerignore

# ユーザー固有コンフィグ（ベースコンフィグのみイメージに含める）
Configs/config.yml
Configs/config_ft.yml
Configs/config_iori.yml
Configs/config_iori_ft.yml
Configs/config_kanade.yml
```

---

## データフロー（変更後）

```
docker compose up train
  ↓
entrypoint.sh
  ├─ Data/run_config.yaml 読込（無ければデフォルト使用）
  ├─ merge_config.py: ベースコンフィグ + ユーザーオーバーライド → /tmp/config_stage{1,2}.yml
  ├─ 話者自動検出: Data/*/wav/ + transcript_utf8.txt を持つディレクトリ列挙
  ├─ preprocess_data.py: 全話者のトランスクリプト → Data/train_list.txt, Data/val_list.txt
  ├─ WAVキャッシュ pre-warm: 全WAV → /tmp/wave_cache/*.npy
  ├─ Stage 1: accelerate launch train_first.py -p /tmp/config_stage1.yml
  │           → チェックポイント: Data/output/epoch_1st_*.pth, Data/output/first_stage.pth
  ├─ Stage 2: accelerate launch finetune_accelerate.py -p /tmp/config_stage2.yml
  │           → チェックポイント: Data/output/epoch_2nd_*.pth
  └─ 完了
```

---

## 後方互換性
- `train_first.py`, `finetune_accelerate.py`: コンフィグ消費ロジックは変更なし（-p で渡されるYAMLの形式は同一）
- `meldataset.py`: パスの扱い変更なし（root_path="" + フルパス）
- `run_config.yaml` が無い場合: 全てベースコンフィグのデフォルト値で動作
- hf_cache named volume: 維持（Dockerローカルストレージ、クラウドマウントではない）

## 削除対象ファイル
- `sha256:ed71...` — 空のDockerアーティファクト（ただし現在のワーキングツリーに存在しないため、操作不要）
