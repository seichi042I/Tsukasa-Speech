"""Gradio Web UI for Tsukasa Speech TTS inference and training."""

import os
import os.path as osp
import re
import signal
import subprocess
import threading
import traceback

import yaml
import torch
import numpy as np
import gradio as gr

from inference import (
    resolve_model_dir,
    normalize_config,
    load_inference_model,
    compute_ref_style,
    lookup_style_from_db,
    load_repr_style,
    synthesize,
)
from Utils.phonemize.mixed_phon import smart_phonemize

# ---------------------------------------------------------------------------
# Model cache
# ---------------------------------------------------------------------------
_current_model = {
    "model": None,
    "model_params": None,
    "style_db_path": None,
    "device": None,
    "sr": 24000,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def detect_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def split_sentences(text):
    """Split text on Japanese sentence-ending punctuation (。) for chunked synthesis.

    Keeps the delimiter attached to each segment.
    Returns a list of non-empty sentence strings.
    """
    parts = re.split(r'(。)', text)
    sentences = []
    buf = ""
    for part in parts:
        buf += part
        if part == '。':
            s = buf.strip()
            if s:
                sentences.append(s)
            buf = ""
    # Trailing text without 。
    s = buf.strip()
    if s:
        sentences.append(s)
    return sentences


def discover_model_dirs():
    """Scan Models/ and Data/*/output/ for valid model directories.

    Returns list of (display_name, dir_path) tuples for Gradio dropdown.
    """
    candidates = []  # (label, path)
    seen_paths = set()

    def _try_add(label, d):
        d_abs = osp.abspath(d)
        if d_abs in seen_paths:
            return
        try:
            resolve_model_dir(d)
            candidates.append((label, d))
            seen_paths.add(d_abs)
        except FileNotFoundError:
            pass

    # Finetuned models: Data/{model_name}/output/
    if osp.isdir("Data"):
        for model_name in sorted(os.listdir("Data")):
            output_dir = osp.join("Data", model_name, "output")
            if not osp.isdir(output_dir):
                continue
            _try_add(model_name, output_dir)

    # Base models: Models/{name}/
    if osp.isdir("Models"):
        for name in sorted(os.listdir("Models")):
            d = osp.join("Models", name)
            if not osp.isdir(d):
                continue
            _try_add(f"{name} (base)", d)

    return candidates


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------

def load_model_handler(model_dir):
    """Load a model from the selected directory."""
    if not model_dir:
        return "モデルディレクトリを選択してください。"

    try:
        config_path, checkpoint_path, style_db_path = resolve_model_dir(model_dir)
        config = yaml.safe_load(open(config_path))
        config = normalize_config(config)
        sr = config["preprocess_params"].get("sr", 24000)
        device = detect_device()

        model, model_params = load_inference_model(config, checkpoint_path, device)

        _current_model["model"] = model
        _current_model["model_params"] = model_params
        _current_model["style_db_path"] = style_db_path
        _current_model["device"] = device
        _current_model["sr"] = sr

        if style_db_path:
            db_meta = torch.load(style_db_path, map_location='cpu')
            db_type = db_meta.get('db_type', 'full')
            n_entries = db_meta['bert_embeds'].shape[0]
            style_info = f"Style DB: {style_db_path} ({db_type}, {n_entries}エントリ)"
            del db_meta
        else:
            style_info = "Style DB: なし"
        return (
            f"読み込み完了: {model_dir}\n"
            f"デバイス: {device} | SR: {sr}\n"
            f"{style_info}"
        )
    except Exception as e:
        return f"読み込みエラー: {e}"


def generate_handler(text, style_mode, speaker_id, ref_audio, diffusion_steps):
    """Generate speech from text."""
    # Validation
    if _current_model["model"] is None:
        return None, "", "エラー: モデルが読み込まれていません。先にモデルを読み込んでください。"
    if not text or not text.strip():
        return None, "", "エラー: テキストを入力してください。"

    model = _current_model["model"]
    model_params = _current_model["model_params"]
    style_db_path = _current_model["style_db_path"]
    device = _current_model["device"]
    sr = _current_model["sr"]
    speaker_id = int(speaker_id)

    try:
        # Split long text into sentences for chunked synthesis
        sentences = split_sentences(text.strip())
        if not sentences:
            return None, "", "エラー: テキストを入力してください。"

        # Phonemize full text for display
        phonemized = smart_phonemize(text)

        # Pre-compute style vectors (once) for non-text-search modes
        shared_style = None
        if style_mode == "代表スタイル":
            if style_db_path is None:
                return None, phonemized, "エラー: Style DBが見つかりません。代表スタイルは利用できません。"
            shared_style = load_repr_style(style_db_path, speaker_id, device=device)
        elif style_mode == "テキスト類似検索":
            if style_db_path is None:
                return None, phonemized, "エラー: Style DBが見つかりません。テキスト類似検索は利用できません。"
            # Style will be looked up per sentence below
        elif style_mode == "リファレンス音声":
            if ref_audio is None:
                return None, phonemized, "エラー: リファレンス音声をアップロードしてください。"
            shared_style = compute_ref_style(model, ref_audio, sr=sr, device=device)
        else:
            return None, phonemized, f"エラー: 不明なスタイルモード: {style_mode}"

        # Synthesize each sentence and concatenate
        wav_chunks = []
        silence = np.zeros(int(sr * 0.1), dtype=np.float32)  # 0.1s gap between sentences

        for i, sentence in enumerate(sentences):
            if style_mode == "テキスト類似検索":
                ref_ss, ref_sp = lookup_style_from_db(
                    model, sentence, style_db_path, speaker_id, device=device,
                )
            else:
                ref_ss, ref_sp = shared_style

            chunk = synthesize(
                model, model_params, sentence,
                ref_ss, ref_sp,
                device=device,
                diffusion_steps=int(diffusion_steps),
                sr=sr,
            )
            wav_chunks.append(chunk)
            if i < len(sentences) - 1:
                wav_chunks.append(silence)

        wav = np.concatenate(wav_chunks)

        status = f"生成完了 ({len(sentences)}文)" if len(sentences) > 1 else "生成完了"
        return (sr, wav), phonemized, status

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None, phonemized, "エラー: GPU メモリ不足です。テキストを短くするか、diffusion stepsを減らしてください。"
    except Exception as e:
        traceback.print_exc()
        return None, "", f"エラー: {e}"


def toggle_ref_audio(style_mode):
    """Show/hide reference audio based on style mode."""
    return gr.update(visible=(style_mode == "リファレンス音声"))


# ---------------------------------------------------------------------------
# UI layout
# ---------------------------------------------------------------------------

def build_inference_tab():
    """Build the inference tab UI."""
    model_choices = discover_model_dirs()

    with gr.Row():
        # Left column: settings
        with gr.Column(scale=1):
            gr.Markdown("### モデル設定")
            model_dir = gr.Dropdown(
                choices=model_choices,
                label="モデル",
                allow_custom_value=True,
            )
            load_btn = gr.Button("モデルを読み込む", variant="primary")
            model_status = gr.Textbox(label="読込状態", interactive=False)

            gr.Markdown("### スタイル設定")
            style_mode = gr.Radio(
                choices=["代表スタイル", "テキスト類似検索", "リファレンス音声"],
                value="代表スタイル",
                label="スタイルモード",
            )
            speaker_id = gr.Number(value=0, label="話者ID", precision=0)
            ref_audio = gr.Audio(
                label="リファレンス音声",
                type="filepath",
                visible=False,
            )

            gr.Markdown("### 生成パラメータ")
            diffusion_steps = gr.Slider(
                minimum=0, maximum=20, step=1, value=5,
                label="Diffusion Steps",
            )

        # Right column: input/output
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="テキスト入力",
                placeholder="ここにテキストを入力してください...",
                lines=3,
            )
            generate_btn = gr.Button("音声を生成", variant="primary")
            audio_output = gr.Audio(label="生成音声", type="numpy")
            phoneme_output = gr.Textbox(label="音素変換結果", interactive=False)
            gen_status = gr.Textbox(label="ステータス", interactive=False)

    # Events
    load_btn.click(
        fn=load_model_handler,
        inputs=[model_dir],
        outputs=[model_status],
    )
    generate_btn.click(
        fn=generate_handler,
        inputs=[text_input, style_mode, speaker_id, ref_audio, diffusion_steps],
        outputs=[audio_output, phoneme_output, gen_status],
    )
    style_mode.change(
        fn=toggle_ref_audio,
        inputs=[style_mode],
        outputs=[ref_audio],
    )


# ---------------------------------------------------------------------------
# Training: helpers
# ---------------------------------------------------------------------------

def discover_training_models():
    """Scan Data/ for directories that contain speaker subdirectories."""
    models = []
    if not osp.isdir("Data"):
        return models
    for name in sorted(os.listdir("Data")):
        d = osp.join("Data", name)
        if not osp.isdir(d):
            continue
        # Check if it contains at least one speaker dir (has wav/ + transcript)
        for sub in os.listdir(d):
            sub_path = osp.join(d, sub)
            if (osp.isdir(sub_path)
                    and osp.isdir(osp.join(sub_path, "wav"))
                    and osp.isfile(osp.join(sub_path, "transcript_utf8.txt"))):
                models.append(name)
                break
    return models


def get_model_data_status(model_name):
    """Return a summary of speakers and entries for a model directory."""
    if not model_name:
        return "モデルを選択してください。"
    data_dir = osp.join("Data", model_name)
    if not osp.isdir(data_dir):
        return f"ディレクトリが見つかりません: {data_dir}"

    speakers = []
    total_entries = 0
    for name in sorted(os.listdir(data_dir)):
        sp_dir = osp.join(data_dir, name)
        transcript = osp.join(sp_dir, "transcript_utf8.txt")
        if osp.isdir(sp_dir) and osp.isdir(osp.join(sp_dir, "wav")) and osp.isfile(transcript):
            count = 0
            with open(transcript, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("処理開始"):
                        count += 1
            speakers.append(f"{name} ({count}件)")
            total_entries += count

    if not speakers:
        return f"話者ディレクトリが見つかりません。\n期待構造: Data/{model_name}/話者名/wav/*.wav + transcript_utf8.txt"

    # Check if train/val lists exist
    train_list = osp.join(data_dir, "train_list.txt")
    val_list = osp.join(data_dir, "val_list.txt")
    list_status = ""
    if osp.isfile(train_list):
        with open(train_list, "r") as f:
            train_count = sum(1 for line in f if line.strip())
        list_status += f"\ntrain_list.txt: {train_count}件"
    if osp.isfile(val_list):
        with open(val_list, "r") as f:
            val_count = sum(1 for line in f if line.strip())
        list_status += f"\nval_list.txt: {val_count}件"

    return (
        f"話者: {', '.join(speakers)} ({len(speakers)}名)\n"
        f"全エントリ: {total_entries}件"
        f"{list_status}"
    )


# ---------------------------------------------------------------------------
# Training: subprocess manager
# ---------------------------------------------------------------------------

class TrainingManager:
    """Manages training subprocess lifecycle."""

    def __init__(self):
        self.process: subprocess.Popen | None = None
        self.tb_process: subprocess.Popen | None = None
        self.log_buffer: list[str] = []
        self.status: str = "idle"
        self._lock = threading.Lock()
        self._log_thread: threading.Thread | None = None

    def _stream_output(self):
        """Read subprocess stdout/stderr in a background thread."""
        proc = self.process
        if proc is None:
            return
        try:
            for line in iter(proc.stdout.readline, ""):
                if not line:
                    break
                with self._lock:
                    self.log_buffer.append(line)
        except (ValueError, OSError):
            pass
        proc.wait()
        with self._lock:
            rc = proc.returncode
            if rc == 0:
                self.log_buffer.append(f"\n=== プロセス正常終了 (code {rc}) ===\n")
            elif rc == -signal.SIGTERM or rc == -signal.SIGINT:
                self.log_buffer.append(f"\n=== プロセス停止 (signal {-rc}) ===\n")
            else:
                self.log_buffer.append(f"\n=== プロセス異常終了 (code {rc}) ===\n")
            self.status = "idle"
            self.process = None
            self._stop_tensorboard()

    def _launch(self, cmd, env=None, status_label="running"):
        """Launch a subprocess and start log streaming."""
        if self.process is not None:
            return "エラー: 既にプロセスが実行中です。先に停止してください。"

        merged_env = os.environ.copy()
        merged_env["PYTHONUNBUFFERED"] = "1"
        if env:
            merged_env.update(env)

        self.log_buffer.clear()
        self.log_buffer.append(f"$ {' '.join(cmd)}\n")
        self.status = status_label

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=merged_env,
            start_new_session=True,
        )
        self._log_thread = threading.Thread(target=self._stream_output, daemon=True)
        self._log_thread.start()
        return f"開始しました (PID: {self.process.pid})"

    def start_preprocess(self, model_name, val_ratio, max_duration, n_jobs):
        data_dir = osp.join("Data", model_name)
        cmd = [
            "python", "preprocess_data.py",
            "--data-dir", data_dir,
            "--val-ratio", str(val_ratio),
            "--n_jobs", str(n_jobs),
            "--cache-wavs",
        ]
        if max_duration and float(max_duration) > 0:
            cmd.extend(["--max-duration", str(max_duration)])
        return self._launch(cmd, status_label="preprocessing")

    def _start_tensorboard(self, log_dir):
        self._stop_tensorboard()
        tb_logdir = osp.join(log_dir, "tensorboard")
        os.makedirs(tb_logdir, exist_ok=True)
        try:
            self.tb_process = subprocess.Popen(
                ["tensorboard", "--logdir", tb_logdir, "--host", "0.0.0.0"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            with self._lock:
                self.log_buffer.append(
                    f"TensorBoard 起動 (PID: {self.tb_process.pid}) http://0.0.0.0:6006\n"
                )
        except FileNotFoundError:
            with self._lock:
                self.log_buffer.append("TensorBoard が見つかりません。スキップします。\n")

    def _stop_tensorboard(self):
        if self.tb_process is None:
            return
        try:
            pgid = os.getpgid(self.tb_process.pid)
            os.killpg(pgid, signal.SIGTERM)
            self.tb_process.wait(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired, OSError):
            try:
                self.tb_process.kill()
            except OSError:
                pass
        self.tb_process = None

    def start_training(self, model_name, stage, gpu_tier, max_steps, batch_size, warmup_steps=0):
        data_dir = osp.join("Data", model_name)

        # Write run_config.yaml
        run_config = {"stage": str(stage)}
        if gpu_tier and gpu_tier != "auto":
            run_config["gpu_tier"] = gpu_tier
        if max_steps and int(max_steps) > 0:
            run_config["max_steps"] = int(max_steps)
        if batch_size and int(batch_size) > 0:
            run_config["batch_size"] = int(batch_size)
        if warmup_steps and int(warmup_steps) > 0:
            run_config["stage1_steps"] = int(warmup_steps)

        os.makedirs(data_dir, exist_ok=True)
        run_config_path = osp.join(data_dir, "run_config.yaml")
        with open(run_config_path, "w", encoding="utf-8") as f:
            yaml.dump(run_config, f, default_flow_style=False, allow_unicode=True)
        self.log_buffer.append(f"run_config.yaml を書き出しました: {run_config_path}\n")

        env = {"DATA_DIR": data_dir, "STAGE": str(stage)}
        cmd = ["bash", "train.sh"]
        result = self._launch(cmd, env=env, status_label="training")
        self._start_tensorboard(osp.join(data_dir, "output"))
        return result

    def start_style_db(self, model_name):
        # Find output dir - look for subdirectories or use output directly
        output_base = osp.join("Data", model_name, "output")
        model_dir = None
        if osp.isdir(output_base):
            # Check for tier subdirectories first (e.g. MidVRAM)
            for sub in sorted(os.listdir(output_base)):
                sub_path = osp.join(output_base, sub)
                inf_dir = osp.join(sub_path, "inference")
                if osp.isdir(inf_dir):
                    model_dir = sub_path
                    break
            # Fall back to output dir itself
            if model_dir is None and osp.isdir(osp.join(output_base, "inference")):
                model_dir = output_base
        if model_dir is None:
            return "エラー: 推論用モデルが見つかりません。学習を先に実行してください。"

        cmd = ["python", "precompute_styles.py", "--model-dir", model_dir]
        return self._launch(cmd, status_label="postprocessing")

    def stop(self):
        if self.process is None:
            return "プロセスは実行されていません。"
        pgid = os.getpgid(self.process.pid)
        try:
            os.killpg(pgid, signal.SIGTERM)
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(pgid, signal.SIGKILL)
            self.process.wait()
        except ProcessLookupError:
            pass
        self._stop_tensorboard()
        return "プロセスを停止しました。"

    def read_log(self):
        with self._lock:
            text = "".join(self.log_buffer[-500:])
        return text

    def get_status(self):
        labels = {
            "idle": "待機中",
            "preprocessing": "前処理中...",
            "training": "学習中...",
            "postprocessing": "後処理中...",
        }
        return labels.get(self.status, self.status)


_training_manager = TrainingManager()


# ---------------------------------------------------------------------------
# Training: event handlers
# ---------------------------------------------------------------------------

def on_model_select(model_name):
    return get_model_data_status(model_name)


def on_create_model(new_name):
    if not new_name or not new_name.strip():
        return "モデル名を入力してください。", gr.update()
    name = new_name.strip()
    if not re.match(r'^[\w\-]+$', name):
        return "モデル名は英数字・アンダースコア・ハイフンのみ使用できます。", gr.update()
    model_dir = osp.join("Data", name)
    if osp.exists(model_dir):
        return f"既に存在します: {model_dir}", gr.update()
    os.makedirs(model_dir, exist_ok=True)
    models = discover_training_models()
    if name not in models:
        models.append(name)
    return f"作成しました: {model_dir}", gr.update(choices=models, value=name)


def on_refresh_models():
    models = discover_training_models()
    return gr.update(choices=models)


def on_preprocess(model_name, val_ratio, max_duration, n_jobs):
    if not model_name:
        return "モデルを選択してください。"
    return _training_manager.start_preprocess(model_name, val_ratio, max_duration, n_jobs)


def on_start_training(model_name, active_tab,
                      ft_gpu_tier, ft_max_steps, ft_batch_size, ft_warmup_steps,
                      pt_gpu_tier, pt_max_steps, pt_batch_size):
    if not model_name:
        return "モデルを選択してください。"
    if active_tab == "pretrain":
        stage, gpu_tier, max_steps, batch_size = "1", pt_gpu_tier, pt_max_steps, pt_batch_size
        warmup_steps = 0
    else:
        stage, gpu_tier, max_steps, batch_size = "2", ft_gpu_tier, ft_max_steps, ft_batch_size
        warmup_steps = ft_warmup_steps
    return _training_manager.start_training(model_name, stage, gpu_tier, max_steps, batch_size, warmup_steps)


def on_stop_training():
    return _training_manager.stop()


def on_style_db(model_name):
    if not model_name:
        return "モデルを選択してください。"
    return _training_manager.start_style_db(model_name)


def on_poll_log():
    return _training_manager.read_log(), _training_manager.get_status()


# ---------------------------------------------------------------------------
# Training: UI
# ---------------------------------------------------------------------------

def build_training_tab():
    """Build the training tab UI."""
    model_choices = discover_training_models()

    with gr.Row():
        # Left column: settings
        with gr.Column(scale=1):
            gr.Markdown("### モデル選択")
            with gr.Row():
                model_dropdown = gr.Dropdown(
                    choices=model_choices,
                    label="モデル名",
                    allow_custom_value=False,
                    scale=3,
                )
                refresh_btn = gr.Button("更新", scale=1)

            with gr.Row():
                new_model_input = gr.Textbox(
                    label="新規モデル名",
                    placeholder="my_model",
                    scale=3,
                )
                create_btn = gr.Button("作成", scale=1)
            create_status = gr.Textbox(label="", interactive=False, visible=True, max_lines=1)

            gr.Markdown("### 学習パラメータ")
            active_tab = gr.State("finetune")
            with gr.Tabs() as train_tabs:
                with gr.Tab("Finetuning", id="finetune"):
                    ft_gpu_tier = gr.Dropdown(
                        choices=["auto", "low", "mid", "high"],
                        value="auto",
                        label="GPU Tier",
                    )
                    ft_max_steps = gr.Number(
                        value=50000,
                        label="Max Steps",
                        precision=0,
                    )
                    ft_batch_size = gr.Number(
                        value=0,
                        label="Batch Size (0=config default)",
                        precision=0,
                    )
                    ft_warmup_steps = gr.Number(
                        value=0,
                        label="Stage1 Warm Up Steps (0=config default)",
                        precision=0,
                    )
                with gr.Tab("Pre-training", id="pretrain"):
                    pt_gpu_tier = gr.Dropdown(
                        choices=["auto", "low", "mid", "high"],
                        value="auto",
                        label="GPU Tier",
                    )
                    pt_max_steps = gr.Number(
                        value=50000,
                        label="Max Steps",
                        precision=0,
                    )
                    pt_batch_size = gr.Number(
                        value=0,
                        label="Batch Size (0=config default)",
                        precision=0,
                    )

            _tab_id_map = {"Finetuning": "finetune", "Pre-training": "pretrain"}

            def on_tab_select(evt: gr.SelectData):
                return _tab_id_map.get(evt.value, "finetune")

            train_tabs.select(fn=on_tab_select, outputs=[active_tab])

            gr.Markdown("### 前処理パラメータ")
            val_ratio = gr.Number(
                value=0.1,
                label="Val Ratio",
            )
            max_duration = gr.Number(
                value=0,
                label="Max Duration (秒, 0=無制限)",
            )
            n_jobs = gr.Number(
                value=4,
                label="前処理 並列数",
                precision=0,
            )

            gr.Markdown("### 後処理")
            style_db_btn = gr.Button("Style DB 生成")

        # Right column: execution
        with gr.Column(scale=1):
            gr.Markdown("### データ状態")
            data_status = gr.Textbox(
                label="",
                interactive=False,
                lines=5,
            )

            gr.Markdown("### 実行")
            with gr.Row():
                preprocess_btn = gr.Button("前処理")
                train_btn = gr.Button("学習開始", variant="primary")
                stop_btn = gr.Button("学習停止", variant="stop")

            process_status = gr.Textbox(
                label="ステータス",
                interactive=False,
                max_lines=1,
                value="待機中",
            )

            gr.Markdown("### ログ")
            log_area = gr.Textbox(
                label="",
                interactive=False,
                lines=20,
                max_lines=20,
                autoscroll=True,
            )

    # --- Events ---
    model_dropdown.change(
        fn=on_model_select,
        inputs=[model_dropdown],
        outputs=[data_status],
    )
    refresh_btn.click(
        fn=on_refresh_models,
        outputs=[model_dropdown],
    )
    create_btn.click(
        fn=on_create_model,
        inputs=[new_model_input],
        outputs=[create_status, model_dropdown],
    )
    preprocess_btn.click(
        fn=on_preprocess,
        inputs=[model_dropdown, val_ratio, max_duration, n_jobs],
        outputs=[process_status],
    )
    train_btn.click(
        fn=on_start_training,
        inputs=[model_dropdown, active_tab,
                ft_gpu_tier, ft_max_steps, ft_batch_size, ft_warmup_steps,
                pt_gpu_tier, pt_max_steps, pt_batch_size],
        outputs=[process_status],
    )
    stop_btn.click(
        fn=on_stop_training,
        outputs=[process_status],
    )
    style_db_btn.click(
        fn=on_style_db,
        inputs=[model_dropdown],
        outputs=[process_status],
    )

    # Timer for log polling (every 2 seconds)
    timer = gr.Timer(2)
    timer.tick(
        fn=on_poll_log,
        outputs=[log_area, process_status],
    )


# ---------------------------------------------------------------------------
# App entry point
# ---------------------------------------------------------------------------

def create_app():
    with gr.Blocks(title="Tsukasa Speech") as app:
        gr.Markdown("# Tsukasa Speech")
        with gr.Tabs():
            with gr.Tab("音声合成"):
                build_inference_tab()
            with gr.Tab("学習"):
                build_training_tab()
    return app


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    app = create_app()
    app.launch(server_name=args.host, server_port=args.port)
