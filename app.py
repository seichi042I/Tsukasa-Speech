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
    predict_prosody,
    synthesize_from_prosody,
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


def prosody_states_to_editor_data(prosody_states, hop_length=300):
    """Convert list of prosody_state dicts to F0EditorData dict for the custom component."""
    if not prosody_states:
        return None
    sr = prosody_states[0]['sr']
    sentences = []
    for state in prosody_states:
        sentences.append({
            'f0': state['F0'].squeeze().cpu().tolist(),
            'n': state['N'].squeeze().cpu().tolist(),
            'f0_original': state['F0_original'].squeeze().cpu().tolist(),
            'n_original': state['N_original'].squeeze().cpu().tolist(),
            'phonemes': state.get('phonemes', []),
        })
    return {
        'sentences': sentences,
        'hop_length': hop_length,
        'sr': sr,
    }


def editor_data_to_prosody_states(editor_data, prosody_states):
    """Apply edited F0/N values from editor_data back into prosody_states tensors."""
    if editor_data is None or prosody_states is None:
        return prosody_states
    for i, (sent_data, state) in enumerate(zip(editor_data['sentences'], prosody_states)):
        device = state['F0'].device
        state['F0'] = torch.tensor(sent_data['f0'], device=device).unsqueeze(0)
        state['N'] = torch.tensor(sent_data['n'], device=device).unsqueeze(0)
    return prosody_states


def split_sentences(text, min_chunk=25):
    """Split text on 。 for chunked synthesis, keeping short phrases together.

    After a split point, the next min_chunk characters are never split.
    Beyond min_chunk, the text is split at the next 。.
    """
    parts = re.split(r'(。)', text)
    sentences = []
    buf = ""
    chunk_len = 0  # character count since last split
    for part in parts:
        buf += part
        chunk_len += len(part)
        if part == '。' and chunk_len > min_chunk:
            s = buf.strip()
            if s:
                sentences.append(s)
            buf = ""
            chunk_len = 0
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


def generate_handler(text, style_mode, speaker_id, ref_audio, diffusion_steps, style_strength,
                     prosody_states, editor_data=None):
    """Generate speech from text. Uses pre-computed prosody if available.

    If editor_data is provided (from F0Editor component), edited F0/N values
    are applied back to prosody_states before synthesis.
    """
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
        # Apply F0Editor edits back to prosody_states
        if editor_data is not None and prosody_states is not None:
            prosody_states = editor_data_to_prosody_states(editor_data, prosody_states)

        # If prosody_states exist (from preview), use them directly
        if prosody_states is not None:
            wav_chunks = []
            silence = np.zeros(int(sr * 0.1), dtype=np.float32)
            phonemized_parts = []

            for i, state in enumerate(prosody_states):
                chunk = synthesize_from_prosody(model, state)
                wav_chunks.append(chunk)
                phonemized_parts.append(state['phonemized'])
                if i < len(prosody_states) - 1:
                    wav_chunks.append(silence)

            wav = np.concatenate(wav_chunks)
            phonemized = ' | '.join(phonemized_parts)
            n = len(prosody_states)
            status = f"生成完了 ({n}文, プレビューから)" if n > 1 else "生成完了 (プレビューから)"
            return (sr, wav), phonemized, status

        # Fallback: synthesize from scratch
        sentences = split_sentences(text.strip())
        if not sentences:
            return None, "", "エラー: テキストを入力してください。"

        phonemized = smart_phonemize(text)

        shared_style = None
        if style_mode == "代表スタイル":
            if style_db_path is None:
                return None, phonemized, "エラー: Style DBが見つかりません。代表スタイルは利用できません。"
            shared_style = load_repr_style(style_db_path, speaker_id, device=device)
        elif style_mode == "テキスト類似検索":
            if style_db_path is None:
                return None, phonemized, "エラー: Style DBが見つかりません。テキスト類似検索は利用できません。"
        elif style_mode == "リファレンス音声":
            if ref_audio is None:
                return None, phonemized, "エラー: リファレンス音声をアップロードしてください。"
            shared_style = compute_ref_style(model, ref_audio, sr=sr, device=device)
        else:
            return None, phonemized, f"エラー: 不明なスタイルモード: {style_mode}"

        wav_chunks = []
        silence = np.zeros(int(sr * 0.1), dtype=np.float32)

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
                style_strength=float(style_strength),
            )
            wav_chunks.append(chunk)
            if i < len(sentences) - 1:
                wav_chunks.append(silence)

        wav = np.concatenate(wav_chunks)

        status = f"生成完了 ({len(sentences)}文)" if len(sentences) > 1 else "生成完了"
        return (sr, wav), phonemized, status

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None, "", "エラー: GPU メモリ不足です。テキストを短くするか、diffusion stepsを減らしてください。"
    except Exception as e:
        traceback.print_exc()
        return None, "", f"エラー: {e}"


def toggle_ref_audio(style_mode):
    """Show/hide reference audio based on style mode."""
    return gr.update(visible=(style_mode == "リファレンス音声"))


def preview_handler(text, style_mode, speaker_id, ref_audio, diffusion_steps, style_strength):
    """Predict prosody (F0/energy) without decoding to waveform.

    Returns: (editor_data, prosody_states, phonemized, status)
    """
    if _current_model["model"] is None:
        return None, None, "", "エラー: モデルが読み込まれていません。"
    if not text or not text.strip():
        return None, None, "", "エラー: テキストを入力してください。"

    model = _current_model["model"]
    model_params = _current_model["model_params"]
    style_db_path = _current_model["style_db_path"]
    device = _current_model["device"]
    sr = _current_model["sr"]
    speaker_id = int(speaker_id)

    try:
        sentences = split_sentences(text.strip())
        if not sentences:
            return None, None, "", "エラー: テキストを入力してください。"

        # Pre-compute style vectors
        shared_style = None
        if style_mode == "代表スタイル":
            if style_db_path is None:
                return None, None, "", "エラー: Style DBが見つかりません。"
            shared_style = load_repr_style(style_db_path, speaker_id, device=device)
        elif style_mode == "テキスト類似検索":
            if style_db_path is None:
                return None, None, "", "エラー: Style DBが見つかりません。"
        elif style_mode == "リファレンス音声":
            if ref_audio is None:
                return None, None, "", "エラー: リファレンス音声をアップロードしてください。"
            shared_style = compute_ref_style(model, ref_audio, sr=sr, device=device)
        else:
            return None, None, "", f"エラー: 不明なスタイルモード: {style_mode}"

        prosody_states = []
        phonemized_parts = []
        for sentence in sentences:
            if style_mode == "テキスト類似検索":
                ref_ss, ref_sp = lookup_style_from_db(
                    model, sentence, style_db_path, speaker_id, device=device,
                )
            else:
                ref_ss, ref_sp = shared_style

            state = predict_prosody(
                model, model_params, sentence,
                ref_ss, ref_sp,
                device=device,
                diffusion_steps=int(diffusion_steps),
                sr=sr,
                style_strength=float(style_strength),
            )
            prosody_states.append(state)
            phonemized_parts.append(state['phonemized'])

        phonemized = ' | '.join(phonemized_parts)

        hop_length = 300
        total_frames = sum(s['F0'].shape[-1] for s in prosody_states)
        total_sec = total_frames * hop_length / sr
        status = f"プレビュー完了 ({len(sentences)}文, {total_sec:.2f}秒)"

        editor_data = prosody_states_to_editor_data(prosody_states, hop_length)
        return editor_data, prosody_states, phonemized, status

    except Exception as e:
        traceback.print_exc()
        return None, None, "", f"エラー: {e}"


# ---------------------------------------------------------------------------
# F0 Canvas Editor (HTML + inline JS)
# ---------------------------------------------------------------------------

def _load_f0_editor_js():
    """Load the F0 canvas editor JavaScript from static/f0_editor.js."""
    js_path = osp.join(osp.dirname(osp.abspath(__file__)), "static", "f0_editor.js")
    with open(js_path, "r", encoding="utf-8") as f:
        return f.read()


def _build_f0_editor_html():
    """Build the HTML string for the Canvas F0 editor (no scripts)."""
    return """
<div style="border:1px solid #ddd; border-radius:8px; overflow:hidden; background:#fafafa;">
  <div class="f0-toolbar" style="display:flex; gap:6px; padding:6px 10px; background:#f0f0f0; border-bottom:1px solid #ddd; align-items:center; flex-wrap:wrap;">
    <button class="f0-toggle-btn" style="padding:4px 12px; border:1px solid #ccc; border-radius:4px; background:#fff; color:#333; cursor:pointer; font-size:13px;">F0 / Energy</button>
    <button class="f0-undo-btn" disabled style="padding:4px 10px; border:1px solid #ccc; border-radius:4px; background:#fff; color:#333; cursor:pointer; font-size:13px;">&#x21A9; Undo</button>
    <button class="f0-redo-btn" disabled style="padding:4px 10px; border:1px solid #ccc; border-radius:4px; background:#fff; color:#333; cursor:pointer; font-size:13px;">&#x21AA; Redo</button>
    <button class="f0-reset-btn" style="padding:4px 10px; border:1px solid #ccc; border-radius:4px; background:#fff; color:#333; cursor:pointer; font-size:13px;">&#x21BA; Reset</button>
    <button class="f0-clearsel-btn" style="padding:4px 10px; border:1px solid #ccc; border-radius:4px; background:#fff; color:#333; cursor:pointer; font-size:13px;">&#x2717; Deselect</button>
    <span style="margin-left:auto; font-size:11px; color:#666;">Drag: edit | Alt+Drag: select | Shift/Right: pan | Wheel: zoom</span>
  </div>
  <canvas class="f0-canvas" style="display:block; width:100%; cursor:crosshair;"></canvas>
</div>
"""


def _build_f0_editor_js_on_load(js_code):
    """Build the js_on_load string for Canvas F0 editor initialization.

    Uses ``element.querySelector()`` (Gradio 6 js_on_load API) instead of
    ``document.getElementById()`` so the editor works inside the component's
    DOM subtree.
    """
    # The js_code defines the F0CanvasEditor class globally.
    # The init IIFE below finds the canvas inside `element` (the gr.HTML root).
    init_js = """
(function() {
  var canvas = element.querySelector('.f0-canvas');
  if (!canvas) return;

  var root = canvas.parentElement;
  var w = root.clientWidth || 600;
  var h = 350;
  var editor = new F0CanvasEditor(canvas);
  editor.resize(w, h);
  window.f0Canvas = editor;

  function updateToolbar() {
    var undoBtn = element.querySelector('.f0-undo-btn');
    var redoBtn = element.querySelector('.f0-redo-btn');
    var toggleBtn = element.querySelector('.f0-toggle-btn');
    if (undoBtn) undoBtn.disabled = !editor.canUndo;
    if (redoBtn) redoBtn.disabled = !editor.canRedo;
    if (toggleBtn) {
      var t = editor.getEditTarget();
      toggleBtn.textContent = t === 'f0' ? 'F0 / Energy' : 'Energy / F0';
      toggleBtn.style.fontWeight = 'bold';
    }
  }

  editor.setOnChange(function() { updateToolbar(); });

  element.querySelector('.f0-toggle-btn').onclick = function() { editor.toggleTarget(); updateToolbar(); };
  element.querySelector('.f0-undo-btn').onclick = function() { editor.undo(); updateToolbar(); };
  element.querySelector('.f0-redo-btn').onclick = function() { editor.redo(); updateToolbar(); };
  element.querySelector('.f0-reset-btn').onclick = function() { editor.resetToOriginal(); updateToolbar(); };
  element.querySelector('.f0-clearsel-btn').onclick = function() { editor.clearSelection(); };

  if (typeof ResizeObserver !== 'undefined') {
    new ResizeObserver(function() {
      var newW = root.clientWidth;
      if (newW > 0 && newW !== w) {
        w = newW;
        editor.resize(w, h);
      }
    }).observe(root);
  }
})();
"""
    return js_code + "\n" + init_js


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
            with gr.Row():
                model_dir = gr.Dropdown(
                    choices=model_choices,
                    label="モデル",
                    allow_custom_value=True,
                    scale=3,
                )
                inf_refresh_btn = gr.Button("更新", scale=1)
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
                minimum=0, maximum=100, step=1, value=0,
                label="Diffusion Steps",
            )
            style_strength = gr.Slider(
                minimum=0.0, maximum=1.0, step=0.02, value=0.5,
                label="スタイル強度 (0=平均的 / 1=特徴的)",
            )

        # Right column: input/output
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="テキスト入力",
                placeholder="ここにテキストを入力してください...",
                lines=3,
            )
            with gr.Row():
                preview_btn = gr.Button("プレビュー (F0予測)")
                generate_btn = gr.Button("音声を生成", variant="primary")

            prosody_state = gr.State(None)

            # Canvas-based F0 editor (gr.HTML + gr.JSON bridge)
            editor_json = gr.JSON(visible=False)
            f0_editor_html = gr.HTML(
                value=_build_f0_editor_html(),
                js_on_load=_build_f0_editor_js_on_load(_load_f0_editor_js()),
            )

            audio_output = gr.Audio(label="生成音声", type="numpy")
            phoneme_output = gr.Textbox(label="音素変換結果", interactive=False)
            gen_status = gr.Textbox(label="ステータス", interactive=False)

    # Events
    load_btn.click(
        fn=load_model_handler,
        inputs=[model_dir],
        outputs=[model_status],
    )

    # Preview: Python predicts prosody → editor_data → gr.JSON → .then(js) → Canvas
    preview_btn.click(
        fn=preview_handler,
        inputs=[text_input, style_mode, speaker_id, ref_audio, diffusion_steps, style_strength],
        outputs=[editor_json, prosody_state, phoneme_output, gen_status],
    ).then(
        fn=lambda d: d,
        inputs=[editor_json],
        outputs=[editor_json],
        js="(data) => { if (window.f0Canvas && data) { window.f0Canvas.setData(data); } return data; }",
    )

    # Generate: Canvas getData() → gr.JSON → Python
    # Step 1: JS extracts canvas data, returns it; fn=identity round-trips
    #         the value so editor_json is updated on both client and server.
    # Step 2: .then() calls the actual handler with the updated editor_json.
    generate_btn.click(
        fn=lambda d: d,
        inputs=[editor_json],
        outputs=[editor_json],
        js="(data) => { return window.f0Canvas ? window.f0Canvas.getData() : data; }",
    ).then(
        fn=generate_handler,
        inputs=[text_input, style_mode, speaker_id, ref_audio,
                diffusion_steps, style_strength, prosody_state, editor_json],
        outputs=[audio_output, phoneme_output, gen_status],
    )

    inf_refresh_btn.click(
        fn=lambda: gr.update(choices=discover_model_dirs()),
        outputs=[model_dir],
    )
    style_mode.change(
        fn=toggle_ref_audio,
        inputs=[style_mode],
        outputs=[ref_audio],
    )

    # Invalidate prosody_states when inputs change
    invalidate_inputs = [text_input, style_mode, speaker_id, ref_audio, diffusion_steps, style_strength]
    for component in invalidate_inputs:
        component.change(
            fn=lambda: None,
            outputs=[prosody_state],
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
