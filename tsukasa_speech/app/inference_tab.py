"""Inference tab: model loading, text-to-speech generation, and F0 editor."""

import os
import os.path as osp
import re
import traceback

import torch
import numpy as np
import yaml
import gradio as gr

from tsukasa_speech.inference.model_loader import resolve_model_dir, normalize_config, load_inference_model
from tsukasa_speech.inference.style import compute_ref_style, lookup_style_from_db, load_repr_style
from tsukasa_speech.inference.core import synthesize, predict_prosody, synthesize_from_prosody
from tsukasa_speech.utils.phonemize.mixed_phon import smart_phonemize

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
    """Split text on punctuation for chunked synthesis, keeping short phrases together.

    After a split point, the next *min_chunk* characters are never split.
    Beyond *min_chunk*, the text is split at the next full-stop.
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
    # Trailing text without full-stop
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
    """Generate speech from text.  Uses pre-computed prosody if available.

    If *editor_data* is provided (from F0Editor component), edited F0/N values
    are applied back to *prosody_states* before synthesis.
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
    js_path = osp.join(osp.dirname(osp.abspath(__file__)), "..", "..", "static", "f0_editor.js")
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
    """Build the inference tab UI inside the current Gradio context."""
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

    # Preview: Python predicts prosody -> editor_data -> gr.JSON -> .then(js) -> Canvas
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

    # Generate: Canvas getData() -> gr.JSON -> Python
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
