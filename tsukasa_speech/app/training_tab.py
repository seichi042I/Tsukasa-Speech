"""Training tab: data preparation, training launch, and log monitoring."""

import os
import os.path as osp
import re

import gradio as gr

from tsukasa_speech.app.training_manager import TrainingManager

# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_training_manager = TrainingManager()

# ---------------------------------------------------------------------------
# Helpers
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
# Event handlers
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
# UI layout
# ---------------------------------------------------------------------------

def build_training_tab():
    """Build the training tab UI inside the current Gradio context."""
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
