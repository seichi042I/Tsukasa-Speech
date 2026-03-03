"""Subprocess manager for training, preprocessing, and style-DB generation."""

import os
import os.path as osp
import signal
import subprocess
import sys
import threading

import yaml


class TrainingManager:
    """Manages training subprocess lifecycle."""

    def __init__(self):
        self.process: subprocess.Popen | None = None
        self.tb_process: subprocess.Popen | None = None
        self.log_buffer: list[str] = []
        self.status: str = "idle"
        self._lock = threading.Lock()
        self._log_thread: threading.Thread | None = None

    # -- internal helpers ----------------------------------------------------

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

    # -- public API ----------------------------------------------------------

    def start_preprocess(self, model_name, val_ratio, max_duration, n_jobs):
        data_dir = osp.join("Data", model_name)
        cmd = [
            sys.executable, "-m", "tsukasa_speech.preprocessing.phonemize_data",
            "--data-dir", data_dir,
            "--val-ratio", str(val_ratio),
            "--n_jobs", str(n_jobs),
            "--cache-wavs",
        ]
        if max_duration and float(max_duration) > 0:
            cmd.extend(["--max-duration", str(max_duration)])
        return self._launch(cmd, status_label="preprocessing")

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
        cmd = [sys.executable, "-m", "tsukasa_speech.training",
               "--data-dir", data_dir, "--stage", str(stage)]
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

        cmd = [sys.executable, "-m", "tsukasa_speech.preprocessing.build_styles", "--model-dir", model_dir]
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
