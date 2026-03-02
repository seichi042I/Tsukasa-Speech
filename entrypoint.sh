#!/bin/bash

echo "=== Tsukasa Speech Container ==="
echo "Started at: $(date)"

# ---- Environment setup ----
CACHE_DIR="${TSUKASA_CACHE_DIR:-/tmp/wave_cache}"
mkdir -p "$CACHE_DIR"

# ---- Download models if not present + pre-cache HF models ----
echo ""
echo "=== Checking model weights ==="
python download_models.py || {
    echo "ERROR: Model download failed."
    echo "Container staying alive for SSH access."
    sleep infinity
}

# ---- Interactive mode: shell only ----
if [ "${STAGE:-}" = "shell" ]; then
    echo "Interactive mode (STAGE=shell). Run 'train' to start training."
    exec bash
fi

# ---- Launch Gradio UI ----
echo ""
echo "=== Starting Gradio UI ==="
exec python app.py --host 0.0.0.0 --port "${GRADIO_PORT:-7860}"
