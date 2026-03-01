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

# ---- Auto-start training ----
echo "Starting training pipeline..."
if ./train.sh; then
    echo ""
    echo "=== Training completed successfully ==="
else
    echo ""
    echo "=== Training failed (exit code: $?) ==="
    echo "SSH in and run 'train' to retry."
fi

# Keep container alive for SSH access after training
echo ""
echo "Container staying alive for SSH access. Run 'train' to re-run."
sleep infinity
