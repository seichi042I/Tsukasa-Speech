#!/bin/bash
set -euo pipefail
echo "=== Tsukasa Speech Training Pipeline ==="
echo "Started at: $(date)"

python -m tsukasa_speech.training \
    --data-dir "${DATA_DIR:-Data}" \
    --stage "${STAGE:-}" \
    --cache-dir "${TSUKASA_CACHE_DIR:-}" \
    --n-jobs "${N_JOBS:-4}" \
    --gpu-tier "${GPU_TIER_OVERRIDE:-}"

echo "=== Training complete at: $(date) ==="
