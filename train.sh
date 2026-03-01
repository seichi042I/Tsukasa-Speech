#!/bin/bash
set -euo pipefail

echo "=== Tsukasa Speech Training Pipeline ==="
echo "Started at: $(date)"

DATA_DIR="${DATA_DIR:-Data}"
CACHE_DIR="${TSUKASA_CACHE_DIR:-/tmp/wave_cache}"
RUN_CONFIG="${DATA_DIR}/run_config.yaml"
N_JOBS="${N_JOBS:-4}"

mkdir -p "$CACHE_DIR"
mkdir -p "${DATA_DIR}/output"

# ---- Step 1: Read stage from env or run_config.yaml ----
if [ -n "${STAGE:-}" ]; then
    echo "Stage from environment: $STAGE"
elif [ -f "$RUN_CONFIG" ]; then
    STAGE=$(python -c "
import yaml
with open('${RUN_CONFIG}') as f:
    cfg = yaml.safe_load(f) or {}
print(cfg.get('stage', 'all'))
")
    echo "Stage from run_config.yaml: $STAGE"
else
    STAGE="all"
    echo "No run_config.yaml found, using default stage: $STAGE"
fi

# ---- Step 2: Auto-detect GPU tier ----
echo ""
echo "=== GPU Detection ==="

GPU_TIER=$(python detect_gpu.py 2>&1 | tee /dev/stderr | tail -1)
BASE_CONFIG=$(python detect_gpu.py --config)

# Allow override via GPU_TIER env var or run_config.yaml
if [ -n "${GPU_TIER_OVERRIDE:-}" ]; then
    GPU_TIER="$GPU_TIER_OVERRIDE"
    BASE_CONFIG="Configs/config_${GPU_TIER}_vram.yml"
    echo "Overridden by GPU_TIER_OVERRIDE env: $GPU_TIER"
elif [ -f "$RUN_CONFIG" ]; then
    USER_TIER=$(python -c "
import yaml
with open('${RUN_CONFIG}') as f:
    cfg = yaml.safe_load(f) or {}
print(cfg.get('gpu_tier', ''))
" 2>/dev/null)
    if [ -n "$USER_TIER" ]; then
        GPU_TIER="$USER_TIER"
        BASE_CONFIG="Configs/config_${USER_TIER}_vram.yml"
        echo "Overridden by run_config.yaml gpu_tier: $GPU_TIER"
    fi
fi

echo "Selected: tier=$GPU_TIER  config=$BASE_CONFIG"

if [ ! -f "$BASE_CONFIG" ]; then
    echo "ERROR: Config not found: $BASE_CONFIG"
    exit 1
fi

# ---- Step 3: Detect speakers ----
echo ""
echo "=== Speaker Detection ==="

NUM_SPEAKERS=$(python -c "
import os
data_dir = '${DATA_DIR}'
speakers = sorted([
    d for d in os.listdir(data_dir)
    if os.path.isdir(os.path.join(data_dir, d))
    and os.path.isdir(os.path.join(data_dir, d, 'wav'))
    and os.path.isfile(os.path.join(data_dir, d, 'transcript_utf8.txt'))
])
print(len(speakers))
if speakers:
    print('Speakers: ' + ', '.join(speakers), file=__import__('sys').stderr)
" 2>&1 | tee /dev/stderr | head -1)

echo "Detected $NUM_SPEAKERS speaker(s)"

if [ "$NUM_SPEAKERS" -eq 0 ]; then
    echo "ERROR: No speaker directories found in ${DATA_DIR}/"
    echo "Expected structure: ${DATA_DIR}/speaker_name/wav/*.wav + transcript_utf8.txt"
    exit 1
fi

# ---- Step 4: Build final configs (base + user overrides) ----
echo ""
echo "=== Building Configs ==="

STAGE1_CONFIG="${DATA_DIR}/output/config_stage1.yml"
STAGE2_CONFIG="${DATA_DIR}/output/config_stage2.yml"

python merge_config.py \
    --base "$BASE_CONFIG" \
    --run-config "$RUN_CONFIG" \
    --output "$STAGE1_CONFIG" \
    --stage 1 \
    --num-speakers "$NUM_SPEAKERS"

python merge_config.py \
    --base "$BASE_CONFIG" \
    --run-config "$RUN_CONFIG" \
    --output "$STAGE2_CONFIG" \
    --stage 2 \
    --num-speakers "$NUM_SPEAKERS"

# ---- Step 5: Preprocess if needed ----
TRAIN_LIST="${DATA_DIR}/train_list.txt"
VAL_LIST="${DATA_DIR}/val_list.txt"

if [ -f "$TRAIN_LIST" ] && [ -f "$VAL_LIST" ]; then
    echo ""
    echo "train_list.txt and val_list.txt already exist, skipping phonemization."
else
    echo ""
    echo "=== Preprocessing: phonemize + WAV cache ($N_JOBS workers) ==="
    VAL_RATIO=$(python -c "
import yaml
try:
    with open('${RUN_CONFIG}') as f:
        cfg = yaml.safe_load(f) or {}
    print(cfg.get('val_ratio', 0.1))
except FileNotFoundError:
    print(0.1)
")
    MAX_DUR_FLAG=""
    MAX_DUR=$(python -c "
import yaml
try:
    with open('${RUN_CONFIG}') as f:
        cfg = yaml.safe_load(f) or {}
    v = cfg.get('max_duration', '')
    if v: print(v)
except FileNotFoundError:
    pass
" 2>/dev/null || true)
    if [ -n "${MAX_DUR:-}" ]; then
        MAX_DUR_FLAG="--max-duration $MAX_DUR"
    fi

    python preprocess_data.py \
        --data-dir "$DATA_DIR" \
        --val-ratio "$VAL_RATIO" \
        --n_jobs "$N_JOBS" \
        --cache-wavs --cache-dir "$CACHE_DIR" \
        $MAX_DUR_FLAG
fi

# ---- Step 6: Pre-warm wave cache (parallel) ----
echo ""
echo "=== Pre-warming wave cache ($N_JOBS workers) ==="

python -c "
import os, sys, numpy as np, soundfile as sf
from multiprocessing import Pool

cache_dir = '${CACHE_DIR}'
os.makedirs(cache_dir, exist_ok=True)

paths = set()
for list_file in ['${TRAIN_LIST}', '${VAL_LIST}']:
    with open(list_file) as f:
        for line in f:
            line = line.strip()
            if line:
                paths.add(line.split('|')[0])

to_cache = []
for p in sorted(paths):
    cache_key = p.replace(os.sep, '_').replace('/', '_')
    cache_file = os.path.join(cache_dir, cache_key + '.npy')
    if not os.path.exists(cache_file):
        to_cache.append(p)

already = len(paths) - len(to_cache)
print(f'Wave cache: {len(paths)} total, {already} already cached, {len(to_cache)} to process')

if to_cache:
    def cache_one(wav_path):
        import librosa
        cache_key = wav_path.replace(os.sep, '_').replace('/', '_')
        cache_file = os.path.join(cache_dir, cache_key + '.npy')
        try:
            wave, sr = sf.read(wav_path)
            if wave.ndim > 1: wave = wave[:, 0]
            if sr != 24000: wave = librosa.resample(wave, orig_sr=sr, target_sr=24000)
            wave = np.concatenate([np.zeros([5000]), wave, np.zeros([5000])], axis=0)
            np.save(cache_file, wave)
            return True
        except Exception as e:
            print(f'  FAILED: {wav_path}: {e}', file=sys.stderr)
            return False

    n_jobs = int('${N_JOBS}')
    with Pool(n_jobs) as pool:
        results = pool.map(cache_one, to_cache, chunksize=max(1, len(to_cache) // (n_jobs * 4)))
    ok = sum(results)
    print(f'  Cached {ok}/{len(to_cache)} new files')
else:
    print('  All files already cached')
"

# ---- Step 7: Run training ----
run_stage1() {
    echo ""
    echo "=== Stage 1: Acoustic Pre-Training ==="
    echo "  Config: $STAGE1_CONFIG (tier: $GPU_TIER)"
    accelerate launch train_first.py -p "$STAGE1_CONFIG"
}

run_stage2() {
    echo ""
    echo "=== Stage 2: Joint Fine-Tuning ==="
    echo "  Config: $STAGE2_CONFIG (tier: $GPU_TIER)"
    accelerate launch finetune_accelerate.py -p "$STAGE2_CONFIG"
}

case "$STAGE" in
    1)     run_stage1 ;;
    2)     run_stage2 ;;
    all)   run_stage1 && run_stage2 ;;
    *)     echo "Unknown stage: $STAGE"; exit 1 ;;
esac

echo ""
echo "=== Training complete at: $(date) ==="
