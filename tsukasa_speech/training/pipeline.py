"""Training pipeline orchestrator.

Replaces the bash-based train.sh with a pure-Python pipeline that executes
all training steps: GPU detection, speaker discovery, config merging,
preprocessing, wave cache warming, preflight checks, and accelerate launch.

Usage (programmatic):
    from tsukasa_speech.training.pipeline import run_pipeline
    run_pipeline(data_dir="Data/SEICHI_MIX", stage="2")

Usage (CLI):
    python -m tsukasa_speech.training --data-dir Data/SEICHI_MIX --stage 2
"""

import os
import subprocess
import sys
from dataclasses import dataclass, field

import yaml


@dataclass
class PipelineConfig:
    """Configuration for the training pipeline.

    Paths are resolved lazily by resolve_paths() based on data_dir and gpu tier.
    """
    data_dir: str = "Data"
    stage: str = ""               # "1", "2", "all", "" (auto-detect)
    cache_dir: str = ""           # "" -> env TSUKASA_CACHE_DIR or /tmp/wave_cache
    n_jobs: int = 4
    gpu_tier_override: str = ""

    # Resolved by resolve_paths() / detect_gpu()
    run_config_path: str = ""
    base_config_path: str = ""
    stage1_config_path: str = ""
    stage2_config_path: str = ""
    train_list_path: str = ""
    val_list_path: str = ""
    output_dir: str = ""

    # Populated by pipeline steps
    gpu_tier: str = ""
    speakers: list = field(default_factory=list)
    run_config: dict = field(default_factory=dict)


def resolve_paths(cfg: PipelineConfig) -> None:
    """Resolve all derived paths from data_dir. Single source of truth for paths."""
    cfg.run_config_path = os.path.join(cfg.data_dir, "run_config.yaml")
    cfg.stage1_config_path = os.path.join(cfg.data_dir, "output", "config_stage1.yml")
    cfg.stage2_config_path = os.path.join(cfg.data_dir, "output", "config_stage2.yml")
    cfg.train_list_path = os.path.join(cfg.data_dir, "train_list.txt")
    cfg.val_list_path = os.path.join(cfg.data_dir, "val_list.txt")
    cfg.output_dir = os.path.join(cfg.data_dir, "output")

    if not cfg.cache_dir:
        cfg.cache_dir = os.environ.get("TSUKASA_CACHE_DIR", "/tmp/wave_cache")

    os.makedirs(cfg.cache_dir, exist_ok=True)
    os.makedirs(cfg.output_dir, exist_ok=True)


def _load_run_config(cfg: PipelineConfig) -> dict:
    """Load run_config.yaml if it exists, cache on cfg."""
    if cfg.run_config:
        return cfg.run_config
    if os.path.isfile(cfg.run_config_path):
        with open(cfg.run_config_path, 'r', encoding='utf-8') as f:
            cfg.run_config = yaml.safe_load(f) or {}
    return cfg.run_config


def determine_stage(cfg: PipelineConfig) -> str:
    """Determine training stage: CLI arg > env > run_config > default '2'."""
    if cfg.stage:
        print(f"Stage from argument: {cfg.stage}")
        return cfg.stage

    run_config = _load_run_config(cfg)
    if run_config.get('stage'):
        stage = str(run_config['stage'])
        print(f"Stage from run_config.yaml: {stage}")
        return stage

    print("No stage specified, using default: 2 (finetuning)")
    return "2"


def detect_gpu(cfg: PipelineConfig) -> tuple[str, str]:
    """Detect GPU tier and base config path.

    Priority: gpu_tier_override > run_config.yaml gpu_tier > auto-detect.
    Returns (tier, base_config_path).
    """
    from tsukasa_speech.config.gpu import detect_gpu_tier, TIER_CONFIGS, get_tier_description

    print()
    print("=== GPU Detection ===")

    tier, info = detect_gpu_tier()

    # Print GPU info
    if info['gpus']:
        for gpu in info['gpus']:
            print(f"  GPU {gpu['index']}: {gpu['name']} ({gpu['vram_mb']} MB)")
    else:
        print(f"  No GPU detected: {info.get('error', 'unknown')}")
    print(f"  Max VRAM: {info['max_vram_mb']} MB")
    print(f"  Auto-detected tier: {tier} - {get_tier_description(tier)}")

    # Override from CLI
    if cfg.gpu_tier_override:
        tier = cfg.gpu_tier_override
        print(f"  Overridden by --gpu-tier: {tier}")
    else:
        # Override from run_config
        run_config = _load_run_config(cfg)
        user_tier = run_config.get('gpu_tier', '')
        if user_tier:
            tier = str(user_tier)
            print(f"  Overridden by run_config.yaml gpu_tier: {tier}")

    base_config = TIER_CONFIGS.get(tier)
    if base_config is None:
        valid = ', '.join(TIER_CONFIGS.keys())
        print(f"ERROR: Unknown tier '{tier}'. Valid tiers: {valid}")
        sys.exit(1)

    print(f"  Selected: tier={tier}  config={base_config}")

    if not os.path.isfile(base_config):
        print(f"ERROR: Config not found: {base_config}")
        sys.exit(1)

    cfg.gpu_tier = tier
    cfg.base_config_path = base_config
    return tier, base_config


def discover_speakers_step(cfg: PipelineConfig) -> list[str]:
    """Discover speaker directories under data_dir."""
    from tsukasa_speech.preprocessing.phonemize_data import discover_speakers

    print()
    print("=== Speaker Detection ===")
    speakers = discover_speakers(cfg.data_dir)

    if not speakers:
        print(f"ERROR: No speaker directories found in {cfg.data_dir}/")
        print(f"Expected structure: {cfg.data_dir}/speaker_name/wav/*.wav + transcript_utf8.txt")
        sys.exit(1)

    print(f"Detected {len(speakers)} speaker(s): {', '.join(speakers)}")
    cfg.speakers = speakers
    return speakers


def build_configs(cfg: PipelineConfig) -> None:
    """Build stage configs by merging base config with user overrides."""
    from tsukasa_speech.config.merge import merge_training_config

    print()
    print("=== Building Configs ===")
    num_speakers = len(cfg.speakers)

    for stage in [1, 2]:
        output_path = cfg.stage1_config_path if stage == 1 else cfg.stage2_config_path
        merge_training_config(
            base_config_path=cfg.base_config_path,
            run_config_path=cfg.run_config_path,
            output_path=output_path,
            stage=stage,
            num_speakers=num_speakers,
            data_dir=cfg.data_dir,
        )


def preprocess_if_needed(cfg: PipelineConfig) -> None:
    """Run phonemization + WAV caching if train/val lists don't exist yet."""
    if os.path.isfile(cfg.train_list_path) and os.path.isfile(cfg.val_list_path):
        print()
        print("train_list.txt and val_list.txt already exist, skipping phonemization.")
        return

    print()
    print(f"=== Preprocessing: phonemize + WAV cache ({cfg.n_jobs} workers) ===")

    run_config = _load_run_config(cfg)
    val_ratio = run_config.get('val_ratio', 0.1)
    max_duration = run_config.get('max_duration', None)

    cmd = [
        sys.executable, "-m", "tsukasa_speech.preprocessing.phonemize_data",
        "--data-dir", cfg.data_dir,
        "--val-ratio", str(val_ratio),
        "--n_jobs", str(cfg.n_jobs),
        "--cache-wavs", "--cache-dir", cfg.cache_dir,
    ]
    if max_duration:
        cmd.extend(["--max-duration", str(max_duration)])

    subprocess.check_call(cmd)


def warm_wave_cache(cfg: PipelineConfig) -> None:
    """Pre-warm wave cache for all files in train/val lists."""
    from tsukasa_speech.preprocessing.phonemize_data import warm_cache_from_lists

    print()
    print(f"=== Pre-warming wave cache ({cfg.n_jobs} workers) ===")
    warm_cache_from_lists(
        train_list_path=cfg.train_list_path,
        val_list_path=cfg.val_list_path,
        cache_dir=cfg.cache_dir,
        n_jobs=cfg.n_jobs,
    )


def preflight_check(cfg: PipelineConfig) -> None:
    """Verify required model weights and files exist before training."""
    from tsukasa_speech.utils.download import check_models

    print()
    print("=== Preflight Check ===")
    missing = check_models()
    if missing:
        print(f"WARNING: {len(missing)} model weight(s) missing:")
        for f in missing:
            print(f"  - {f}")
        print("Training may fail. Run 'python download_models.py' to fetch them.")
    else:
        print("  All model weights present.")

    # Check that train/val lists exist (should exist after preprocess step)
    for path, label in [(cfg.train_list_path, "train_list.txt"),
                        (cfg.val_list_path, "val_list.txt")]:
        if not os.path.isfile(path):
            print(f"ERROR: {label} not found at {path}")
            sys.exit(1)

    print("  Preflight OK.")


def _launch_accelerate(config_path: str, script: str) -> None:
    """Launch a training script via accelerate."""
    cmd = ["accelerate", "launch", script, "-p", config_path]
    print(f"  $ {' '.join(cmd)}")
    subprocess.check_call(cmd)


def run_pipeline(
    data_dir: str = "Data",
    stage: str = "",
    cache_dir: str = "",
    n_jobs: int = 4,
    gpu_tier: str = "",
) -> None:
    """Run the full training pipeline.

    This is the main entry point, replacing the bash train.sh script.
    """
    cfg = PipelineConfig(
        data_dir=data_dir,
        stage=stage,
        cache_dir=cache_dir,
        n_jobs=n_jobs,
        gpu_tier_override=gpu_tier,
    )

    # Step 0: Resolve paths
    resolve_paths(cfg)

    # Step 1: Determine stage
    cfg.stage = determine_stage(cfg)

    # Step 2: Detect GPU tier
    detect_gpu(cfg)

    # Step 3: Discover speakers
    discover_speakers_step(cfg)

    # Step 4: Build merged configs
    build_configs(cfg)

    # Step 5: Preprocess if needed
    preprocess_if_needed(cfg)

    # Step 6: Warm wave cache
    warm_wave_cache(cfg)

    # Step 7: Preflight check
    preflight_check(cfg)

    # Step 8: Run training
    if cfg.stage == "1":
        print()
        print(f"=== Stage 1: Acoustic Pre-Training ===")
        print(f"  Config: {cfg.stage1_config_path} (tier: {cfg.gpu_tier})")
        _launch_accelerate(cfg.stage1_config_path, "train_first.py")

    elif cfg.stage == "2":
        print()
        print(f"=== Stage 2: Joint Fine-Tuning ===")
        print(f"  Config: {cfg.stage2_config_path} (tier: {cfg.gpu_tier})")
        _launch_accelerate(cfg.stage2_config_path, "finetune_accelerate.py")

    elif cfg.stage == "all":
        print()
        print(f"=== Stage 1: Acoustic Pre-Training ===")
        print(f"  Config: {cfg.stage1_config_path} (tier: {cfg.gpu_tier})")
        _launch_accelerate(cfg.stage1_config_path, "train_first.py")

        print()
        print(f"=== Stage 2: Joint Fine-Tuning ===")
        print(f"  Config: {cfg.stage2_config_path} (tier: {cfg.gpu_tier})")
        _launch_accelerate(cfg.stage2_config_path, "finetune_accelerate.py")

    else:
        print(f"ERROR: Unknown stage: {cfg.stage}")
        sys.exit(1)
