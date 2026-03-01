#!/usr/bin/env python3
"""Download pretrained model weights if not present.

Usage:
  # Check if all models exist
  python download_models.py --check

  # Download from HuggingFace repo
  MODEL_REPO=username/tsukasa-speech-models python download_models.py

  # Or pass repo as argument
  python download_models.py --repo username/tsukasa-speech-models

Environment variables:
  MODEL_REPO  : HuggingFace repo ID (e.g. "username/tsukasa-speech-models")
  HF_TOKEN    : HuggingFace token for private repos (optional)
"""
import argparse
import os
import sys

# Model weight files required for training (relative to /app)
REQUIRED_MODELS = [
    "Models/Style_Tsukasa_v02/Top_ckpt_24khz.pth",  # 2.0GB - pretrained checkpoint
    "Utils/ASR/bst_00080.pth",                        # 91MB  - ASR model
    "Utils/JDC/bst.t7",                                # 21MB  - F0 pitch extractor
    "Utils/PLBERT/step_1050000.t7",                    # 1.8GB - PL-BERT
]


def check_models(base_dir="."):
    """Return list of missing model files."""
    missing = []
    for path in REQUIRED_MODELS:
        full = os.path.join(base_dir, path)
        if not os.path.isfile(full):
            missing.append(path)
    return missing


def download_from_hf(repo_id, base_dir=".", token=None):
    """Download model weights from a HuggingFace repo.

    Expected repo structure (mirrors local paths):
      Models/Style_Tsukasa_v02/Top_ckpt_24khz.pth
      Utils/ASR/bst_00080.pth
      Utils/JDC/bst.t7
      Utils/PLBERT/step_1050000.t7
    """
    from huggingface_hub import hf_hub_download

    for path in REQUIRED_MODELS:
        dest = os.path.join(base_dir, path)
        if os.path.isfile(dest):
            print(f"  [skip] {path}")
            continue

        os.makedirs(os.path.dirname(dest), exist_ok=True)
        print(f"  [download] {path} ...")
        hf_hub_download(
            repo_id=repo_id,
            filename=path,
            local_dir=base_dir,
            token=token,
        )
        print(f"  [done] {path}")


def ensure_hf_models_cached(token=None):
    """Pre-cache HuggingFace models used during training (Whisper SLM).

    These are loaded via transformers from_pretrained() during training.
    Pre-caching avoids network access at training time.
    """
    from huggingface_hub import snapshot_download

    hf_models = [
        "openai/whisper-large-v2",                    # ~6GB - SLM encoder weights
        "Respair/Whisper_Large_v2_Encoder_Block",     # config only
    ]
    for repo_id in hf_models:
        print(f"  [cache] {repo_id} ...")
        try:
            snapshot_download(repo_id, token=token)
            print(f"  [ok]    {repo_id}")
        except Exception as e:
            print(f"  [warn]  {repo_id}: {e}")
            print(f"          (SLM loss requires this model. Set lambda_slm: 0 to skip.)")


def main():
    parser = argparse.ArgumentParser(description="Download pretrained models")
    parser.add_argument("--check", action="store_true", help="Only check, don't download")
    parser.add_argument("--repo", default=os.environ.get("MODEL_REPO", "Respair/Tsukasa_Speech"), help="HuggingFace repo ID")
    parser.add_argument("--base-dir", default=".", help="Base directory (default: cwd)")
    args = parser.parse_args()

    missing = check_models(args.base_dir)

    if not missing:
        print("All model weights present.")
    else:
        print(f"Missing {len(missing)} model file(s):")
        for f in missing:
            print(f"  - {f}")

        if args.check:
            return 1

        repo_id = args.repo
        if not repo_id:
            print()
            print("ERROR: Models not found and MODEL_REPO not set.")
            print("Options:")
            print("  1. Set MODEL_REPO env var to your HuggingFace repo ID")
            print("  2. Mount models via volume (e.g. -v /path/to/models:/app/Models)")
            print("  3. Place model files manually in the expected paths")
            return 1

        token = os.environ.get("HF_TOKEN")
        print(f"\nDownloading from HuggingFace: {repo_id}")
        download_from_hf(repo_id, args.base_dir, token=token)

        # Verify
        still_missing = check_models(args.base_dir)
        if still_missing:
            print(f"\nERROR: Still missing after download:")
            for f in still_missing:
                print(f"  - {f}")
            return 1

        print("\nAll models downloaded successfully.")

    # Pre-cache HuggingFace models (Whisper for SLM loss)
    if not args.check:
        print("\n=== Pre-caching HuggingFace models ===")
        ensure_hf_models_cached(token=os.environ.get("HF_TOKEN"))

    return 0


if __name__ == "__main__":
    sys.exit(main())
