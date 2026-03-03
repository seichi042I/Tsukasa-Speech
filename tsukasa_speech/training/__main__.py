"""Entry point for ``python -m tsukasa_speech.training``.

All parameters fall back to environment variables for Docker compatibility.
"""
import argparse
import os

from tsukasa_speech.training.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Tsukasa Speech Training Pipeline")
    parser.add_argument("--data-dir",
                        default=os.environ.get("DATA_DIR", "Data"),
                        help="Data directory (default: $DATA_DIR or 'Data')")
    parser.add_argument("--stage",
                        default=os.environ.get("STAGE", ""),
                        help="Training stage: 1, 2, all (default: $STAGE or auto)")
    parser.add_argument("--cache-dir",
                        default=os.environ.get("TSUKASA_CACHE_DIR", ""),
                        help="WAV cache directory (default: $TSUKASA_CACHE_DIR or /tmp/wave_cache)")
    parser.add_argument("--n-jobs", type=int,
                        default=int(os.environ.get("N_JOBS", "4")),
                        help="Parallel workers (default: $N_JOBS or 4)")
    parser.add_argument("--gpu-tier",
                        default=os.environ.get("GPU_TIER_OVERRIDE", ""),
                        help="GPU tier override: low, mid, high (default: $GPU_TIER_OVERRIDE or auto)")
    args = parser.parse_args()

    run_pipeline(
        data_dir=args.data_dir,
        stage=args.stage,
        cache_dir=args.cache_dir,
        n_jobs=args.n_jobs,
        gpu_tier=args.gpu_tier,
    )


if __name__ == "__main__":
    main()
