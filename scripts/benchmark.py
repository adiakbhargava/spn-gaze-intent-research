#!/usr/bin/env python3
"""
Latency Benchmark Script

Benchmarks the fusion pipeline against BCI2000 clinical data and generates
the comparison report (Module 5).

The BCI2000 benchmark only uses system timing states (Roundtrip, SourceTime)
which are paradigm-independent — any BCI2000 .dat file works regardless of
task type (motor imagery, P300, TDU/FES, etc.). No eye-tracking data is needed.

Usage:
    # Full benchmark with synthetic BCI2000 data
    python scripts/benchmark.py

    # Benchmark with real BCI2000 .dat files (any paradigm)
    python scripts/benchmark.py --bci2000-dir data/raw/bci2000

    # Quick benchmark (fewer iterations)
    python scripts/benchmark.py --n-iterations 50
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.bci2000_parser import generate_synthetic_bci2000_dat
from src.data.eegeyenet_loader import generate_synthetic_dataset
from src.features.feature_pipeline import FeatureMode, extract_dataset_features
from src.models.baseline import build_random_forest
from src.benchmark.latency import (
    benchmark_bci2000_latency,
    benchmark_fusion_pipeline,
    compare_latencies,
    generate_comparison_report,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output")


def load_or_train_model(model_path: str | None = None):
    """Load a saved model or train a quick one."""
    if model_path and Path(model_path).exists():
        logger.info(f"Loading model from {model_path}")
        with open(model_path, "rb") as f:
            return pickle.load(f)

    logger.info("Training quick benchmark model...")
    dataset = generate_synthetic_dataset(n_subjects=5, trials_per_subject=40)
    X, y, _ = extract_dataset_features(dataset, mode=FeatureMode.FUSED)
    X = np.nan_to_num(X)

    model = build_random_forest(n_estimators=50, max_depth=6)
    model.fit(X, y)
    return model


def main():
    parser = argparse.ArgumentParser(description="Run latency benchmarks")
    parser.add_argument("--model", type=str, default="models/saved/best_model.pkl",
                       help="Path to trained model")
    parser.add_argument("--bci2000-dir", type=str, default=None,
                       help="Directory containing BCI2000 .dat files (any paradigm — "
                            "only system timing states are used, no eye-tracking needed)")
    parser.add_argument("--n-iterations", type=int, default=200,
                       help="Number of benchmark iterations for fusion pipeline")
    parser.add_argument("--eeg-method", default="welch",
                       choices=["welch", "ar", "fft"],
                       help="Feature extraction method to benchmark")
    parser.add_argument("--clean", action="store_true",
                       help="Clear output and synthetic data before running")
    args = parser.parse_args()

    if args.clean:
        import shutil
        for d in [OUTPUT_DIR, Path("data/synthetic/bci2000")]:
            if d.exists():
                shutil.rmtree(d)
                logger.info(f"Removed {d}/")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_or_train_model(args.model)

    # --- BCI2000 Benchmark ---
    logger.info("\n" + "=" * 60)
    logger.info("BCI2000 LATENCY ANALYSIS")
    logger.info("=" * 60)

    if args.bci2000_dir:
        dat_dir = Path(args.bci2000_dir)
        if not dat_dir.exists():
            logger.error(f"BCI2000 directory not found: {dat_dir}")
            logger.error("On Windows Git Bash, use forward slashes:")
            logger.error('  python scripts/benchmark.py --bci2000-dir "C:/AAKB/Research/CI_999_06/Cursor001"')
            sys.exit(1)

        # Search for .dat files — first in directory, then recursively
        dat_files = sorted(dat_dir.glob("*.dat"))
        if not dat_files:
            dat_files = sorted(dat_dir.rglob("*.dat"))
            if dat_files:
                logger.info(f"Found {len(dat_files)} .dat files in subdirectories of {dat_dir}")

        if not dat_files:
            logger.error(f"No .dat files found in {dat_dir} (or subdirectories)")
            logger.error(f"Contents of {dat_dir}:")
            for item in sorted(dat_dir.iterdir()):
                logger.error(f"  {'[dir]' if item.is_dir() else '[file]'} {item.name}")
            sys.exit(1)

        logger.info(f"Found {len(dat_files)} .dat files:")
        for f in dat_files:
            logger.info(f"  {f}")
    else:
        # Generate synthetic .dat files
        logger.info("Generating synthetic BCI2000 .dat files...")
        dat_dir = Path("data/synthetic/bci2000")
        dat_dir.mkdir(parents=True, exist_ok=True)
        dat_files = []
        for i in range(3):
            fp = generate_synthetic_bci2000_dat(
                dat_dir / f"run_{i:02d}.dat",
                n_channels=16,
                n_blocks=500,
                sampling_rate=256.0,
            )
            dat_files.append(fp)

    bci2000_stats = benchmark_bci2000_latency(dat_files)

    bci_rt = bci2000_stats.get("roundtrip_aggregate", {})
    if bci_rt:
        logger.info(f"BCI2000 Roundtrip: mean={bci_rt['mean_ms']:.1f}ms, "
                    f"median={bci_rt['median_ms']:.1f}ms, "
                    f"p95={bci_rt['p95_ms']:.1f}ms")

    # --- Fusion Pipeline Benchmark ---
    logger.info("\n" + "=" * 60)
    logger.info("FUSION PIPELINE LATENCY BENCHMARK")
    logger.info("=" * 60)

    fusion_stats = benchmark_fusion_pipeline(
        model=model,
        n_iterations=args.n_iterations,
        eeg_method=args.eeg_method,
    )

    fusion_total = fusion_stats.get("total", {})
    logger.info(f"Fusion Pipeline: mean={fusion_total['mean_ms']:.1f}ms, "
                f"median={fusion_total['median_ms']:.1f}ms, "
                f"p95={fusion_total['p95_ms']:.1f}ms")

    feat = fusion_stats.get("feature_extraction", {})
    model_lat = fusion_stats.get("model_inference", {})
    logger.info(f"  Feature extraction: {feat['mean_ms']:.1f}ms mean")
    logger.info(f"  Model inference:    {model_lat['mean_ms']:.1f}ms mean")

    # --- Comparison ---
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON")
    logger.info("=" * 60)

    comparison = compare_latencies(bci2000_stats, fusion_stats)
    if "verdict" in comparison:
        logger.info(comparison["verdict"])

    # Generate report
    report = generate_comparison_report(
        bci2000_stats, fusion_stats, comparison,
        output_path=OUTPUT_DIR / "latency_comparison.txt",
    )
    logger.info(f"\nFull report saved to {OUTPUT_DIR / 'latency_comparison.txt'}")

    # Save raw data as JSON
    with open(OUTPUT_DIR / "benchmark_data.json", "w") as f:
        json.dump({
            "bci2000": bci2000_stats,
            "fusion": fusion_stats,
            "comparison": comparison,
        }, f, indent=2, default=str)
    logger.info(f"Raw data saved to {OUTPUT_DIR / 'benchmark_data.json'}")

    # Print the report
    print("\n" + report)


if __name__ == "__main__":
    main()
