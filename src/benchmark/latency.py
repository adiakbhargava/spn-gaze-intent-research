"""
Latency Benchmarking & Comparison Module (Module 5)

Compares latency characteristics between:
1. BCI2000 clinical system (Roundtrip from .dat files)
2. This fusion pipeline's real-time inference latency

The comparison demonstrates that a Python-based fusion pipeline can achieve
latencies competitive with the established BCI2000 C++ platform, while
adding the multimodal gaze+EEG fusion that BCI2000 doesn't natively support.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from src.data.bci2000_parser import BCI2000Run, compute_latency_stats, parse_dat_file
from src.streaming.realtime_inference import LatencyTracker

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Complete benchmark comparison result."""
    bci2000_stats: dict
    fusion_stats: dict
    comparison: dict
    metadata: dict


def benchmark_bci2000_latency(
    dat_files: list[str | Path],
) -> dict:
    """
    Compute latency statistics from BCI2000 .dat files.

    Uses inter-block intervals (SourceTime deltas with 16-bit wrap handling)
    as the primary timing metric.  The raw Roundtrip state is also recorded
    but is a 16-bit timestamp in BCI2000 v1.1, not a latency delta.

    Args:
        dat_files: List of paths to BCI2000 .dat files

    Returns:
        Aggregated latency statistics
    """
    all_block_intervals = []
    all_roundtrips_raw = []
    per_file_stats = []

    for filepath in dat_files:
        try:
            run = parse_dat_file(filepath)
            stats = compute_latency_stats(run)
            per_file_stats.append(stats)

            # Block intervals (the meaningful timing metric)
            intervals = run.get_block_interval_ms()
            if intervals is not None:
                valid = intervals[~np.isnan(intervals)]
                all_block_intervals.extend(valid.tolist())

            # Raw roundtrip (16-bit timestamp — kept for reference)
            rt = run.get_roundtrip_ms()
            if rt is not None:
                valid = rt[rt > 0]
                all_roundtrips_raw.extend(valid.tolist())

        except Exception as e:
            logger.warning(f"Could not process {filepath}: {e}")

    result = {
        "n_files": len(dat_files),
        "n_files_processed": len(per_file_stats),
        "per_file": per_file_stats,
    }

    # Primary metric: inter-block processing interval
    if all_block_intervals:
        bi = np.array(all_block_intervals)
        result["block_interval_aggregate"] = {
            "mean_ms": float(np.mean(bi)),
            "median_ms": float(np.median(bi)),
            "std_ms": float(np.std(bi)),
            "min_ms": float(np.min(bi)),
            "max_ms": float(np.max(bi)),
            "p95_ms": float(np.percentile(bi, 95)),
            "p99_ms": float(np.percentile(bi, 99)),
            "n_intervals": len(bi),
        }
        # Also provide as "roundtrip_aggregate" for backward compatibility
        # with compare_latencies() which uses this key
        result["roundtrip_aggregate"] = result["block_interval_aggregate"].copy()
        result["roundtrip_aggregate"]["n_blocks"] = len(bi)

    # Raw roundtrip values (kept for reference — these are timestamps, not latencies)
    if all_roundtrips_raw:
        rt = np.array(all_roundtrips_raw)
        result["roundtrip_raw"] = {
            "note": "Raw 16-bit Roundtrip state values (timestamps, NOT latencies)",
            "mean": float(np.mean(rt)),
            "median": float(np.median(rt)),
            "min": float(np.min(rt)),
            "max": float(np.max(rt)),
        }

    return result


def benchmark_fusion_pipeline(
    model,
    n_eeg_channels: int = 128,
    n_samples: int = 500,
    sfreq: float = 500.0,
    n_iterations: int = 200,
    eeg_method: str = "welch",
    channel_groups: dict[str, list[int]] | None = None,
) -> dict:
    """
    Benchmark the fusion pipeline's inference latency.

    Runs the full pipeline (feature extraction + classification) on
    synthetic data to measure timing characteristics.

    Args:
        model: Trained classifier
        n_eeg_channels: Number of EEG channels
        n_samples: Samples per window
        sfreq: Sampling frequency
        n_iterations: Number of benchmark iterations
        eeg_method: Feature extraction method
        channel_groups: EEG channel group indices for regional aggregation.
            If None, auto-resolves from standard 128-channel montage.

    Returns:
        Latency statistics
    """
    from src.features.eeg_features import extract_eeg_features
    from src.features.gaze_features import extract_gaze_features

    # Auto-resolve channel groups if not provided
    if channel_groups is None:
        from src.data.eegeyenet_loader import _get_128_channel_names
        from src.features.feature_pipeline import resolve_channel_groups
        ch_names = _get_128_channel_names()[:n_eeg_channels]
        channel_groups = resolve_channel_groups(ch_names) or None

    rng = np.random.RandomState(42)
    total_latencies = []
    feature_latencies = []
    model_latencies = []

    # Warm up
    eeg = rng.randn(n_eeg_channels, n_samples).astype(np.float64) * 20
    gaze = rng.randn(n_samples, 3).astype(np.float64)
    eeg_feats = extract_eeg_features(eeg, sfreq, method=eeg_method,
                                      channel_groups=channel_groups)
    gaze_feats = extract_gaze_features(gaze, sfreq=sfreq)
    features = np.concatenate([eeg_feats, gaze_feats]).reshape(1, -1)
    features = np.nan_to_num(features)
    if hasattr(model, 'predict_proba'):
        model.predict_proba(features)

    for i in range(n_iterations):
        eeg = rng.randn(n_eeg_channels, n_samples).astype(np.float64) * 20
        gaze = rng.randn(n_samples, 3).astype(np.float64)

        t0 = time.perf_counter()

        # Feature extraction
        t_feat = time.perf_counter()
        eeg_feats = extract_eeg_features(eeg, sfreq, method=eeg_method,
                                          channel_groups=channel_groups)
        gaze_feats = extract_gaze_features(gaze, sfreq=sfreq)
        features = np.concatenate([eeg_feats, gaze_feats]).reshape(1, -1)
        features = np.nan_to_num(features)
        t_feat_end = time.perf_counter()

        # Classification
        t_model = time.perf_counter()
        if hasattr(model, 'predict_proba'):
            model.predict_proba(features)
        t_model_end = time.perf_counter()

        total_latencies.append((t_model_end - t0) * 1000)
        feature_latencies.append((t_feat_end - t_feat) * 1000)
        model_latencies.append((t_model_end - t_model) * 1000)

    total = np.array(total_latencies)
    feat = np.array(feature_latencies)
    model_lat = np.array(model_latencies)

    return {
        "total": {
            "mean_ms": float(np.mean(total)),
            "median_ms": float(np.median(total)),
            "std_ms": float(np.std(total)),
            "min_ms": float(np.min(total)),
            "max_ms": float(np.max(total)),
            "p95_ms": float(np.percentile(total, 95)),
            "p99_ms": float(np.percentile(total, 99)),
        },
        "feature_extraction": {
            "mean_ms": float(np.mean(feat)),
            "median_ms": float(np.median(feat)),
            "p95_ms": float(np.percentile(feat, 95)),
        },
        "model_inference": {
            "mean_ms": float(np.mean(model_lat)),
            "median_ms": float(np.median(model_lat)),
            "p95_ms": float(np.percentile(model_lat, 95)),
        },
        "n_iterations": n_iterations,
        "config": {
            "n_eeg_channels": n_eeg_channels,
            "n_samples": n_samples,
            "sfreq": sfreq,
            "eeg_method": eeg_method,
        },
    }


def compare_latencies(
    bci2000_stats: dict,
    fusion_stats: dict,
) -> dict:
    """
    Generate a comparison between BCI2000 and fusion pipeline latencies.

    Compares the BCI2000 inter-block processing interval (from SourceTime
    deltas) against the fusion pipeline's per-window inference latency.

    Returns:
        Comparison dictionary with analysis
    """
    comparison = {}

    # Use block_interval_aggregate (primary) or roundtrip_aggregate (compat)
    bci_timing = (bci2000_stats.get("block_interval_aggregate")
                  or bci2000_stats.get("roundtrip_aggregate", {}))
    fusion_total = fusion_stats.get("total", {})

    if bci_timing and fusion_total:
        comparison["bci2000_block_interval_median_ms"] = bci_timing.get("median_ms", 0)
        comparison["fusion_median_ms"] = fusion_total.get("median_ms", 0)
        comparison["bci2000_block_interval_p95_ms"] = bci_timing.get("p95_ms", 0)
        comparison["fusion_p95_ms"] = fusion_total.get("p95_ms", 0)

        bci_median = bci_timing.get("median_ms", 0)
        fusion_median = fusion_total.get("median_ms", 0)
        if bci_median > 0:
            ratio = fusion_median / bci_median
            comparison["median_ratio"] = ratio
            if ratio < 1:
                comparison["verdict"] = (
                    f"Fusion pipeline inference ({fusion_median:.1f}ms) completes "
                    f"well within one BCI2000 block interval ({bci_median:.1f}ms), "
                    f"confirming real-time viability — {1/ratio:.1f}x headroom"
                )
            else:
                comparison["verdict"] = (
                    f"Fusion pipeline ({fusion_median:.1f}ms) takes "
                    f"{ratio:.1f}x longer than one BCI2000 block interval "
                    f"({bci_median:.1f}ms); consider reducing channels or "
                    f"using a faster feature extraction method"
                )

        # Where does the time go?
        feat_ms = fusion_stats.get("feature_extraction", {}).get("mean_ms", 0)
        model_ms = fusion_stats.get("model_inference", {}).get("mean_ms", 0)
        total_ms = fusion_total.get("mean_ms", 1)
        comparison["time_breakdown"] = {
            "feature_extraction_pct": feat_ms / total_ms * 100,
            "model_inference_pct": model_ms / total_ms * 100,
            "overhead_pct": (total_ms - feat_ms - model_ms) / total_ms * 100,
        }

    # Consumer hardware implications
    comparison["consumer_hardware_notes"] = {
        "channel_reduction": (
            "Consumer headsets (4-8 channels vs. 128 research-grade) would reduce "
            "feature extraction time ~10-20x but lower signal quality"
        ),
        "wireless_latency": (
            "Bluetooth EEG adds 10-50ms transport latency; must be factored into "
            "total system latency budget"
        ),
        "snr_impact": (
            "Lower SNR from fewer/dry electrodes requires more robust classification; "
            "fusion with eye-tracking becomes even more critical"
        ),
    }

    return comparison


def generate_comparison_report(
    bci2000_stats: dict,
    fusion_stats: dict,
    comparison: dict,
    output_path: Optional[str | Path] = None,
) -> str:
    """
    Generate a formatted comparison report.

    Args:
        bci2000_stats: BCI2000 latency statistics
        fusion_stats: Fusion pipeline latency statistics
        comparison: Comparison analysis
        output_path: Optional path to save the report

    Returns:
        Formatted report string
    """
    lines = []
    lines.append("=" * 72)
    lines.append("LATENCY COMPARISON REPORT")
    lines.append("BCI2000 Clinical System vs. EEG+Gaze Fusion Pipeline")
    lines.append("=" * 72)
    lines.append("")

    # BCI2000 Section
    lines.append("BCI2000 Block Processing Interval")
    lines.append("-" * 40)
    bci_bi = (bci2000_stats.get("block_interval_aggregate")
              or bci2000_stats.get("roundtrip_aggregate", {}))
    if bci_bi:
        lines.append(f"  Mean:       {bci_bi.get('mean_ms', 0):>8.1f} ms")
        lines.append(f"  Median:     {bci_bi.get('median_ms', 0):>8.1f} ms")
        lines.append(f"  Std/Jitter: {bci_bi.get('std_ms', 0):>8.1f} ms")
        lines.append(f"  P95:        {bci_bi.get('p95_ms', 0):>8.1f} ms")
        lines.append(f"  P99:        {bci_bi.get('p99_ms', 0):>8.1f} ms")
        lines.append(f"  Min:        {bci_bi.get('min_ms', 0):>8.1f} ms")
        lines.append(f"  Max:        {bci_bi.get('max_ms', 0):>8.1f} ms")
        n = bci_bi.get("n_intervals") or bci_bi.get("n_blocks", 0)
        lines.append(f"  Intervals:  {n:>8d}")
        lines.append(f"  N files:    {bci2000_stats.get('n_files_processed', 0):>8d}")
    else:
        lines.append("  No BCI2000 data available")
    lines.append("")

    # Fusion Pipeline Section
    lines.append("Fusion Pipeline Inference Latency")
    lines.append("-" * 40)
    fusion_total = fusion_stats.get("total", {})
    if fusion_total:
        lines.append(f"  Mean:    {fusion_total.get('mean_ms', 0):>8.1f} ms")
        lines.append(f"  Median:  {fusion_total.get('median_ms', 0):>8.1f} ms")
        lines.append(f"  Std:     {fusion_total.get('std_ms', 0):>8.1f} ms")
        lines.append(f"  P95:     {fusion_total.get('p95_ms', 0):>8.1f} ms")
        lines.append(f"  P99:     {fusion_total.get('p99_ms', 0):>8.1f} ms")
    lines.append("")

    feat = fusion_stats.get("feature_extraction", {})
    model = fusion_stats.get("model_inference", {})
    lines.append("  Time Breakdown:")
    lines.append(f"    Feature Extraction: {feat.get('mean_ms', 0):>6.1f} ms (mean)")
    lines.append(f"    Model Inference:    {model.get('mean_ms', 0):>6.1f} ms (mean)")
    lines.append("")

    # Comparison
    lines.append("Comparison")
    lines.append("-" * 40)
    if "verdict" in comparison:
        lines.append(f"  {comparison['verdict']}")
    if "time_breakdown" in comparison:
        tb = comparison["time_breakdown"]
        lines.append(f"\n  Time Budget:")
        lines.append(f"    Feature extraction: {tb.get('feature_extraction_pct', 0):.1f}%")
        lines.append(f"    Model inference:    {tb.get('model_inference_pct', 0):.1f}%")
        lines.append(f"    Overhead:           {tb.get('overhead_pct', 0):.1f}%")
    lines.append("")

    # Consumer hardware notes
    lines.append("Consumer Hardware Implications")
    lines.append("-" * 40)
    notes = comparison.get("consumer_hardware_notes", {})
    for key, note in notes.items():
        lines.append(f"  [{key}]")
        lines.append(f"    {note}")
        lines.append("")

    lines.append("=" * 72)
    report = "\n".join(lines)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report)
        logger.info(f"Report saved to {output_path}")

    return report
