#!/usr/bin/env python3
"""
Preprocess real EEGEyeNet Position Task data into the pipeline's expected format.

IMPORTANT — DATA LIMITATION:
The EEGEyeNet benchmark NPZ files contain only EEG data and target position
labels.  They do NOT contain eye-tracking time series.  The raw per-subject
.mat files that included synchronized gaze recordings were hosted on a linked
Dropbox storage provider on OSF (osf.io/ktv7m) which is no longer available.

Because of this, real EEGEyeNet data can only be used for EEG-only evaluation
(validating that the EEG feature pipeline works on genuine neural signals).
Gaze features and the full EEG+gaze fusion ablation are validated on synthetic
data, which correctly models the intent-vs-observe Midas touch problem with
calibrated difficulty matching BCI literature.

The Position task is a gaze-prediction regression benchmark (subjects saccade
to dots on screen).  It does NOT have distinct "intent" and "observe"
conditions — every trial is a directed saccade.  Any binary label split
(e.g., median eccentricity) is artificial and does not represent the
cognitive intent distinction that the pipeline classifies.

Supported input:
    EEGEyeNet OSF .npz files (e.g., Position_task_with_dots_synchronised_max.npz)
    Contains: EEG (N, 500, 129) + labels (N, 3) [subject_id, target_x, target_y]

Usage:
    # Inspect NPZ contents
    python scripts/preprocess_real_data.py --info --npz data.npz

    # Preprocess for EEG-only evaluation
    python scripts/preprocess_real_data.py --npz data.npz
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def inspect_npz(npz_path: str):
    """Print all keys, shapes, and dtypes in an NPZ file without processing."""
    npz_path = Path(npz_path)
    logger.info(f"Inspecting: {npz_path}")
    logger.info(f"File size: {npz_path.stat().st_size / 1e6:.1f} MB")

    data = np.load(str(npz_path), allow_pickle=True)
    logger.info(f"\nKeys ({len(data.files)}):")
    for key in sorted(data.files):
        arr = data[key]
        logger.info(f"  {key:30s}  shape={str(arr.shape):20s}  dtype={arr.dtype}")

    # Check for eye-tracking data
    et_keys = ["ET", "eye_tracking", "et_data", "gaze", "eyetrack"]
    et_found = [k for k in data.files if k.lower() in [e.lower() for e in et_keys]]
    if et_found:
        logger.info(f"\nEye-tracking data detected: {et_found}")
    else:
        logger.info("\nNo eye-tracking time series found.")
        logger.info("This NPZ can only be used for EEG-only evaluation.")


def preprocess(
    npz_path: str,
    output_dir: str,
    max_subjects: int | None = None,
    max_trials_per_subject: int = 80,
):
    """
    Convert Position task NPZ to pipeline format (EEG-only).

    Extracts real 128-channel EEG data and subject IDs.  Labels are created
    via median split on target eccentricity for compatibility with the
    pipeline's binary classification interface, but these labels do NOT
    represent cognitive intent — they are an artificial spatial split.

    Gaze data is NOT included.  Use synthetic data for fusion evaluation.
    """
    npz_path = Path(npz_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading {npz_path} ...")
    data = np.load(str(npz_path), allow_pickle=True)

    # Log available keys
    logger.info(f"NPZ keys: {data.files}")
    for key in data.files:
        logger.info(f"  {key}: shape={data[key].shape}, dtype={data[key].dtype}")

    eeg_raw = data["EEG"]       # (N, 500, 129)
    labels_raw = data["labels"]  # (N, 3) -> [subject_id, target_x, target_y]

    n_total = eeg_raw.shape[0]
    n_samples = eeg_raw.shape[1]
    n_channels = eeg_raw.shape[2] - 1  # drop reference channel

    logger.info(f"Raw data: {n_total} trials, {n_samples} samples, {n_channels + 1} channels")

    subject_ids_raw = labels_raw[:, 0].astype(int)
    target_x = labels_raw[:, 1]
    target_y = labels_raw[:, 2]
    unique_subjects = np.unique(subject_ids_raw)

    if max_subjects is not None:
        unique_subjects = unique_subjects[:max_subjects]
    logger.info(f"Using {len(unique_subjects)} subjects")

    # Screen center estimate (median of all target positions)
    cx = np.median(target_x)
    cy = np.median(target_y)
    logger.info(f"Estimated screen center: ({cx:.1f}, {cy:.1f})")

    # Eccentricity-based binary labels (artificial split for pipeline compat)
    eccentricity = np.sqrt((target_x - cx) ** 2 + (target_y - cy) ** 2)
    median_ecc = np.median(eccentricity)
    binary_labels = (eccentricity <= median_ecc).astype(int)

    logger.info(f"Median eccentricity: {median_ecc:.1f} px")
    logger.info(
        "NOTE: Labels are an artificial spatial split (near/far target), "
        "NOT cognitive intent vs. observe."
    )

    # Collect trials with balanced classes per subject
    rng = np.random.RandomState(42)
    selected_indices = []

    for subj in unique_subjects:
        mask = subject_ids_raw == subj
        subj_indices = np.where(mask)[0]
        subj_labels = binary_labels[subj_indices]

        # Balance classes
        intent_idx = subj_indices[subj_labels == 1]
        observe_idx = subj_indices[subj_labels == 0]
        n_per_class = min(len(intent_idx), len(observe_idx), max_trials_per_subject // 2)

        if n_per_class < 5:
            continue

        rng.shuffle(intent_idx)
        rng.shuffle(observe_idx)
        selected_indices.extend(intent_idx[:n_per_class])
        selected_indices.extend(observe_idx[:n_per_class])

    selected_indices = np.array(selected_indices)
    rng.shuffle(selected_indices)

    n_selected = len(selected_indices)
    logger.info(f"Selected {n_selected} balanced trials")

    # Extract and transpose EEG: (N, samples, channels) -> (N, channels, samples)
    logger.info("Extracting EEG data (this may take a moment for large files)...")
    eeg_out = np.zeros((n_selected, n_channels, n_samples), dtype=np.float32)
    for i, idx in enumerate(selected_indices):
        eeg_out[i] = eeg_raw[idx, :, :n_channels].T  # drop ref channel, transpose
        if (i + 1) % 1000 == 0:
            logger.info(f"  Processed {i + 1}/{n_selected} trials")

    # Placeholder gaze — zeros, clearly not real data
    gaze_out = np.zeros((n_selected, n_samples, 3), dtype=np.float32)

    labels_out = binary_labels[selected_indices]
    subjects_out = subject_ids_raw[selected_indices]

    # Save arrays
    np.save(output_dir / "eeg_data.npy", eeg_out)
    np.save(output_dir / "gaze_data.npy", gaze_out)
    np.save(output_dir / "labels.npy", labels_out)
    np.save(output_dir / "subject_ids.npy", subjects_out)

    # Save metadata
    metadata = {
        "source_file": str(npz_path),
        "npz_keys": data.files,
        "gaze_source": "none_available",
        "gaze_note": (
            "No eye-tracking time series in EEGEyeNet benchmark NPZ files. "
            "Raw .mat files with gaze data were on OSF Dropbox (no longer available). "
            "Gaze features validated on synthetic data only."
        ),
        "label_note": (
            "Labels are an artificial median split on target eccentricity, "
            "NOT cognitive intent vs. observe. Use for EEG feature validation only."
        ),
        "n_trials": int(n_selected),
        "n_channels": int(n_channels),
        "n_samples": int(n_samples),
        "n_subjects": int(len(np.unique(subjects_out))),
        "n_class_0": int(np.sum(labels_out == 0)),
        "n_class_1": int(np.sum(labels_out == 1)),
        "screen_center": [float(cx), float(cy)],
        "median_eccentricity_px": float(median_ecc),
        "max_subjects_setting": max_subjects,
        "max_trials_per_subject_setting": max_trials_per_subject,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"\nSaved to {output_dir}:")
    logger.info(f"  eeg_data.npy:    {eeg_out.shape} ({eeg_out.nbytes / 1e6:.1f} MB)")
    logger.info(f"  gaze_data.npy:   zeros placeholder (no real gaze available)")
    logger.info(f"  labels.npy:      {labels_out.shape} (artificial spatial split)")
    logger.info(f"  subject_ids.npy: {subjects_out.shape} ({len(np.unique(subjects_out))} subjects)")
    logger.info(f"\nFor EEG-only evaluation:")
    logger.info(f"  python scripts/train.py --data-dir {output_dir.parent.parent} --paradigm prosaccade")
    logger.info(f"\nFor full fusion ablation, use synthetic data:")
    logger.info(f"  python scripts/train.py --synthetic")


def main():
    parser = argparse.ArgumentParser(description="Preprocess EEGEyeNet Position task data")
    parser.add_argument("--npz", default="data/raw/eegeyenet/Position_task_with_dots_synchronised_min.npz",
                        help="Path to the NPZ file")
    parser.add_argument("--output-dir", default="data/raw/eegeyenet/prosaccade/preprocessed",
                        help="Output directory for preprocessed files")
    parser.add_argument("--max-subjects", type=int, default=50,
                        help="Max subjects to use (default: 50 for memory)")
    parser.add_argument("--max-trials-per-subject", type=int, default=80,
                        help="Max trials per subject per class")
    parser.add_argument("--info", action="store_true",
                        help="Inspect the NPZ file (print keys, shapes, dtypes) without processing")
    args = parser.parse_args()

    if args.info:
        inspect_npz(args.npz)
    else:
        preprocess(args.npz, args.output_dir, args.max_subjects, args.max_trials_per_subject)


if __name__ == "__main__":
    main()
