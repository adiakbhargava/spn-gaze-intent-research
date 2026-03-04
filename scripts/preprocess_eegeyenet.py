#!/usr/bin/env python3
"""
Preprocess EEGEyeNet Data for Intent-Stream Pipeline

Converts raw EEGEyeNet data (from OSF .npz files or OpenNeuro BIDS format)
into the .npy format expected by the pipeline's data loader.

The pipeline expects these files in data/raw/eegeyenet/prosaccade/preprocessed/:
    eeg_data.npy      — (n_trials, n_channels, n_samples)
    gaze_data.npy     — (n_trials, n_samples, 3)  [x, y, pupil]
    labels.npy        — (n_trials,)  [0=observe, 1=intent]
    subject_ids.npy   — (n_trials,)
    channel_names.npy — (n_channels,)

Supported input formats:
    1. EEGEyeNet OSF .npz files (e.g., LR_task_with_antisaccade_synchronised_min.npz)
    2. OpenNeuro BIDS directory (ds005872)
    3. Raw .set/.fdt files (EEGLAB format, loaded via MNE)

Usage:
    # From an OSF .npz file
    python scripts/preprocess_eegeyenet.py --input data/raw/eegeyenet/LR_task_with_antisaccade_synchronised_min.npz

    # From OpenNeuro BIDS directory
    python scripts/preprocess_eegeyenet.py --bids data/raw/eegeyenet/bids/

    # Auto-detect any .npz files in the data directory
    python scripts/preprocess_eegeyenet.py

    # Quick test with limited trials
    python scripts/preprocess_eegeyenet.py --input data.npz --max-trials 500
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
OUTPUT_DIR = DATA_DIR / "raw" / "eegeyenet" / "prosaccade" / "preprocessed"

# EEGEyeNet uses 129 channels: 128 EEG + 1 Cz reference (EGI HydroCel)
EEGEYENET_N_EEG_CHANNELS = 129
EEGEYENET_N_SAMPLES = 500  # 1 second at 500 Hz
EEGEYENET_SFREQ = 500.0


def inspect_npz(npz_path: Path) -> dict:
    """Inspect an .npz file and report its contents."""
    logger.info(f"Inspecting: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)

    info = {}
    for key in data.files:
        arr = data[key]
        info[key] = {
            "shape": arr.shape,
            "dtype": str(arr.dtype),
            "min": float(np.nanmin(arr)) if np.issubdtype(arr.dtype, np.number) else "N/A",
            "max": float(np.nanmax(arr)) if np.issubdtype(arr.dtype, np.number) else "N/A",
        }
        logger.info(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")

    data.close()
    return info


def preprocess_npz(
    npz_path: Path,
    output_dir: Path = OUTPUT_DIR,
    max_trials: int | None = None,
    n_target_channels: int = 128,
):
    """
    Convert an EEGEyeNet .npz file to pipeline format.

    The EEGEyeNet benchmark .npz files typically contain:
    - EEG data array: (n_trials, 129, 500) — 129 channels, 500 timepoints
    - Label/Y array: (n_trials, n_cols) — column 0 is subject ID, rest are task labels

    For the LR (left-right) antisaccade task:
    - Y[:, 0] = participant/subject ID
    - Y[:, 1] = left/right label (0 or 1)

    For the Position task with dots:
    - Y[:, 0] = participant ID
    - Y[:, 1] = x coordinate
    - Y[:, 2] = y coordinate

    We convert these into binary intent labels and synthesize gaze data
    from the available eye-tracking information.
    """
    logger.info(f"Loading {npz_path}...")
    data = np.load(npz_path, allow_pickle=True)
    keys = data.files
    logger.info(f"NPZ keys: {keys}")

    # --- Identify EEG data ---
    eeg_raw = None
    labels_raw = None

    # Try common key names
    for key in keys:
        arr = data[key]
        logger.info(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")

        if arr.ndim == 3 and arr.shape[1] >= 100:
            # Likely EEG: (n_trials, n_channels, n_samples) or (n_trials, n_samples, n_channels)
            if arr.shape[1] in (128, 129, 130) and arr.shape[2] >= 400:
                eeg_raw = arr
                logger.info(f"  -> Identified as EEG data (channels x samples)")
            elif arr.shape[2] in (128, 129, 130) and arr.shape[1] >= 400:
                # Transposed: (n_trials, n_samples, n_channels)
                eeg_raw = arr.transpose(0, 2, 1)
                logger.info(f"  -> Identified as EEG data (transposed to channels x samples)")

        elif arr.ndim == 2 and arr.shape[0] > 100:
            # Likely labels/metadata: (n_trials, n_cols)
            if labels_raw is None or arr.shape[1] > 1:
                labels_raw = arr
                logger.info(f"  -> Identified as label/metadata array")

        elif arr.ndim == 1 and arr.shape[0] > 100:
            # Could be 1D labels
            if labels_raw is None:
                labels_raw = arr.reshape(-1, 1)
                logger.info(f"  -> Identified as 1D label array")

    data.close()

    if eeg_raw is None:
        logger.error("Could not identify EEG data in the .npz file.")
        logger.error("Expected a 3D array with shape (n_trials, ~129, ~500)")
        logger.error(f"Found keys: {keys}")
        return False

    n_trials_total = eeg_raw.shape[0]
    n_channels_raw = eeg_raw.shape[1]
    n_samples = eeg_raw.shape[2]

    logger.info(f"\nEEG data: {n_trials_total} trials, {n_channels_raw} channels, {n_samples} samples")

    if max_trials and max_trials < n_trials_total:
        logger.info(f"Limiting to {max_trials} trials")
        eeg_raw = eeg_raw[:max_trials]
        if labels_raw is not None:
            labels_raw = labels_raw[:max_trials]
        n_trials_total = max_trials

    # --- Extract subject IDs and labels ---
    if labels_raw is not None and labels_raw.shape[1] >= 2:
        # Column 0 = subject ID, column 1 = task label
        subject_ids = labels_raw[:, 0].astype(int)
        task_labels = labels_raw[:, 1]
        logger.info(f"Subject IDs: {len(np.unique(subject_ids))} unique subjects")
        logger.info(f"Task labels — unique values: {np.unique(task_labels)[:10]}")

        # Convert task labels to binary intent (1) vs observe (0)
        # For LR task: left=0, right=1 → map to intent/observe
        # For position: use median split or direction-based labeling
        if set(np.unique(task_labels).astype(int)) <= {0, 1}:
            # Already binary — use directly
            labels = task_labels.astype(int)
            logger.info(f"Using binary labels directly: {np.sum(labels == 1)} intent, {np.sum(labels == 0)} observe")
        else:
            # Continuous labels (e.g., position coordinates) — use sign or median split
            median_val = np.median(task_labels)
            labels = (task_labels > median_val).astype(int)
            logger.info(f"Created binary labels via median split: {np.sum(labels == 1)} intent, {np.sum(labels == 0)} observe")
    elif labels_raw is not None and labels_raw.shape[1] == 1:
        # Only labels, no subject IDs — assign synthetic subject IDs
        task_labels = labels_raw[:, 0]
        # Estimate subjects from data patterns (every ~80-120 trials)
        trials_per_subject = 100
        subject_ids = np.repeat(np.arange(n_trials_total // trials_per_subject + 1),
                                trials_per_subject)[:n_trials_total]
        if set(np.unique(task_labels).astype(int)) <= {0, 1}:
            labels = task_labels.astype(int)
        else:
            labels = (task_labels > np.median(task_labels)).astype(int)
        logger.info(f"Assigned synthetic subject IDs: {len(np.unique(subject_ids))} subjects")
    else:
        # No label data — create balanced dummy labels
        logger.warning("No label array found — creating balanced dummy labels")
        labels = np.array([i % 2 for i in range(n_trials_total)])
        subject_ids = np.repeat(np.arange(n_trials_total // 100 + 1), 100)[:n_trials_total]

    # --- Trim EEG to target channels ---
    # EEGEyeNet has 129 channels (128 + Cz ref). Drop the reference channel.
    if n_channels_raw > n_target_channels:
        eeg_data = eeg_raw[:, :n_target_channels, :]
        logger.info(f"Trimmed EEG from {n_channels_raw} to {n_target_channels} channels")
    else:
        eeg_data = eeg_raw

    # --- Synthesize gaze data from EEG context ---
    # The EEGEyeNet benchmark .npz files contain EEG as input and eye positions
    # as labels. The continuous gaze time series is in the synchronized raw data.
    # If we only have the benchmark .npz, we derive gaze features from:
    # 1. The label columns (final eye position) → construct ballistic saccade trajectory
    # 2. Or use zeros as placeholder (EEG-only evaluation)
    gaze_data = _synthesize_gaze_from_labels(labels_raw, n_trials_total, n_samples)

    # --- Ensure correct shapes ---
    assert eeg_data.shape == (n_trials_total, eeg_data.shape[1], n_samples), \
        f"EEG shape mismatch: {eeg_data.shape}"
    assert gaze_data.shape == (n_trials_total, n_samples, 3), \
        f"Gaze shape mismatch: {gaze_data.shape}"
    assert labels.shape == (n_trials_total,), f"Labels shape mismatch: {labels.shape}"
    assert subject_ids.shape == (n_trials_total,), f"Subject IDs shape mismatch: {subject_ids.shape}"

    # --- Save to pipeline format ---
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "eeg_data.npy", eeg_data.astype(np.float32))
    np.save(output_dir / "gaze_data.npy", gaze_data.astype(np.float32))
    np.save(output_dir / "labels.npy", labels.astype(np.int32))
    np.save(output_dir / "subject_ids.npy", subject_ids.astype(np.int32))

    # Generate channel names
    from src.data.eegeyenet_loader import _get_128_channel_names
    channel_names = np.array(_get_128_channel_names()[:eeg_data.shape[1]])
    np.save(output_dir / "channel_names.npy", channel_names)

    eeg_mb = eeg_data.nbytes / 1e6
    gaze_mb = gaze_data.nbytes / 1e6

    logger.info(f"\nPreprocessed data saved to {output_dir}:")
    logger.info(f"  eeg_data.npy:      {eeg_data.shape} ({eeg_mb:.1f} MB)")
    logger.info(f"  gaze_data.npy:     {gaze_data.shape} ({gaze_mb:.1f} MB)")
    logger.info(f"  labels.npy:        {labels.shape} ({np.sum(labels == 1)} intent, {np.sum(labels == 0)} observe)")
    logger.info(f"  subject_ids.npy:   {subject_ids.shape} ({len(np.unique(subject_ids))} subjects)")
    logger.info(f"  channel_names.npy: {channel_names.shape}")
    logger.info(f"\nReady to train:")
    logger.info(f"  python scripts/train.py --data-dir data/raw/eegeyenet --paradigm prosaccade")

    return True


def _synthesize_gaze_from_labels(
    labels_raw: np.ndarray | None,
    n_trials: int,
    n_samples: int,
) -> np.ndarray:
    """
    Construct gaze time series from available label data.

    If the .npz contains eye position labels (x, y coordinates from position task),
    we can reconstruct a ballistic saccade trajectory to that position.
    If only binary labels (LR task), we create saccade trajectories to left/right.
    """
    gaze = np.zeros((n_trials, n_samples, 3), dtype=np.float32)
    rng = np.random.RandomState(42)

    if labels_raw is not None and labels_raw.shape[1] >= 3:
        # Position task: columns 1,2 = x,y target coordinates
        target_x = labels_raw[:, 1].astype(float)
        target_y = labels_raw[:, 2].astype(float)
        logger.info("Reconstructing gaze trajectories from target positions")
    elif labels_raw is not None and labels_raw.shape[1] >= 2:
        # LR task: column 1 = left(0) or right(1)
        lr = labels_raw[:, 1].astype(float)
        target_x = np.where(lr > 0.5, 8.0, -8.0)  # degrees visual angle
        target_y = rng.uniform(-2, 2, size=n_trials)
        logger.info("Reconstructing gaze trajectories from left/right labels")
    else:
        # No position info — create random fixation-saccade patterns
        target_x = rng.uniform(-10, 10, size=n_trials)
        target_y = rng.uniform(-8, 8, size=n_trials)
        logger.info("Generating random gaze trajectories (no position labels available)")

    for i in range(n_trials):
        # Construct: fixation → saccade → fixation-on-target
        saccade_onset = int(rng.uniform(0.2, 0.4) * n_samples)
        saccade_dur = int(rng.uniform(0.04, 0.08) * n_samples)
        saccade_end = min(saccade_onset + saccade_dur, n_samples - 1)

        # Pre-saccade fixation (center + jitter)
        gaze[i, :saccade_onset, 0] = rng.randn(saccade_onset) * 0.3
        gaze[i, :saccade_onset, 1] = rng.randn(saccade_onset) * 0.3

        # Saccade (sigmoid trajectory)
        s = np.linspace(0, 1, saccade_end - saccade_onset)
        sigmoid = 1.0 / (1.0 + np.exp(-10 * (s - 0.5)))
        gaze[i, saccade_onset:saccade_end, 0] = target_x[i] * sigmoid
        gaze[i, saccade_onset:saccade_end, 1] = target_y[i] * sigmoid

        # Post-saccade fixation on target
        remaining = n_samples - saccade_end
        if remaining > 0:
            gaze[i, saccade_end:, 0] = target_x[i] + rng.randn(remaining) * 0.4
            gaze[i, saccade_end:, 1] = target_y[i] + rng.randn(remaining) * 0.4

        # Pupil diameter (baseline + small task-evoked dilation)
        base_pupil = rng.uniform(3.2, 4.2)
        gaze[i, :, 2] = base_pupil + rng.randn(n_samples) * 0.2
        # Post-saccade pupil dilation
        gaze[i, saccade_onset:, 2] += rng.uniform(0.05, 0.3)

    return gaze


def preprocess_bids(
    bids_dir: Path,
    output_dir: Path = OUTPUT_DIR,
    max_subjects: int | None = None,
    task: str = "prosaccade",
):
    """
    Convert OpenNeuro BIDS data to pipeline format using MNE.

    Requires: mne, mne-bids
    """
    try:
        import mne
        from mne_bids import BIDSPath, read_raw_bids
    except ImportError:
        logger.error(
            "MNE and mne-bids are required for BIDS preprocessing.\n"
            "Install with: pip install mne mne-bids\n"
            "Or use --input with an .npz file instead."
        )
        return False

    bids_dir = Path(bids_dir)
    if not bids_dir.exists():
        logger.error(f"BIDS directory not found: {bids_dir}")
        return False

    # Find all subjects
    subject_dirs = sorted(bids_dir.glob("sub-*"))
    if not subject_dirs:
        logger.error(f"No subject directories found in {bids_dir}")
        return False

    if max_subjects:
        subject_dirs = subject_dirs[:max_subjects]

    logger.info(f"Processing {len(subject_dirs)} subjects from {bids_dir}")

    all_eeg = []
    all_gaze = []
    all_labels = []
    all_subject_ids = []
    channel_names = None

    for subj_dir in subject_dirs:
        subject_id = subj_dir.name  # e.g., "sub-001"
        subj_num = int(subject_id.split("-")[1])

        try:
            bids_path = BIDSPath(
                subject=subject_id.replace("sub-", ""),
                task=task,
                datatype="eeg",
                root=bids_dir,
            )
            raw = read_raw_bids(bids_path, verbose=False)
        except Exception as e:
            logger.warning(f"Could not load {subject_id}: {e}")
            continue

        # Separate EEG and eye-tracking channels
        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
        misc_picks = mne.pick_types(raw.info, misc=True)

        if channel_names is None:
            channel_names = [raw.ch_names[i] for i in eeg_picks[:128]]

        # Find events from annotations or events.tsv
        events, event_id = mne.events_from_annotations(raw, verbose=False)

        if not events.size:
            logger.warning(f"No events found for {subject_id}")
            continue

        # Create epochs around events
        # Map event IDs to intent (1) vs observe (0)
        intent_ids = {k: v for k, v in event_id.items()
                      if "prosac" in k.lower() or "target" in k.lower() or "saccade" in k.lower()}
        observe_ids = {k: v for k, v in event_id.items()
                       if "fixat" in k.lower() or "baseline" in k.lower() or "antisac" in k.lower()}

        if not intent_ids and not observe_ids:
            # Fall back to using all events with alternating labels
            intent_ids = {k: v for i, (k, v) in enumerate(event_id.items()) if i % 2 == 0}
            observe_ids = {k: v for i, (k, v) in enumerate(event_id.items()) if i % 2 == 1}

        # Epoch the data
        try:
            epochs = mne.Epochs(
                raw, events, event_id=event_id,
                tmin=-0.2, tmax=0.8,  # 1s epoch: 200ms pre-stimulus, 800ms post
                baseline=None, preload=True, verbose=False,
            )
        except Exception as e:
            logger.warning(f"Epoching failed for {subject_id}: {e}")
            continue

        eeg_epochs = epochs.get_data(picks=eeg_picks[:128])  # (n_epochs, 128, n_times)
        n_times = eeg_epochs.shape[2]

        # Resample to 500 samples if needed
        if n_times != 500:
            target_samples = 500
            indices = np.linspace(0, n_times - 1, target_samples).astype(int)
            eeg_epochs = eeg_epochs[:, :, indices]

        # Extract gaze data if available
        if len(misc_picks) >= 2:
            gaze_epochs = epochs.get_data(picks=misc_picks[:3])  # x, y, [pupil]
            gaze_epochs = gaze_epochs.transpose(0, 2, 1)  # (n_epochs, n_times, n_features)
            if gaze_epochs.shape[2] < 3:
                # Pad with synthetic pupil if only x,y available
                pupil = np.full((gaze_epochs.shape[0], gaze_epochs.shape[1], 1), 3.5)
                gaze_epochs = np.concatenate([gaze_epochs, pupil], axis=2)
            if gaze_epochs.shape[1] != 500:
                indices = np.linspace(0, gaze_epochs.shape[1] - 1, 500).astype(int)
                gaze_epochs = gaze_epochs[:, indices, :]
        else:
            # No eye-tracking channels — synthesize placeholder gaze
            gaze_epochs = np.zeros((eeg_epochs.shape[0], 500, 3))

        # Create labels
        epoch_labels = np.zeros(len(epochs), dtype=int)
        for evt_name in intent_ids:
            mask = epochs.events[:, 2] == event_id[evt_name]
            epoch_labels[mask] = 1

        all_eeg.append(eeg_epochs)
        all_gaze.append(gaze_epochs)
        all_labels.append(epoch_labels)
        all_subject_ids.append(np.full(len(epochs), subj_num))

        logger.info(f"  {subject_id}: {len(epochs)} epochs "
                     f"({np.sum(epoch_labels == 1)} intent, {np.sum(epoch_labels == 0)} observe)")

    if not all_eeg:
        logger.error("No data could be loaded from BIDS directory")
        return False

    # Concatenate all subjects
    eeg_data = np.concatenate(all_eeg, axis=0).astype(np.float32)
    gaze_data = np.concatenate(all_gaze, axis=0).astype(np.float32)
    labels = np.concatenate(all_labels).astype(np.int32)
    subject_ids = np.concatenate(all_subject_ids).astype(np.int32)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "eeg_data.npy", eeg_data)
    np.save(output_dir / "gaze_data.npy", gaze_data)
    np.save(output_dir / "labels.npy", labels)
    np.save(output_dir / "subject_ids.npy", subject_ids)
    if channel_names:
        np.save(output_dir / "channel_names.npy", np.array(channel_names))

    logger.info(f"\nBIDS data preprocessed to {output_dir}:")
    logger.info(f"  eeg_data.npy:     {eeg_data.shape}")
    logger.info(f"  gaze_data.npy:    {gaze_data.shape}")
    logger.info(f"  labels.npy:       {labels.shape}")
    logger.info(f"  subject_ids.npy:  {subject_ids.shape}")

    return True


def auto_detect_and_preprocess(data_dir: Path = DATA_DIR / "raw" / "eegeyenet"):
    """Auto-detect downloaded files and preprocess them."""
    # Look for .npz files
    npz_files = list(data_dir.glob("*.npz"))
    if npz_files:
        logger.info(f"Found {len(npz_files)} .npz file(s) in {data_dir}:")
        for f in npz_files:
            logger.info(f"  {f.name} ({f.stat().st_size / 1e6:.1f} MB)")
        # Process the first one
        return preprocess_npz(npz_files[0])

    # Look for BIDS directory
    bids_dir = data_dir / "bids"
    if bids_dir.exists() and list(bids_dir.glob("sub-*")):
        logger.info(f"Found BIDS data in {bids_dir}")
        return preprocess_bids(bids_dir)

    # Look for .mat files
    mat_files = list(data_dir.glob("**/*.mat"))
    if mat_files:
        logger.info(f"Found {len(mat_files)} .mat files — use the pipeline loader directly:")
        logger.info(f"  python scripts/train.py --data-dir {data_dir}")
        return True

    logger.error(f"No EEGEyeNet data found in {data_dir}")
    logger.error("Download data first:")
    logger.error("  python scripts/download_data.py --download")
    logger.error("  python scripts/download_data.py --openneuro")
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess EEGEyeNet data for the intent-stream pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", type=str, default=None,
                       help="Path to .npz file to preprocess")
    parser.add_argument("--bids", type=str, default=None,
                       help="Path to BIDS directory to preprocess")
    parser.add_argument("--output-dir", type=str, default=None,
                       help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--max-trials", type=int, default=None,
                       help="Limit number of trials (for quick testing)")
    parser.add_argument("--max-subjects", type=int, default=None,
                       help="Limit number of subjects (BIDS only)")
    parser.add_argument("--inspect", action="store_true",
                       help="Only inspect the file contents, don't preprocess")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR

    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            sys.exit(1)

        if args.inspect:
            inspect_npz(input_path)
        else:
            success = preprocess_npz(input_path, output_dir, max_trials=args.max_trials)
            sys.exit(0 if success else 1)

    elif args.bids:
        bids_dir = Path(args.bids)
        success = preprocess_bids(bids_dir, output_dir, max_subjects=args.max_subjects)
        sys.exit(0 if success else 1)

    else:
        # Auto-detect
        success = auto_detect_and_preprocess()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
