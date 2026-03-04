"""
Combined Feature Pipeline

Orchestrates EEG and gaze feature extraction, producing aligned feature
matrices suitable for fusion classification. Supports three extraction
modes for ablation analysis:

1. **EEG-only:** Band power + SPN + ERD/ERS features
2. **Gaze-only:** Fixation + saccade + pupil + dispersion features
3. **Fused:** Concatenated EEG + gaze features (the Axion Click thesis)

This ablation is central to validating whether multimodal fusion provides
a meaningful improvement over unimodal approaches — the fundamental
question that Axion Click is built on.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

import numpy as np

from src.data.eegeyenet_loader import EEGEyeNetDataset, EEGEyeNetTrial
from src.features.artifact_rejection import reject_dataset, RejectionStats
from src.features.eeg_features import extract_eeg_features
from src.features.gaze_features import extract_gaze_features

logger = logging.getLogger(__name__)


class FeatureMode(str, Enum):
    EEG_ONLY = "eeg_only"
    GAZE_ONLY = "gaze_only"
    FUSED = "fused"


# ---------------------------------------------------------------------------
# Neuroscience-informed channel groups
# ---------------------------------------------------------------------------
# These map standard 10-20/10-5 electrode names to functional regions relevant
# to intent decoding.  Using regional mean+std instead of per-channel values
# reduces dimensionality by ~15× while focusing on physiologically meaningful
# signals.

CHANNEL_GROUP_NAMES: dict[str, list[str]] = {
    # Occipitoparietal — posterior alpha, SPN (Reddy et al. CHI 2024)
    "occipitoparietal": [
        "Oz", "O1", "O2", "Pz", "P3", "P4", "POz", "PO3", "PO4",
    ],
    # Central — sensorimotor mu/beta, ERD/ERS during motor preparation
    "central": [
        "Cz", "C3", "C4",
    ],
    # Frontal-midline — theta engagement, executive attention
    "frontal_midline": [
        "Fz", "FCz",
    ],
    # Parietal — P300, spatial attention, orienting
    "parietal": [
        "Pz", "P3", "P4", "P7", "P8",
    ],
}


def resolve_channel_groups(
    channel_names: list[str],
    group_definitions: Optional[dict[str, list[str]]] = None,
) -> dict[str, list[int]]:
    """Resolve channel name groups to index groups.

    Args:
        channel_names: Ordered list of channel names (length = n_channels).
        group_definitions: Mapping of group_name → list of channel names.
            Defaults to :data:`CHANNEL_GROUP_NAMES`.

    Returns:
        Dict mapping group_name → list of integer indices into *channel_names*.
        Groups where no channel matched are omitted.
    """
    if group_definitions is None:
        group_definitions = CHANNEL_GROUP_NAMES

    # Build name → index lookup (case-insensitive)
    name_to_idx: dict[str, int] = {
        name.lower(): idx for idx, name in enumerate(channel_names)
    }

    resolved: dict[str, list[int]] = {}
    for group, names in group_definitions.items():
        indices = []
        for ch_name in names:
            idx = name_to_idx.get(ch_name.lower())
            if idx is not None and idx not in indices:
                indices.append(idx)
        if indices:
            resolved[group] = sorted(indices)

    if resolved:
        total_ch = sum(len(v) for v in resolved.values())
        logger.info(
            f"Channel groups resolved: {len(resolved)} groups, "
            f"{total_ch} unique channels (from {len(channel_names)} total)"
        )
    else:
        logger.warning(
            "No channel groups could be resolved — check channel names. "
            "Falling back to per-channel features."
        )

    return resolved


def extract_trial_features(
    trial: EEGEyeNetTrial,
    mode: FeatureMode = FeatureMode.FUSED,
    eeg_method: str = "welch",
    channel_groups: Optional[dict[str, list[int]]] = None,
    velocity_threshold: float = 30.0,
) -> np.ndarray:
    """
    Extract features from a single trial.

    Args:
        trial: An EEGEyeNetTrial with synchronized EEG and gaze data
        mode: Feature extraction mode (eeg_only, gaze_only, fused)
        eeg_method: Band power estimation method
        channel_groups: EEG channel groupings for regional features
        velocity_threshold: Saccade detection threshold

    Returns:
        1D feature vector
    """
    features = []

    if mode in (FeatureMode.EEG_ONLY, FeatureMode.FUSED):
        eeg_feats = extract_eeg_features(
            trial.eeg_data,
            trial.eeg_sfreq,
            method=eeg_method,
            channel_groups=channel_groups,
            include_spn=True,
            include_erds=True,
            stimulus_sample=trial.stimulus_onset_sample or None,
        )
        features.append(eeg_feats)

    if mode in (FeatureMode.GAZE_ONLY, FeatureMode.FUSED):
        gaze_feats = extract_gaze_features(
            trial.gaze_data,
            sfreq=trial.gaze_sfreq,
            velocity_threshold=velocity_threshold,
        )
        features.append(gaze_feats)

    return np.concatenate(features)


def extract_dataset_features(
    dataset: EEGEyeNetDataset,
    mode: FeatureMode = FeatureMode.FUSED,
    eeg_method: str = "welch",
    channel_groups: Optional[dict[str, list[int]]] = None,
    use_channel_groups: bool = True,
    verbose: bool = True,
    reject_artifacts: bool = False,
    amplitude_threshold: float = 100.0,
    gradient_threshold: float = 50.0,
    flat_threshold: float = 0.5,
    gaze_loss_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract features from all trials in a dataset.

    Args:
        dataset: Loaded EEGEyeNetDataset
        mode: Feature extraction mode
        eeg_method: Band power estimation method
        channel_groups: EEG channel groupings (indices). If None and
            *use_channel_groups* is True, groups are auto-resolved from
            the first trial's channel_names using :func:`resolve_channel_groups`.
        use_channel_groups: Whether to auto-resolve channel groups when
            *channel_groups* is not provided.  Reduces EEG features from
            ~640 per-channel values to ~40 group-level aggregates.
        verbose: Whether to log progress
        reject_artifacts: Whether to apply artifact rejection before
            feature extraction. When True, trials that fail amplitude,
            gradient, flat-channel, or gaze-loss checks are excluded.
        amplitude_threshold: Max absolute EEG amplitude in µV (default 100).
        gradient_threshold: Max sample-to-sample EEG jump in µV (default 50).
        flat_threshold: Min channel std in µV to be considered active (default 0.5).
        gaze_loss_threshold: Max fraction of lost gaze samples (default 0.5).

    Returns:
        Tuple of:
        - X: Feature matrix, shape (n_trials, n_features)
        - y: Label vector, shape (n_trials,)
        - subject_ids: Subject ID vector, shape (n_trials,)
    """
    # --- Artifact rejection (optional) ---
    trials = dataset.trials
    if reject_artifacts:
        trials, rej_stats = reject_dataset(
            trials,
            amplitude_threshold=amplitude_threshold,
            gradient_threshold=gradient_threshold,
            flat_threshold=flat_threshold,
            gaze_loss_threshold=gaze_loss_threshold,
            verbose=verbose,
        )

    # Auto-resolve channel groups from dataset channel names
    if channel_groups is None and use_channel_groups and mode != FeatureMode.GAZE_ONLY:
        ch_names = _get_channel_names(dataset)
        if ch_names:
            channel_groups = resolve_channel_groups(ch_names)
            if not channel_groups:
                channel_groups = None  # Fall back to per-channel
        else:
            logger.info("No channel names available — using per-channel features")

    X_list = []
    y_list = []
    subj_list = []
    n_errors = 0
    n_trials = len(trials)

    for i, trial in enumerate(trials):
        try:
            feat_vec = extract_trial_features(trial, mode=mode, eeg_method=eeg_method,
                                               channel_groups=channel_groups)
            X_list.append(feat_vec)
            y_list.append(trial.label)
            subj_list.append(trial.subject_id)
        except Exception as e:
            n_errors += 1
            if n_errors <= 5:
                logger.warning(f"Trial {i} feature extraction failed: {e}")

        if verbose and (i + 1) % 100 == 0:
            logger.info(f"Extracted features: {i + 1}/{n_trials}")

    if not X_list:
        raise ValueError("No features could be extracted from any trial")

    X = np.vstack(X_list)
    y = np.array(y_list)
    subject_ids = np.array(subj_list)

    # Handle NaN/Inf
    nan_mask = np.isnan(X) | np.isinf(X)
    if np.any(nan_mask):
        n_nan = np.sum(nan_mask)
        logger.warning(f"Replacing {n_nan} NaN/Inf values with 0")
        X[nan_mask] = 0.0

    logger.info(
        f"Feature extraction complete: X={X.shape}, "
        f"y={y.shape} (intent={np.sum(y == 1)}, observe={np.sum(y == 0)}), "
        f"mode={mode.value}"
    )
    return X, y, subject_ids


def _get_channel_names(dataset: EEGEyeNetDataset) -> list[str]:
    """Get channel names from the dataset, falling back to standard montage."""
    # Check first trial
    if dataset.trials and dataset.trials[0].channel_names:
        return dataset.trials[0].channel_names

    # Fall back to standard 128-channel BioSemi names
    from src.data.eegeyenet_loader import _get_128_channel_names
    n_ch = dataset.trials[0].eeg_data.shape[0] if dataset.trials else 128
    names = _get_128_channel_names()[:n_ch]
    logger.info(f"Using standard {n_ch}-channel BioSemi montage for channel group resolution")
    return names


def get_feature_names(
    mode: FeatureMode = FeatureMode.FUSED,
    n_eeg_channels: int = 128,
    channel_groups: Optional[dict[str, list[int]]] = None,
) -> list[str]:
    """Get descriptive names for each feature dimension.

    When *channel_groups* is provided, EEG feature names reflect group-level
    aggregation (mean+std per group per band) rather than per-channel values.
    """
    names = []

    if mode in (FeatureMode.EEG_ONLY, FeatureMode.FUSED):
        bands = ["theta", "alpha", "mu", "beta", "low_gamma"]
        if channel_groups:
            for group in channel_groups:
                for band in bands:
                    names.append(f"eeg_{group}_{band}_mean")
                    names.append(f"eeg_{group}_{band}_std")
        else:
            for band in bands:
                for ch in range(n_eeg_channels):
                    names.append(f"eeg_ch{ch}_{band}")

        names.extend(["spn_mean", "spn_min"])
        names.extend(["erds_mu", "erds_beta"])

    if mode in (FeatureMode.GAZE_ONLY, FeatureMode.FUSED):
        names.extend([
            "n_fixations", "mean_fix_duration", "max_fix_duration",
            "std_fix_duration", "mean_fix_dispersion",
            "n_saccades", "mean_sac_amplitude", "max_sac_amplitude",
            "mean_peak_velocity", "max_peak_velocity",
            "pupil_mean", "pupil_std", "pupil_range",
            "pupil_change", "pupil_velocity_mean",
            "dispersion_rms", "dispersion_bcea", "range_x", "range_y",
        ])

    return names
