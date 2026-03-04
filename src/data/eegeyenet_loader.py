"""
EEGEyeNet Dataset Loader

Loads and preprocesses the EEGEyeNet dataset (Kastrati et al., 2021), which provides
synchronized 128-channel EEG and eye-tracking data from 356 subjects across three
paradigms: pro-saccade, anti-saccade, and visual symbol search.

For this pipeline, we primarily use the pro-saccade paradigm where subjects make
directed gaze shifts to targets — providing clear "intent to select" vs "passive
observation" conditions needed for the Midas touch classification problem.

Reference:
    Kastrati et al. "EEGEyeNet: a Simultaneous Electroencephalography and
    Eye-tracking Dataset and Benchmark for Eye Movement Prediction" (2021)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# EEGEyeNet standard 128-channel BioSemi layout (subset of relevant channels)
OCCIPITOPARIETAL_CHANNELS = ["Oz", "O1", "O2", "Pz", "P3", "P4", "POz", "PO3", "PO4"]
CENTRAL_CHANNELS = ["Cz", "C3", "C4"]
FRONTAL_CHANNELS = ["Fz", "F3", "F4", "Fp1", "Fp2"]

# EEGEyeNet paradigm condition mappings
PROSACCADE_CONDITIONS = {
    "target_directed": 1,   # Subject saccades toward target (intent)
    "fixation_baseline": 0,  # Subject maintains fixation (no intent)
}

ANTISACCADE_CONDITIONS = {
    "target_directed": 1,    # Correct anti-saccade (controlled gaze)
    "fixation_baseline": 0,
    "error_prosaccade": -1,  # Error trial — looked at target involuntarily
}


@dataclass
class EEGEyeNetTrial:
    """A single trial from the EEGEyeNet dataset."""
    subject_id: int
    trial_idx: int
    paradigm: str
    condition: int  # 1 = target_directed (intent), 0 = baseline (no intent)

    # EEG data: (n_channels, n_samples)
    eeg_data: np.ndarray = field(default=None, repr=False)
    # Eye-tracking data: (n_samples, n_features) — [x, y, pupil]
    gaze_data: np.ndarray = field(default=None, repr=False)

    eeg_sfreq: float = 500.0
    gaze_sfreq: float = 500.0
    channel_names: list[str] = field(default_factory=list)

    # Timing
    stimulus_onset_sample: int = 0
    response_onset_sample: Optional[int] = None

    @property
    def duration_ms(self) -> float:
        return (self.eeg_data.shape[1] / self.eeg_sfreq) * 1000

    @property
    def label(self) -> int:
        """Binary label: 1 = intent to select, 0 = passive observation."""
        return 1 if self.condition == 1 else 0


@dataclass
class EEGEyeNetDataset:
    """Container for loaded EEGEyeNet data."""
    trials: list[EEGEyeNetTrial] = field(default_factory=list)
    paradigm: str = "prosaccade"
    subjects: list[int] = field(default_factory=list)

    @property
    def n_trials(self) -> int:
        return len(self.trials)

    @property
    def n_subjects(self) -> int:
        return len(set(t.subject_id for t in self.trials))

    def get_subject_trials(self, subject_id: int) -> list[EEGEyeNetTrial]:
        return [t for t in self.trials if t.subject_id == subject_id]

    def get_labels(self) -> np.ndarray:
        return np.array([t.label for t in self.trials])

    def get_subject_ids(self) -> np.ndarray:
        return np.array([t.subject_id for t in self.trials])

    def split_by_condition(self) -> dict[str, list[EEGEyeNetTrial]]:
        intent = [t for t in self.trials if t.label == 1]
        observe = [t for t in self.trials if t.label == 0]
        return {"intent": intent, "observe": observe}


def load_eegeyenet_matlab(
    data_dir: str | Path,
    paradigm: str = "prosaccade",
    subjects: Optional[list[int]] = None,
    max_subjects: Optional[int] = None,
) -> EEGEyeNetDataset:
    """
    Load EEGEyeNet data from .mat/.npy preprocessed files.

    The EEGEyeNet dataset is distributed as MATLAB .mat files or preprocessed
    NumPy arrays. This loader handles both formats.

    Args:
        data_dir: Path to the EEGEyeNet data directory
        paradigm: Which paradigm to load (prosaccade, antisaccade, visual_symbol_search)
        subjects: Specific subject IDs to load (None = all)
        max_subjects: Maximum number of subjects to load (for development)

    Returns:
        EEGEyeNetDataset with loaded trials
    """
    data_dir = Path(data_dir)
    dataset = EEGEyeNetDataset(paradigm=paradigm)

    # Check for preprocessed numpy format first
    npy_dir = data_dir / paradigm / "preprocessed"
    mat_dir = data_dir / paradigm

    if npy_dir.exists():
        dataset = _load_from_numpy(npy_dir, paradigm, subjects, max_subjects)
    elif mat_dir.exists():
        dataset = _load_from_matlab(mat_dir, paradigm, subjects, max_subjects)
    else:
        logger.warning(
            f"Data directory not found: {data_dir}. "
            "Use scripts/download_data.py to fetch the EEGEyeNet dataset, "
            "or generate synthetic data with generate_synthetic_dataset()."
        )

    logger.info(
        f"Loaded {dataset.n_trials} trials from {dataset.n_subjects} subjects "
        f"({paradigm} paradigm)"
    )
    return dataset


def _load_from_numpy(
    npy_dir: Path,
    paradigm: str,
    subjects: Optional[list[int]],
    max_subjects: Optional[int],
) -> EEGEyeNetDataset:
    """Load from preprocessed NumPy .npy files."""
    dataset = EEGEyeNetDataset(paradigm=paradigm)

    eeg_file = npy_dir / "eeg_data.npy"
    gaze_file = npy_dir / "gaze_data.npy"
    labels_file = npy_dir / "labels.npy"
    subjects_file = npy_dir / "subject_ids.npy"
    channels_file = npy_dir / "channel_names.npy"

    if not eeg_file.exists():
        logger.warning(f"Preprocessed data not found at {npy_dir}")
        return dataset

    eeg_all = np.load(eeg_file)        # (n_trials, n_channels, n_samples)
    gaze_all = np.load(gaze_file)      # (n_trials, n_samples, 3) — x, y, pupil
    labels = np.load(labels_file)      # (n_trials,)
    subj_ids = np.load(subjects_file)  # (n_trials,)

    channel_names = list(np.load(channels_file)) if channels_file.exists() else []

    # Filter subjects
    unique_subjects = sorted(set(subj_ids))
    if subjects is not None:
        unique_subjects = [s for s in unique_subjects if s in subjects]
    if max_subjects is not None:
        unique_subjects = unique_subjects[:max_subjects]

    subject_set = set(unique_subjects)
    for i in range(len(labels)):
        if subj_ids[i] not in subject_set:
            continue
        trial = EEGEyeNetTrial(
            subject_id=int(subj_ids[i]),
            trial_idx=i,
            paradigm=paradigm,
            condition=int(labels[i]),
            eeg_data=eeg_all[i],
            gaze_data=gaze_all[i],
            channel_names=channel_names,
        )
        dataset.trials.append(trial)

    dataset.subjects = unique_subjects
    return dataset


def _load_from_matlab(
    mat_dir: Path,
    paradigm: str,
    subjects: Optional[list[int]],
    max_subjects: Optional[int],
) -> EEGEyeNetDataset:
    """Load from original MATLAB .mat files using h5py or scipy."""
    dataset = EEGEyeNetDataset(paradigm=paradigm)

    try:
        import h5py
    except ImportError:
        try:
            from scipy.io import loadmat
        except ImportError:
            logger.error("Need h5py or scipy to load .mat files")
            return dataset

    mat_files = sorted(mat_dir.glob("*.mat"))
    if not mat_files:
        logger.warning(f"No .mat files found in {mat_dir}")
        return dataset

    loaded_subjects = 0
    for mat_path in mat_files:
        if max_subjects is not None and loaded_subjects >= max_subjects:
            break

        try:
            # Try h5py first (for MATLAB v7.3+ files)
            with h5py.File(mat_path, "r") as f:
                eeg = np.array(f["EEG"]["data"])
                gaze = np.array(f["ET"]["data"]) if "ET" in f else None
                sub_id = int(np.array(f["subject_id"]).flat[0])
                conditions = np.array(f["conditions"]).flatten()
        except Exception:
            try:
                mat = loadmat(str(mat_path))
                eeg = mat["EEG_data"]
                gaze = mat.get("ET_data")
                sub_id = int(mat["subject_id"].flat[0])
                conditions = mat["conditions"].flatten()
            except Exception as e:
                logger.warning(f"Could not load {mat_path}: {e}")
                continue

        if subjects is not None and sub_id not in subjects:
            continue

        # Create trials from this subject
        n_trials_in_file = conditions.shape[0]
        for t_idx in range(n_trials_in_file):
            trial = EEGEyeNetTrial(
                subject_id=sub_id,
                trial_idx=t_idx,
                paradigm=paradigm,
                condition=int(conditions[t_idx]),
                eeg_data=eeg[t_idx] if eeg.ndim == 3 else eeg,
                gaze_data=gaze[t_idx] if gaze is not None and gaze.ndim == 3 else (
                    gaze if gaze is not None else np.zeros((eeg.shape[-1], 3))
                ),
            )
            dataset.trials.append(trial)

        dataset.subjects.append(sub_id)
        loaded_subjects += 1

    return dataset


def generate_synthetic_dataset(
    n_subjects: int = 20,
    trials_per_subject: int = 80,
    n_channels: int = 128,
    n_samples: int = 500,
    sfreq: float = 500.0,
    noise_level: float = 0.5,
    random_state: int = 42,
) -> EEGEyeNetDataset:
    """
    Generate a synthetic EEGEyeNet-like dataset for development and testing.

    Simulates REALISTIC difficulty levels matching published BCI literature:
    - EEG-only classification: ~60-70% (weak single-trial ERD, high noise)
    - Gaze-only classification: ~65-75% (Midas touch: looking != intending)
    - Fused (EEG+Gaze): ~75-85% (fusion provides meaningful improvement)

    Key realism features:
    - Condition effects in only ~10 occipitoparietal channels (not all 128)
    - Spatially correlated noise (volume conduction)
    - Both conditions have saccades to targets (Midas touch overlap)
    - ~20% of subjects show weak/no condition effect
    - Heavy trial-to-trial variability

    Args:
        n_subjects: Number of simulated subjects
        trials_per_subject: Trials per subject (half intent, half observe)
        n_channels: Number of EEG channels
        n_samples: Samples per trial
        sfreq: Sampling frequency in Hz
        noise_level: Amount of noise to add (0-1)
        random_state: Random seed for reproducibility
    """
    rng = np.random.RandomState(random_state)
    dataset = EEGEyeNetDataset(paradigm="prosaccade")
    t = np.arange(n_samples) / sfreq

    # Standard 10-20 channel names (subset for 128-ch layout)
    channel_names = _get_128_channel_names()[:n_channels]

    # Identify occipitoparietal channels (where alpha ERD / SPN actually occur)
    # Only these ~10 channels carry condition-relevant signal
    occ_channels = set()
    for i, name in enumerate(channel_names):
        if any(name.startswith(p) for p in ("O", "PO", "Pz", "P3", "P4", "P7", "P8", "Iz")):
            occ_channels.add(i)

    # Spatial correlation matrix (volume conduction: nearby channels are correlated)
    # This makes the noise correlated, preventing classifiers from averaging it away
    spatial_cov = np.eye(n_channels)
    for i in range(n_channels):
        for j in range(i + 1, min(i + 6, n_channels)):
            corr = 0.6 ** (j - i)  # exponential decay
            spatial_cov[i, j] = corr
            spatial_cov[j, i] = corr
    # Cholesky for generating correlated noise
    L_spatial = np.linalg.cholesky(spatial_cov)

    for subj in range(n_subjects):
        # Per-subject variability
        alpha_peak = rng.uniform(9.0, 11.0)
        beta_peak = rng.uniform(18.0, 24.0)

        # ~20% of subjects are "non-responders" — show no/reversed condition effect
        subject_responsiveness = rng.uniform(-0.2, 1.0)
        subject_responsiveness = max(subject_responsiveness, 0.0)  # clamp negatives to 0

        # Subject-level gaze habits (some people make bigger saccades in general)
        subject_saccade_gain = rng.uniform(0.7, 1.3)

        for trial_idx in range(trials_per_subject):
            is_intent = trial_idx < (trials_per_subject // 2)
            condition = 1 if is_intent else 0

            # Per-trial variability
            trial_noise_scale = rng.uniform(0.8, 1.3)

            # --- Generate EEG ---
            # Start with spatially correlated 1/f noise (dominates the signal)
            white = rng.randn(n_channels, n_samples)
            correlated = L_spatial @ white  # apply spatial correlation

            eeg = np.zeros((n_channels, n_samples))
            for ch in range(n_channels):
                # 1/f noise (pink noise — dominant in real EEG)
                freqs = np.fft.rfftfreq(n_samples, d=1.0 / sfreq)
                freqs[0] = 1.0
                spectrum = rng.randn(len(freqs)) + 1j * rng.randn(len(freqs))
                spectrum *= 1.0 / np.sqrt(freqs)
                eeg[ch] = np.fft.irfft(spectrum, n=n_samples)

                # Add correlated noise component
                eeg[ch] += correlated[ch] * 2.0

                # Alpha rhythm — SAME baseline in all channels
                # (alpha is everywhere, not condition-specific)
                alpha_amp = rng.uniform(2.0, 4.0)
                phase = rng.uniform(0, 2 * np.pi)
                eeg[ch] += alpha_amp * np.sin(2 * np.pi * alpha_peak * t + phase)

                # Condition-specific alpha ERD: ONLY in occipitoparietal channels
                # and ONLY a tiny modulation (~5-10% of alpha amplitude)
                if ch in occ_channels and subject_responsiveness > 0:
                    erd_strength = subject_responsiveness * rng.uniform(0.04, 0.14)
                    if is_intent:
                        # Alpha suppression (ERD) — intent
                        eeg[ch] *= (1.0 - erd_strength * trial_noise_scale)
                    else:
                        # Alpha enhancement (ERS) — observe
                        eeg[ch] *= (1.0 + erd_strength * 0.5 * trial_noise_scale)

                # SPN-like slow potential — small, only in some intent trials,
                # only in occipitoparietal channels
                if (is_intent and ch in occ_channels and
                        subject_responsiveness > 0.2 and rng.random() > 0.4):
                    spn_onset = int(rng.uniform(0.3, 0.5) * n_samples)
                    spn_amp = rng.uniform(0.3, 1.0) * subject_responsiveness
                    spn = np.zeros(n_samples)
                    spn[spn_onset:] = -spn_amp * (
                        1 - np.exp(-3 * t[:n_samples - spn_onset])
                    )
                    eeg[ch] += spn

            # Scale to realistic microvolts + heavy noise
            eeg *= 15.0
            eeg += trial_noise_scale * noise_level * rng.randn(n_channels, n_samples) * 30.0

            # --- Generate Eye-Tracking (gaze_x, gaze_y, pupil_diameter) ---
            # KEY INSIGHT: Both conditions have saccades to targets!
            # The Midas touch problem means people LOOK at things they don't
            # intend to select. The difference is subtle: timing, dwell duration,
            # approach velocity, and pupil response.
            gaze = np.zeros((n_samples, 3))

            # Both conditions: saccade to a target location
            target_x = rng.uniform(-10, 10)
            target_y = rng.uniform(-10, 10)
            saccade_onset = int(rng.uniform(0.15, 0.45) * n_samples)
            saccade_dur = int(rng.uniform(0.04, 0.08) * n_samples)
            end = min(saccade_onset + saccade_dur, n_samples - 1)

            # Pre-saccade fixation with drift
            drift = rng.uniform(-1.5, 1.5, size=2)
            gaze[:saccade_onset, 0] = drift[0] + rng.randn(saccade_onset) * 0.5
            gaze[:saccade_onset, 1] = drift[1] + rng.randn(saccade_onset) * 0.5

            # Saccade (both conditions make a saccade)
            gain = rng.uniform(0.8, 1.1) * subject_saccade_gain
            s = np.linspace(0, 1, end - saccade_onset)
            sigmoid = 1.0 / (1.0 + np.exp(-10 * (s - 0.5)))
            gaze[saccade_onset:end, 0] = target_x * gain * sigmoid
            gaze[saccade_onset:end, 1] = target_y * gain * sigmoid

            if is_intent:
                # INTENT: stays on target with relatively tight fixation,
                # slightly larger pupil dilation from cognitive load
                fix_noise = rng.uniform(0.3, 0.6)
                gaze[end:, 0] = target_x * gain + rng.randn(n_samples - end) * fix_noise
                gaze[end:, 1] = target_y * gain + rng.randn(n_samples - end) * fix_noise

                # Pupil: moderate dilation (cognitive load of selection)
                base_pupil = rng.uniform(3.4, 4.4)
                gaze[:, 2] = base_pupil + rng.randn(n_samples) * 0.25
                gaze[saccade_onset:, 2] += rng.uniform(0.05, 0.3)
            else:
                # OBSERVE: VERY similar to intent — this IS the Midas touch problem.
                # ~80% of observe trials: stay on target (indistinguishable by gaze)
                # ~20% of observe trials: slight gaze drift after fixation
                fix_noise = rng.uniform(0.35, 0.7)  # slightly more jitter

                if rng.random() < 0.2:
                    # Slight late drift (still hard to distinguish)
                    wander_onset = end + int(rng.uniform(0.2, 0.4) * n_samples)
                    wander_onset = min(wander_onset, n_samples - 1)
                    gaze[end:wander_onset, 0] = target_x * gain + rng.randn(max(wander_onset - end, 0)) * fix_noise
                    gaze[end:wander_onset, 1] = target_y * gain + rng.randn(max(wander_onset - end, 0)) * fix_noise
                    # Small drift (not large wander — stays near target)
                    drift_x = rng.uniform(-1.5, 1.5)
                    drift_y = rng.uniform(-1.5, 1.5)
                    remaining = n_samples - wander_onset
                    if remaining > 0:
                        gaze[wander_onset:, 0] = target_x * gain + drift_x + rng.randn(remaining) * 0.6
                        gaze[wander_onset:, 1] = target_y * gain + drift_y + rng.randn(remaining) * 0.6
                else:
                    # Stays on target — nearly identical to intent
                    gaze[end:, 0] = target_x * gain + rng.randn(n_samples - end) * fix_noise
                    gaze[end:, 1] = target_y * gain + rng.randn(n_samples - end) * fix_noise

                # Pupil: nearly identical range (heavily overlapping)
                base_pupil = rng.uniform(3.3, 4.3)
                gaze[:, 2] = base_pupil + rng.randn(n_samples) * 0.25
                gaze[saccade_onset:, 2] += rng.uniform(0.0, 0.2)

            trial = EEGEyeNetTrial(
                subject_id=subj,
                trial_idx=trial_idx,
                paradigm="prosaccade",
                condition=condition,
                eeg_data=eeg,
                eeg_sfreq=sfreq,
                gaze_data=gaze,
                gaze_sfreq=sfreq,
                channel_names=channel_names,
                stimulus_onset_sample=int(0.5 * n_samples),
            )
            dataset.trials.append(trial)

        dataset.subjects.append(subj)

    logger.info(
        f"Generated synthetic dataset: {dataset.n_trials} trials, "
        f"{dataset.n_subjects} subjects"
    )
    return dataset


def _get_128_channel_names() -> list[str]:
    """Return standard 128-channel BioSemi cap channel names."""
    standard = [
        "Fp1", "Fp2", "AF3", "AF4", "AF7", "AF8", "AFz",
        "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "Fz",
        "FC1", "FC2", "FC3", "FC4", "FC5", "FC6", "FCz",
        "FT7", "FT8", "FT9", "FT10",
        "C1", "C2", "C3", "C4", "C5", "C6", "Cz",
        "CP1", "CP2", "CP3", "CP4", "CP5", "CP6", "CPz",
        "TP7", "TP8", "TP9", "TP10",
        "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "Pz",
        "PO3", "PO4", "PO7", "PO8", "POz",
        "O1", "O2", "Oz",
        "I1", "I2", "Iz",
    ]
    # Extend to 128 channels with BioSemi A/B naming for extras
    while len(standard) < 128:
        standard.append(f"EX{len(standard) - 65}")
    return standard[:128]
