"""
Eye-Tracking Feature Extraction

Extracts gaze-based features from synchronized eye-tracking data, including
fixation metrics, saccade kinematics, and pupillometry. These features capture
the oculomotor signatures that distinguish intentional target selection from
passive viewing — the core of the "Midas touch" problem.

Key features:
- **Fixation duration:** Longer fixations on targets during intentional selection
- **Saccade amplitude/velocity:** Directed saccades are more ballistic
- **Pupil dilation:** Cognitive effort and anticipation dilate the pupil
- **Gaze dispersion:** Tight clustering during intent, diffuse during observation
- **Area-of-Interest (AOI) metrics:** Time spent looking at target regions
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Fixation:
    """A detected fixation event."""
    onset_sample: int
    offset_sample: int
    duration_ms: float
    mean_x: float
    mean_y: float
    dispersion: float  # spatial spread in degrees


@dataclass
class Saccade:
    """A detected saccade event."""
    onset_sample: int
    offset_sample: int
    duration_ms: float
    amplitude: float      # degrees visual angle
    peak_velocity: float  # degrees/second
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    direction: float      # radians


def detect_fixations(
    gaze_x: np.ndarray,
    gaze_y: np.ndarray,
    sfreq: float,
    velocity_threshold: float = 30.0,
    min_duration_ms: float = 100.0,
) -> list[Fixation]:
    """
    Detect fixations using a velocity-based algorithm (I-VT).

    Samples below the velocity threshold are classified as fixation candidates.
    Consecutive fixation samples are grouped, and groups shorter than
    min_duration_ms are discarded.

    Args:
        gaze_x: Horizontal gaze position (degrees)
        gaze_y: Vertical gaze position (degrees)
        sfreq: Sampling frequency in Hz
        velocity_threshold: Degrees/second threshold for fixation detection
        min_duration_ms: Minimum fixation duration in ms

    Returns:
        List of detected Fixation events
    """
    n_samples = len(gaze_x)
    if n_samples < 3:
        return []

    # Compute point-to-point velocity
    dx = np.diff(gaze_x)
    dy = np.diff(gaze_y)
    velocity = np.sqrt(dx**2 + dy**2) * sfreq  # deg/s
    velocity = np.concatenate([[0], velocity])

    # Find fixation samples (below threshold)
    is_fixation = velocity < velocity_threshold

    # Group consecutive fixation samples
    fixations = []
    in_fix = False
    fix_start = 0

    for i in range(n_samples):
        if is_fixation[i] and not in_fix:
            fix_start = i
            in_fix = True
        elif not is_fixation[i] and in_fix:
            duration_ms = (i - fix_start) / sfreq * 1000
            if duration_ms >= min_duration_ms:
                fix_x = gaze_x[fix_start:i]
                fix_y = gaze_y[fix_start:i]
                dispersion = np.sqrt(np.std(fix_x)**2 + np.std(fix_y)**2)
                fixations.append(Fixation(
                    onset_sample=fix_start,
                    offset_sample=i,
                    duration_ms=duration_ms,
                    mean_x=float(np.mean(fix_x)),
                    mean_y=float(np.mean(fix_y)),
                    dispersion=float(dispersion),
                ))
            in_fix = False

    # Handle fixation at end of data
    if in_fix:
        duration_ms = (n_samples - fix_start) / sfreq * 1000
        if duration_ms >= min_duration_ms:
            fix_x = gaze_x[fix_start:]
            fix_y = gaze_y[fix_start:]
            dispersion = np.sqrt(np.std(fix_x)**2 + np.std(fix_y)**2)
            fixations.append(Fixation(
                onset_sample=fix_start,
                offset_sample=n_samples,
                duration_ms=duration_ms,
                mean_x=float(np.mean(fix_x)),
                mean_y=float(np.mean(fix_y)),
                dispersion=float(dispersion),
            ))

    return fixations


def detect_saccades(
    gaze_x: np.ndarray,
    gaze_y: np.ndarray,
    sfreq: float,
    velocity_threshold: float = 30.0,
    min_amplitude: float = 1.0,
) -> list[Saccade]:
    """
    Detect saccades using a velocity-based algorithm.

    Samples above the velocity threshold are classified as saccade candidates.
    Connected saccade samples with sufficient amplitude form saccade events.

    Args:
        gaze_x: Horizontal gaze position (degrees)
        gaze_y: Vertical gaze position (degrees)
        sfreq: Sampling frequency in Hz
        velocity_threshold: Degrees/second threshold for saccade detection
        min_amplitude: Minimum saccade amplitude in degrees

    Returns:
        List of detected Saccade events
    """
    n_samples = len(gaze_x)
    if n_samples < 3:
        return []

    dx = np.diff(gaze_x)
    dy = np.diff(gaze_y)
    velocity = np.sqrt(dx**2 + dy**2) * sfreq
    velocity = np.concatenate([[0], velocity])

    is_saccade = velocity > velocity_threshold

    saccades = []
    in_sac = False
    sac_start = 0

    for i in range(n_samples):
        if is_saccade[i] and not in_sac:
            sac_start = i
            in_sac = True
        elif not is_saccade[i] and in_sac:
            amplitude = np.sqrt(
                (gaze_x[i - 1] - gaze_x[sac_start])**2 +
                (gaze_y[i - 1] - gaze_y[sac_start])**2
            )
            if amplitude >= min_amplitude:
                peak_vel = float(np.max(velocity[sac_start:i]))
                direction = float(np.arctan2(
                    gaze_y[i - 1] - gaze_y[sac_start],
                    gaze_x[i - 1] - gaze_x[sac_start],
                ))
                saccades.append(Saccade(
                    onset_sample=sac_start,
                    offset_sample=i,
                    duration_ms=float((i - sac_start) / sfreq * 1000),
                    amplitude=float(amplitude),
                    peak_velocity=peak_vel,
                    start_x=float(gaze_x[sac_start]),
                    start_y=float(gaze_y[sac_start]),
                    end_x=float(gaze_x[i - 1]),
                    end_y=float(gaze_y[i - 1]),
                    direction=direction,
                ))
            in_sac = False

    return saccades


def compute_pupil_features(
    pupil: np.ndarray,
    sfreq: float,
    baseline_window: Optional[tuple[int, int]] = None,
) -> dict[str, float]:
    """
    Compute pupillometry features.

    Pupil dilation reflects cognitive load, anticipation, and arousal — all
    relevant to distinguishing intentional selection from passive viewing.

    Args:
        pupil: Pupil diameter time series
        sfreq: Sampling frequency
        baseline_window: (start_sample, end_sample) for baseline correction

    Returns:
        Dict of pupil features
    """
    valid = pupil[~np.isnan(pupil)] if np.any(np.isnan(pupil)) else pupil
    if len(valid) < 2:
        return {
            "pupil_mean": 0.0, "pupil_std": 0.0, "pupil_range": 0.0,
            "pupil_change": 0.0, "pupil_velocity_mean": 0.0,
        }

    # Baseline correction
    if baseline_window is not None:
        baseline = np.mean(valid[baseline_window[0]:baseline_window[1]])
        valid = valid - baseline

    features = {
        "pupil_mean": float(np.mean(valid)),
        "pupil_std": float(np.std(valid)),
        "pupil_range": float(np.max(valid) - np.min(valid)),
        "pupil_change": float(valid[-1] - valid[0]),
    }

    # Pupil velocity (rate of dilation/constriction)
    pupil_vel = np.abs(np.diff(valid)) * sfreq
    features["pupil_velocity_mean"] = float(np.mean(pupil_vel))

    return features


def compute_gaze_dispersion(gaze_x: np.ndarray, gaze_y: np.ndarray) -> dict[str, float]:
    """
    Compute spatial dispersion metrics of the gaze pattern.

    Tight gaze clustering indicates focused attention on a target (intent),
    while dispersed gaze suggests exploratory viewing (observation).
    """
    if len(gaze_x) < 2:
        return {"dispersion_rms": 0.0, "dispersion_bcea": 0.0, "range_x": 0.0, "range_y": 0.0}

    # RMS dispersion from centroid
    cx, cy = np.mean(gaze_x), np.mean(gaze_y)
    distances = np.sqrt((gaze_x - cx)**2 + (gaze_y - cy)**2)

    # Bivariate Contour Ellipse Area (BCEA) — standard fixation stability metric
    std_x = np.std(gaze_x)
    std_y = np.std(gaze_y)
    rho = np.corrcoef(gaze_x, gaze_y)[0, 1] if len(gaze_x) > 2 else 0
    if np.isnan(rho):
        rho = 0
    # BCEA for 68% probability region
    k = 1.0  # 1 SD
    bcea = 2 * k * np.pi * std_x * std_y * np.sqrt(1 - rho**2)

    return {
        "dispersion_rms": float(np.sqrt(np.mean(distances**2))),
        "dispersion_bcea": float(bcea),
        "range_x": float(np.max(gaze_x) - np.min(gaze_x)),
        "range_y": float(np.max(gaze_y) - np.min(gaze_y)),
    }


def compute_aoi_features(
    gaze_x: np.ndarray,
    gaze_y: np.ndarray,
    target_x: float,
    target_y: float,
    aoi_radius: float = 2.0,
    sfreq: float = 500.0,
) -> dict[str, float]:
    """
    Compute Area-of-Interest (AOI) features relative to a target position.

    AOI metrics quantify the gaze-target relationship critical for
    determining whether gaze is directed at an interaction target.

    Args:
        gaze_x, gaze_y: Gaze position time series (degrees)
        target_x, target_y: Target position (degrees)
        aoi_radius: AOI radius in degrees visual angle
        sfreq: Sampling frequency

    Returns:
        Dict of AOI features
    """
    distances = np.sqrt((gaze_x - target_x)**2 + (gaze_y - target_y)**2)
    in_aoi = distances <= aoi_radius
    n_samples = len(gaze_x)

    # Dwell time (proportion of trial spent in AOI)
    dwell_ratio = float(np.sum(in_aoi) / max(n_samples, 1))

    # Time to first entry
    aoi_entries = np.where(in_aoi)[0]
    first_entry_ms = float(aoi_entries[0] / sfreq * 1000) if len(aoi_entries) > 0 else -1.0

    # Number of entries (re-entries indicate uncertainty)
    n_entries = 0
    was_in = False
    for i in range(n_samples):
        if in_aoi[i] and not was_in:
            n_entries += 1
        was_in = in_aoi[i]

    return {
        "aoi_dwell_ratio": dwell_ratio,
        "aoi_first_entry_ms": first_entry_ms,
        "aoi_n_entries": float(n_entries),
        "aoi_mean_distance": float(np.mean(distances)),
        "aoi_min_distance": float(np.min(distances)),
    }


def extract_gaze_features(
    gaze_data: np.ndarray,
    sfreq: float = 500.0,
    velocity_threshold: float = 30.0,
    min_fixation_ms: float = 100.0,
    target_position: Optional[tuple[float, float]] = None,
    aoi_radius: float = 2.0,
) -> np.ndarray:
    """
    Extract a complete gaze feature vector for a single trial.

    Low-frequency gaze data (sfreq < 60 Hz, e.g. ALS Tobii at 30 Hz):
    - Saccade detection is skipped — at 30 Hz each sample spans 33 ms,
      shorter than a typical saccade, so I-VT velocity estimates are
      unreliable. Saccade feature slots are filled with zeros.
    - Fixation minimum duration is lowered to 133 ms (4 samples at 30 Hz).

    Args:
        gaze_data: Eye-tracking data, shape (n_samples, 3) — [x, y, pupil]
        sfreq: Sampling frequency in Hz
        velocity_threshold: Saccade detection threshold (deg/s)
        min_fixation_ms: Minimum fixation duration (ms)
        target_position: (x, y) target position for AOI features
        aoi_radius: AOI radius in degrees

    Returns:
        1D feature vector (same length regardless of sfreq, zeros for
        saccade features when sfreq < 60 Hz)
    """
    gaze_x = gaze_data[:, 0]
    gaze_y = gaze_data[:, 1]
    pupil = gaze_data[:, 2] if gaze_data.shape[1] > 2 else np.zeros(len(gaze_x))

    # Low-frequency gaze: adjust detection parameters
    low_freq_mode = sfreq < 60.0
    if low_freq_mode:
        # At 30 Hz, enforce minimum 4-sample fixation (133 ms)
        effective_min_fixation_ms = max(min_fixation_ms, 133.0)
    else:
        effective_min_fixation_ms = min_fixation_ms

    features = []

    # Fixation features
    fixations = detect_fixations(
        gaze_x, gaze_y, sfreq, velocity_threshold, effective_min_fixation_ms
    )
    if fixations:
        fix_durations = [f.duration_ms for f in fixations]
        fix_dispersions = [f.dispersion for f in fixations]
        features.extend([
            len(fixations),                          # n_fixations
            float(np.mean(fix_durations)),            # mean_fixation_duration
            float(np.max(fix_durations)),             # max_fixation_duration
            float(np.std(fix_durations)),             # std_fixation_duration
            float(np.mean(fix_dispersions)),          # mean_fixation_dispersion
        ])
    else:
        features.extend([0, 0.0, 0.0, 0.0, 0.0])

    # Saccade features — skipped for low-frequency gaze (< 60 Hz)
    if low_freq_mode:
        # Saccades span < 2 samples at 30 Hz; I-VT is unreliable.
        # Return zeros to keep feature vector length consistent.
        features.extend([0, 0.0, 0.0, 0.0, 0.0])
    else:
        saccades = detect_saccades(gaze_x, gaze_y, sfreq, velocity_threshold)
        if saccades:
            sac_amplitudes = [s.amplitude for s in saccades]
            sac_velocities = [s.peak_velocity for s in saccades]
            features.extend([
                len(saccades),                            # n_saccades
                float(np.mean(sac_amplitudes)),            # mean_saccade_amplitude
                float(np.max(sac_amplitudes)),             # max_saccade_amplitude
                float(np.mean(sac_velocities)),            # mean_peak_velocity
                float(np.max(sac_velocities)),             # max_peak_velocity
            ])
        else:
            features.extend([0, 0.0, 0.0, 0.0, 0.0])

    # Pupil features
    pupil_feats = compute_pupil_features(pupil, sfreq)
    features.extend([
        pupil_feats["pupil_mean"],
        pupil_feats["pupil_std"],
        pupil_feats["pupil_range"],
        pupil_feats["pupil_change"],
        pupil_feats["pupil_velocity_mean"],
    ])

    # Gaze dispersion features
    disp_feats = compute_gaze_dispersion(gaze_x, gaze_y)
    features.extend([
        disp_feats["dispersion_rms"],
        disp_feats["dispersion_bcea"],
        disp_feats["range_x"],
        disp_feats["range_y"],
    ])

    # AOI features (if target position known)
    if target_position is not None:
        aoi_feats = compute_aoi_features(
            gaze_x, gaze_y, target_position[0], target_position[1], aoi_radius, sfreq
        )
        features.extend([
            aoi_feats["aoi_dwell_ratio"],
            aoi_feats["aoi_first_entry_ms"],
            aoi_feats["aoi_n_entries"],
            aoi_feats["aoi_mean_distance"],
            aoi_feats["aoi_min_distance"],
        ])

    return np.array(features, dtype=np.float64)
