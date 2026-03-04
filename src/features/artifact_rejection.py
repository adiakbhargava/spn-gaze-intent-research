"""
Artifact Rejection for EEG and Gaze Data

Implements simple, threshold-based artifact rejection for single-trial
EEG and eye-tracking data. These methods are intentionally straightforward —
amplitude and gradient thresholds — matching the fast rejection stage used
in real-time BCI pipelines where computational budget is limited.

Rejection criteria:
1. **Peak amplitude:** Any EEG channel exceeds ±threshold_uv (default ±100 µV).
   Catches blinks, muscle bursts, and electrode pops.
2. **Gradient (sample-to-sample jump):** Any consecutive sample pair differs
   by more than gradient_uv (default 50 µV). Catches sudden jumps from
   loose electrodes or movement.
3. **Flat signal:** Any channel has near-zero variance (std < flat_threshold_uv,
   default 0.5 µV) indicating a dead or bridged electrode.
4. **Gaze data loss:** Proportion of NaN/zero gaze samples exceeds
   gaze_loss_threshold (default 50%). Catches blink-related data gaps.

Each rejection function returns a RejectionResult indicating whether the
trial should be kept and the reason for rejection if not.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RejectionResult:
    """Result of artifact rejection for a single trial."""
    keep: bool
    reasons: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.keep


@dataclass
class RejectionStats:
    """Aggregated rejection statistics across a dataset."""
    total_trials: int = 0
    kept_trials: int = 0
    rejected_amplitude: int = 0
    rejected_gradient: int = 0
    rejected_flat: int = 0
    rejected_gaze_loss: int = 0

    @property
    def rejected_trials(self) -> int:
        return self.total_trials - self.kept_trials

    @property
    def rejection_rate(self) -> float:
        return self.rejected_trials / self.total_trials if self.total_trials > 0 else 0.0

    def summary(self) -> str:
        return (
            f"Artifact rejection: kept {self.kept_trials}/{self.total_trials} trials "
            f"({100 * (1 - self.rejection_rate):.1f}% retention)\n"
            f"  Rejected — amplitude: {self.rejected_amplitude}, "
            f"gradient: {self.rejected_gradient}, "
            f"flat: {self.rejected_flat}, "
            f"gaze loss: {self.rejected_gaze_loss}"
        )


# ---------------------------------------------------------------------------
# Per-trial rejection checks
# ---------------------------------------------------------------------------

def check_amplitude(
    eeg: np.ndarray,
    threshold_uv: float = 100.0,
) -> Optional[str]:
    """Check if any EEG channel exceeds the amplitude threshold.

    Args:
        eeg: EEG data, shape (n_channels, n_samples). Values in microvolts.
        threshold_uv: Maximum allowed absolute amplitude.

    Returns:
        Rejection reason string, or None if the trial is clean.
    """
    peak = np.max(np.abs(eeg))
    if peak > threshold_uv:
        # Find which channel(s) violated
        chan_peaks = np.max(np.abs(eeg), axis=1)
        worst_ch = int(np.argmax(chan_peaks))
        return (
            f"amplitude: peak {peak:.1f} µV on ch {worst_ch} "
            f"exceeds ±{threshold_uv} µV"
        )
    return None


def check_gradient(
    eeg: np.ndarray,
    gradient_uv: float = 50.0,
) -> Optional[str]:
    """Check for sudden sample-to-sample jumps in EEG data.

    Args:
        eeg: EEG data, shape (n_channels, n_samples). Values in microvolts.
        gradient_uv: Maximum allowed absolute difference between consecutive samples.

    Returns:
        Rejection reason string, or None if the trial is clean.
    """
    if eeg.shape[1] < 2:
        return None
    diffs = np.abs(np.diff(eeg, axis=1))
    max_diff = np.max(diffs)
    if max_diff > gradient_uv:
        worst_ch = int(np.unravel_index(np.argmax(diffs), diffs.shape)[0])
        return (
            f"gradient: max jump {max_diff:.1f} µV on ch {worst_ch} "
            f"exceeds {gradient_uv} µV"
        )
    return None


def check_flat_channels(
    eeg: np.ndarray,
    flat_threshold_uv: float = 0.5,
    min_flat_channels: int = 1,
) -> Optional[str]:
    """Check for flat (dead/bridged) EEG channels.

    Args:
        eeg: EEG data, shape (n_channels, n_samples). Values in microvolts.
        flat_threshold_uv: Channels with std below this are considered flat.
        min_flat_channels: Number of flat channels required to reject.

    Returns:
        Rejection reason string, or None if the trial is clean.
    """
    chan_std = np.std(eeg, axis=1)
    flat_mask = chan_std < flat_threshold_uv
    n_flat = int(np.sum(flat_mask))
    if n_flat >= min_flat_channels:
        flat_indices = np.where(flat_mask)[0][:5]  # Show up to 5
        return (
            f"flat channels: {n_flat} channel(s) with std < {flat_threshold_uv} µV "
            f"(channels: {flat_indices.tolist()})"
        )
    return None


def check_gaze_loss(
    gaze: np.ndarray,
    loss_threshold: float = 0.5,
) -> Optional[str]:
    """Check for excessive gaze data loss (NaN or zero samples).

    Gaze data typically has shape (n_samples, n_features) where features
    are [x, y, pupil]. During blinks, trackers often report NaN or (0, 0).

    Args:
        gaze: Gaze data, shape (n_samples, n_features) or (n_features, n_samples).
        loss_threshold: Maximum fraction of lost samples (0.0-1.0).

    Returns:
        Rejection reason string, or None if the trial is clean.
    """
    if gaze is None or gaze.size == 0:
        return None

    # Handle both (n_samples, n_features) and (n_features, n_samples) layouts
    # Gaze typically has 2-3 features (x, y, pupil), so the smaller axis is features
    if gaze.ndim == 2 and gaze.shape[0] < gaze.shape[1]:
        # (n_features, n_samples) — just check x,y (first 2 rows)
        xy = gaze[:2, :]
        n_samples = gaze.shape[1]
    elif gaze.ndim == 2:
        # (n_samples, n_features) — check x,y columns
        xy = gaze[:, :2]
        n_samples = gaze.shape[0]
    else:
        return None

    # Count lost samples: NaN or both x and y are exactly 0
    if xy.ndim == 2:
        nan_mask = np.any(np.isnan(xy), axis=0 if xy.shape[0] < xy.shape[1] else 1)
        zero_mask = np.all(xy == 0, axis=0 if xy.shape[0] < xy.shape[1] else 1)
        n_lost = int(np.sum(nan_mask | zero_mask))
    else:
        return None

    loss_rate = n_lost / n_samples if n_samples > 0 else 0.0
    if loss_rate > loss_threshold:
        return (
            f"gaze loss: {loss_rate:.1%} of samples lost "
            f"({n_lost}/{n_samples}), threshold {loss_threshold:.0%}"
        )
    return None


# ---------------------------------------------------------------------------
# Combined trial rejection
# ---------------------------------------------------------------------------

def reject_trial(
    eeg: np.ndarray,
    gaze: Optional[np.ndarray] = None,
    amplitude_threshold: float = 100.0,
    gradient_threshold: float = 50.0,
    flat_threshold: float = 0.5,
    gaze_loss_threshold: float = 0.5,
    check_eeg: bool = True,
    check_gaze: bool = True,
) -> RejectionResult:
    """Run all artifact rejection checks on a single trial.

    Args:
        eeg: EEG data, shape (n_channels, n_samples).
        gaze: Gaze data (optional), shape (n_samples, n_features).
        amplitude_threshold: Max absolute EEG amplitude (µV).
        gradient_threshold: Max sample-to-sample EEG jump (µV).
        flat_threshold: Min channel std to be considered active (µV).
        gaze_loss_threshold: Max fraction of lost gaze samples.
        check_eeg: Whether to apply EEG artifact checks.
        check_gaze: Whether to apply gaze loss checks.

    Returns:
        RejectionResult with keep=True if trial passes all checks.
    """
    reasons = []

    if check_eeg and eeg is not None:
        amp_reason = check_amplitude(eeg, amplitude_threshold)
        if amp_reason:
            reasons.append(amp_reason)

        grad_reason = check_gradient(eeg, gradient_threshold)
        if grad_reason:
            reasons.append(grad_reason)

        flat_reason = check_flat_channels(eeg, flat_threshold)
        if flat_reason:
            reasons.append(flat_reason)

    if check_gaze and gaze is not None:
        gaze_reason = check_gaze_loss(gaze, gaze_loss_threshold)
        if gaze_reason:
            reasons.append(gaze_reason)

    return RejectionResult(keep=len(reasons) == 0, reasons=reasons)


# ---------------------------------------------------------------------------
# Dataset-level rejection
# ---------------------------------------------------------------------------

def reject_dataset(
    trials: list,
    amplitude_threshold: float = 100.0,
    gradient_threshold: float = 50.0,
    flat_threshold: float = 0.5,
    gaze_loss_threshold: float = 0.5,
    max_rejection_rate: float = 0.5,
    verbose: bool = True,
) -> tuple[list, RejectionStats]:
    """Apply artifact rejection to a list of EEGEyeNetTrial objects.

    Args:
        trials: List of EEGEyeNetTrial objects.
        amplitude_threshold: Max absolute EEG amplitude (µV).
        gradient_threshold: Max sample-to-sample EEG jump (µV).
        flat_threshold: Min channel std to be considered active (µV).
        gaze_loss_threshold: Max fraction of lost gaze samples.
        max_rejection_rate: If rejection rate exceeds this, log a warning
            suggesting threshold adjustment. Does NOT override rejection.
        verbose: Whether to log per-trial rejection details.

    Returns:
        Tuple of (clean_trials, stats).
    """
    stats = RejectionStats(total_trials=len(trials))
    clean_trials = []

    for i, trial in enumerate(trials):
        result = reject_trial(
            eeg=trial.eeg_data,
            gaze=trial.gaze_data,
            amplitude_threshold=amplitude_threshold,
            gradient_threshold=gradient_threshold,
            flat_threshold=flat_threshold,
            gaze_loss_threshold=gaze_loss_threshold,
        )

        if result.keep:
            clean_trials.append(trial)
            stats.kept_trials += 1
        else:
            # Categorize the rejection reasons
            for reason in result.reasons:
                if reason.startswith("amplitude"):
                    stats.rejected_amplitude += 1
                elif reason.startswith("gradient"):
                    stats.rejected_gradient += 1
                elif reason.startswith("flat"):
                    stats.rejected_flat += 1
                elif reason.startswith("gaze"):
                    stats.rejected_gaze_loss += 1

            if verbose and stats.rejected_trials <= 10:
                logger.debug(
                    f"Trial {i} rejected: {'; '.join(result.reasons)}"
                )

    # Log summary
    logger.info(stats.summary())

    if stats.rejection_rate > max_rejection_rate:
        logger.warning(
            f"High rejection rate ({stats.rejection_rate:.1%}) — consider relaxing "
            f"thresholds. Current: amplitude={amplitude_threshold} µV, "
            f"gradient={gradient_threshold} µV, flat={flat_threshold} µV, "
            f"gaze_loss={gaze_loss_threshold:.0%}"
        )

    return clean_trials, stats
