"""
EEG Feature Extraction

Extracts spectral band power features from multichannel EEG data for intent
classification. Implements multiple estimation methods (Welch's periodogram,
autoregressive modeling, FFT) with particular focus on the frequency bands
most relevant to the gaze-guided intent decoding problem:

- **Theta (4-7 Hz):** Frontal-midline theta — attentional engagement
- **Alpha (8-12 Hz):** Posterior alpha — suppressed during visual processing (ERD)
- **Mu (8-13 Hz):** Sensorimotor mu — desynchronizes during motor preparation
- **Beta (13-30 Hz):** Central beta — motor planning and execution

The Stimulus-Preceding Negativity (SPN) is captured as a time-domain feature
from occipitoparietal channels, as described in Reddy et al. (CHI 2024).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy import signal as scipy_signal

logger = logging.getLogger(__name__)

# Standard frequency bands
BANDS = {
    "theta": (4, 7),
    "alpha": (8, 12),
    "mu": (8, 13),
    "beta": (13, 30),
    "low_gamma": (30, 45),
}


def compute_band_power_welch(
    eeg: np.ndarray,
    sfreq: float,
    bands: Optional[dict[str, tuple[float, float]]] = None,
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
) -> dict[str, np.ndarray]:
    """
    Compute spectral band power using Welch's method.

    Args:
        eeg: EEG data, shape (n_channels, n_samples)
        sfreq: Sampling frequency in Hz
        bands: Dict of {band_name: (low_freq, high_freq)}
        nperseg: Samples per Welch segment (default: sfreq // 2)
        noverlap: Overlap between segments (default: nperseg // 2)

    Returns:
        Dict mapping band names to arrays of shape (n_channels,) with
        log-transformed band power values (dB)
    """
    if bands is None:
        bands = BANDS

    n_channels, n_samples = eeg.shape
    if nperseg is None:
        nperseg = min(int(sfreq), n_samples)
    if noverlap is None:
        noverlap = nperseg // 2

    nperseg = min(nperseg, n_samples)
    noverlap = min(noverlap, nperseg - 1)

    freqs, psd = scipy_signal.welch(
        eeg, fs=sfreq, nperseg=nperseg, noverlap=noverlap, axis=-1
    )  # psd: (n_channels, n_freqs)

    band_power = {}
    for band_name, (f_low, f_high) in bands.items():
        freq_mask = (freqs >= f_low) & (freqs <= f_high)
        if not np.any(freq_mask):
            band_power[band_name] = np.zeros(n_channels)
            continue
        # Average power in band (log-transformed for normalization)
        bp = np.mean(psd[:, freq_mask], axis=1)
        band_power[band_name] = 10 * np.log10(bp + 1e-12)

    return band_power


def compute_band_power_ar(
    eeg: np.ndarray,
    sfreq: float,
    bands: Optional[dict[str, tuple[float, float]]] = None,
    order: int = 16,
    nfft: int = 256,
) -> dict[str, np.ndarray]:
    """
    Compute spectral band power using autoregressive (Burg) estimation.

    AR methods provide better frequency resolution for short epochs compared
    to FFT-based approaches — important for real-time applications where the
    analysis window may be < 1 second.

    Args:
        eeg: EEG data, shape (n_channels, n_samples)
        sfreq: Sampling frequency in Hz
        bands: Dict of {band_name: (low_freq, high_freq)}
        order: AR model order (default 16, suitable for 256-500 Hz data)
        nfft: Number of FFT points for AR spectrum evaluation

    Returns:
        Dict mapping band names to arrays of shape (n_channels,) with
        log-transformed band power values (dB)
    """
    if bands is None:
        bands = BANDS

    n_channels, n_samples = eeg.shape
    freqs = np.linspace(0, sfreq / 2, nfft // 2 + 1)

    psd = np.zeros((n_channels, len(freqs)))
    for ch in range(n_channels):
        ar_coeffs = _burg_ar(eeg[ch], order)
        psd[ch] = _ar_spectrum(ar_coeffs, nfft, sfreq)

    band_power = {}
    for band_name, (f_low, f_high) in bands.items():
        freq_mask = (freqs >= f_low) & (freqs <= f_high)
        if not np.any(freq_mask):
            band_power[band_name] = np.zeros(n_channels)
            continue
        bp = np.mean(psd[:, freq_mask], axis=1)
        band_power[band_name] = 10 * np.log10(bp + 1e-12)

    return band_power


def _burg_ar(x: np.ndarray, order: int) -> np.ndarray:
    """Estimate AR coefficients using the Burg method."""
    n = len(x)
    # Initialize forward/backward prediction errors
    ef = x.copy().astype(np.float64)
    eb = x.copy().astype(np.float64)
    a = np.zeros(order + 1, dtype=np.float64)
    a[0] = 1.0

    for m in range(1, order + 1):
        # Compute reflection coefficient
        efm = ef[m:]
        ebm = eb[m - 1:-1]
        num = -2.0 * np.dot(efm, ebm)
        den = np.dot(efm, efm) + np.dot(ebm, ebm)
        if den < 1e-12:
            break
        k = num / den

        # Update AR coefficients
        a_new = a.copy()
        for i in range(1, m + 1):
            a_new[i] = a[i] + k * a[m - i]
        a = a_new

        # Update prediction errors
        ef_new = ef[m:] + k * eb[m - 1:-1]
        eb_new = eb[m - 1:-1] + k * ef[m:]
        ef[m:] = ef_new
        eb[m - 1:-1] = eb_new

    return a[:order + 1]


def _ar_spectrum(ar_coeffs: np.ndarray, nfft: int, sfreq: float) -> np.ndarray:
    """Compute power spectrum from AR coefficients."""
    freqs = np.linspace(0, np.pi, nfft // 2 + 1)
    order = len(ar_coeffs) - 1
    spectrum = np.zeros(len(freqs))

    for i, f in enumerate(freqs):
        z = np.exp(-1j * f * np.arange(order + 1))
        h = np.sum(ar_coeffs * z)
        spectrum[i] = 1.0 / (np.abs(h) ** 2 + 1e-12)

    return spectrum


def compute_band_power_fft(
    eeg: np.ndarray,
    sfreq: float,
    bands: Optional[dict[str, tuple[float, float]]] = None,
) -> dict[str, np.ndarray]:
    """
    Compute spectral band power using simple FFT.

    Fast but lower frequency resolution than Welch or AR for short epochs.
    Suitable for longer windows (> 2 seconds).
    """
    if bands is None:
        bands = BANDS

    n_channels, n_samples = eeg.shape
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sfreq)
    fft_vals = np.fft.rfft(eeg, axis=-1)
    psd = np.abs(fft_vals) ** 2 / n_samples

    band_power = {}
    for band_name, (f_low, f_high) in bands.items():
        freq_mask = (freqs >= f_low) & (freqs <= f_high)
        if not np.any(freq_mask):
            band_power[band_name] = np.zeros(n_channels)
            continue
        bp = np.mean(psd[:, freq_mask], axis=1)
        band_power[band_name] = 10 * np.log10(np.abs(bp) + 1e-12)

    return band_power


def compute_spn_amplitude(
    eeg: np.ndarray,
    sfreq: float,
    channel_indices: Optional[list[int]] = None,
    window_ms: tuple[float, float] = (-500, 0),
    stimulus_sample: Optional[int] = None,
) -> np.ndarray:
    """
    Compute Stimulus-Preceding Negativity (SPN) amplitude.

    The SPN is a slow negative ERP component maximal over occipitoparietal
    electrodes in the 500ms preceding an anticipated stimulus. It reflects
    the neural anticipation of sensory feedback and is the key signal that
    Reddy et al. (CHI 2024) used to confirm intent in their eye-brain-computer
    interface.

    Args:
        eeg: EEG data, shape (n_channels, n_samples)
        sfreq: Sampling frequency in Hz
        channel_indices: Indices of occipitoparietal channels (default: first 9)
        window_ms: Time window relative to stimulus onset for SPN measurement
        stimulus_sample: Sample index of stimulus onset (default: middle)

    Returns:
        SPN amplitude for selected channels, shape (n_selected_channels,)
    """
    n_channels, n_samples = eeg.shape

    if channel_indices is None:
        channel_indices = list(range(min(9, n_channels)))

    if stimulus_sample is None:
        stimulus_sample = n_samples // 2

    # Convert ms window to samples
    win_start = stimulus_sample + int(window_ms[0] * sfreq / 1000)
    win_end = stimulus_sample + int(window_ms[1] * sfreq / 1000)

    win_start = max(0, win_start)
    win_end = min(n_samples, win_end)

    if win_end <= win_start:
        return np.zeros(len(channel_indices))

    # SPN = mean amplitude in pre-stimulus window (negative = more SPN)
    spn = np.mean(eeg[channel_indices, win_start:win_end], axis=1)
    return spn


def compute_erds(
    eeg: np.ndarray,
    sfreq: float,
    baseline_window: tuple[int, int] = (0, None),
    active_window: tuple[int, int] = (None, None),
    bands: Optional[dict[str, tuple[float, float]]] = None,
) -> dict[str, np.ndarray]:
    """
    Compute Event-Related Desynchronization/Synchronization (ERD/ERS).

    ERD (negative values) indicates band power decrease relative to baseline,
    commonly observed in mu/beta bands during motor preparation. ERS (positive)
    indicates power increase. Central to BCI2000 cursor control paradigms.

    Returns:
        Dict mapping band names to ERD/ERS percentages per channel
    """
    if bands is None:
        bands = {"mu": (8, 13), "beta": (13, 30)}

    n_channels, n_samples = eeg.shape

    # Default windows
    mid = n_samples // 2
    if baseline_window[1] is None:
        baseline_window = (baseline_window[0], mid)
    if active_window[0] is None:
        active_window = (mid, n_samples if active_window[1] is None else active_window[1])

    baseline_data = eeg[:, baseline_window[0]:baseline_window[1]]
    active_data = eeg[:, active_window[0]:active_window[1]]

    erds = {}
    for band_name, (f_low, f_high) in bands.items():
        bp_baseline = compute_band_power_welch(baseline_data, sfreq, {band_name: (f_low, f_high)})
        bp_active = compute_band_power_welch(active_data, sfreq, {band_name: (f_low, f_high)})

        baseline_power = 10 ** (bp_baseline[band_name] / 10)
        active_power = 10 ** (bp_active[band_name] / 10)

        # ERD/ERS as percentage change
        erds[band_name] = ((active_power - baseline_power) / (baseline_power + 1e-12)) * 100

    return erds


def extract_eeg_features(
    eeg: np.ndarray,
    sfreq: float,
    method: str = "welch",
    bands: Optional[dict[str, tuple[float, float]]] = None,
    channel_groups: Optional[dict[str, list[int]]] = None,
    include_spn: bool = True,
    include_erds: bool = True,
    stimulus_sample: Optional[int] = None,
) -> np.ndarray:
    """
    Extract a complete EEG feature vector for a single trial.

    Combines band power, SPN amplitude, and ERD/ERS into a single feature
    vector suitable for classification.

    Args:
        eeg: EEG data, shape (n_channels, n_samples)
        sfreq: Sampling frequency in Hz
        method: Band power estimation method (welch, ar, fft)
        bands: Frequency bands to extract
        channel_groups: Dict of {group_name: [channel_indices]}
        include_spn: Whether to include SPN amplitude features
        include_erds: Whether to include ERD/ERS features
        stimulus_sample: Sample index of stimulus onset

    Returns:
        1D feature vector
    """
    if bands is None:
        bands = BANDS

    n_channels = eeg.shape[0]

    # Band power features
    if method == "welch":
        bp = compute_band_power_welch(eeg, sfreq, bands)
    elif method == "ar":
        bp = compute_band_power_ar(eeg, sfreq, bands)
    else:
        bp = compute_band_power_fft(eeg, sfreq, bands)

    features = []

    if channel_groups:
        # Extract per-group mean band power
        for group_name, ch_indices in channel_groups.items():
            for band_name, power in bp.items():
                valid_idx = [i for i in ch_indices if i < n_channels]
                if valid_idx:
                    features.append(np.mean(power[valid_idx]))
                    features.append(np.std(power[valid_idx]))
    else:
        # All channels
        for band_name, power in bp.items():
            features.extend(power.tolist())

    # SPN features — should use occipitoparietal channels (Reddy et al. CHI 2024)
    if include_spn:
        if channel_groups and "occipitoparietal" in channel_groups:
            spn_channels = [i for i in channel_groups["occipitoparietal"] if i < n_channels]
        else:
            spn_channels = list(range(min(9, n_channels)))
        spn = compute_spn_amplitude(
            eeg, sfreq,
            channel_indices=spn_channels,
            stimulus_sample=stimulus_sample,
        )
        features.append(np.mean(spn))
        features.append(np.min(spn))

    # ERD/ERS features — should use central (sensorimotor) channels
    if include_erds:
        erds = compute_erds(eeg, sfreq)
        if channel_groups and "central" in channel_groups:
            central_idx = [i for i in channel_groups["central"] if i < n_channels]
        else:
            central_idx = None

        for band_name, erds_vals in erds.items():
            if central_idx:
                features.append(np.mean(erds_vals[central_idx]))
            else:
                features.append(np.mean(erds_vals))

    return np.array(features, dtype=np.float64)
