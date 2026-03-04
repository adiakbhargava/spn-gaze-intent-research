"""Tests for feature extraction modules."""

import numpy as np
import pytest

from src.features.eeg_features import (
    compute_band_power_welch,
    compute_band_power_ar,
    compute_band_power_fft,
    compute_spn_amplitude,
    compute_erds,
    extract_eeg_features,
)
from src.features.gaze_features import (
    detect_fixations,
    detect_saccades,
    compute_pupil_features,
    compute_gaze_dispersion,
    extract_gaze_features,
)
from src.features.feature_pipeline import (
    FeatureMode,
    extract_trial_features,
    extract_dataset_features,
)
from src.features.artifact_rejection import (
    check_amplitude,
    check_gradient,
    check_flat_channels,
    check_gaze_loss,
    reject_trial,
    reject_dataset,
    RejectionResult,
    RejectionStats,
)
from src.data.eegeyenet_loader import generate_synthetic_dataset


class TestEEGFeatures:
    def setup_method(self):
        rng = np.random.RandomState(42)
        self.sfreq = 500.0
        self.n_channels = 16
        self.n_samples = 500
        # Generate signal with known alpha peak
        t = np.arange(self.n_samples) / self.sfreq
        self.eeg = rng.randn(self.n_channels, self.n_samples) * 5
        # Add 10 Hz alpha to all channels
        for ch in range(self.n_channels):
            self.eeg[ch] += 20 * np.sin(2 * np.pi * 10 * t)

    def test_welch_band_power_shape(self):
        bp = compute_band_power_welch(self.eeg, self.sfreq)
        for band_name, power in bp.items():
            assert power.shape == (self.n_channels,), f"{band_name} wrong shape"

    def test_welch_alpha_peak(self):
        """Alpha band should have highest power given 10 Hz signal."""
        bp = compute_band_power_welch(self.eeg, self.sfreq)
        alpha_mean = np.mean(bp["alpha"])
        theta_mean = np.mean(bp["theta"])
        assert alpha_mean > theta_mean, "Alpha should dominate given 10Hz signal"

    def test_ar_band_power(self):
        bp = compute_band_power_ar(self.eeg, self.sfreq)
        assert "alpha" in bp
        assert bp["alpha"].shape == (self.n_channels,)

    def test_fft_band_power(self):
        bp = compute_band_power_fft(self.eeg, self.sfreq)
        assert "alpha" in bp
        assert bp["alpha"].shape == (self.n_channels,)

    def test_spn_amplitude(self):
        spn = compute_spn_amplitude(self.eeg, self.sfreq)
        assert spn.shape == (min(9, self.n_channels),)

    def test_erds(self):
        erds = compute_erds(self.eeg, self.sfreq)
        assert "mu" in erds
        assert "beta" in erds

    def test_extract_eeg_features_vector(self):
        feats = extract_eeg_features(self.eeg, self.sfreq, method="welch")
        assert feats.ndim == 1
        assert len(feats) > 0
        assert not np.any(np.isnan(feats))

    def test_short_window(self):
        """Should handle short windows gracefully."""
        short_eeg = self.eeg[:, :50]
        feats = extract_eeg_features(short_eeg, self.sfreq, method="welch")
        assert len(feats) > 0


class TestGazeFeatures:
    def setup_method(self):
        self.sfreq = 500.0
        self.n_samples = 500
        rng = np.random.RandomState(42)
        # Simulate a saccade from center to target
        # Note: noise must be small enough that point-to-point velocity stays
        # below the saccade threshold (30 deg/s). At 500 Hz, delta of 0.01 deg
        # gives 0.01 * 500 ≈ 5 deg/s, well below threshold.
        self.gaze_x = np.zeros(self.n_samples)
        self.gaze_y = np.zeros(self.n_samples)
        # Fixation at center (0-200) — very small jitter
        self.gaze_x[:200] = rng.randn(200) * 0.01
        self.gaze_y[:200] = rng.randn(200) * 0.01
        # Saccade (200-220)
        t_sac = np.linspace(0, 1, 20)
        self.gaze_x[200:220] = 8 * t_sac
        self.gaze_y[200:220] = 5 * t_sac
        # Fixation on target (220-500) — small jitter
        self.gaze_x[220:] = 8 + rng.randn(280) * 0.01
        self.gaze_y[220:] = 5 + rng.randn(280) * 0.01
        # Pupil
        self.pupil = 4.0 + rng.randn(self.n_samples) * 0.1
        self.gaze_data = np.column_stack([self.gaze_x, self.gaze_y, self.pupil])

    def test_detect_fixations(self):
        fixations = detect_fixations(self.gaze_x, self.gaze_y, self.sfreq)
        assert len(fixations) >= 2, "Should detect at least 2 fixations"
        # First fixation should be near center
        assert abs(fixations[0].mean_x) < 1.0
        assert abs(fixations[0].mean_y) < 1.0

    def test_detect_saccades(self):
        saccades = detect_saccades(self.gaze_x, self.gaze_y, self.sfreq)
        assert len(saccades) >= 1, "Should detect at least 1 saccade"
        # Saccade amplitude should be roughly sqrt(8^2 + 5^2) ≈ 9.4
        assert saccades[0].amplitude > 5

    def test_pupil_features(self):
        feats = compute_pupil_features(self.pupil, self.sfreq)
        assert "pupil_mean" in feats
        assert "pupil_std" in feats
        assert abs(feats["pupil_mean"] - 4.0) < 0.5

    def test_gaze_dispersion(self):
        disp = compute_gaze_dispersion(self.gaze_x, self.gaze_y)
        assert "dispersion_rms" in disp
        assert disp["dispersion_rms"] > 0

    def test_extract_gaze_features_vector(self):
        feats = extract_gaze_features(self.gaze_data, sfreq=self.sfreq)
        assert feats.ndim == 1
        assert len(feats) > 0
        assert not np.any(np.isnan(feats))


class TestFeaturePipeline:
    def setup_method(self):
        self.dataset = generate_synthetic_dataset(
            n_subjects=3, trials_per_subject=10,
            n_channels=16, n_samples=256,
        )

    def test_extract_fused(self):
        X, y, subj = extract_dataset_features(self.dataset, mode=FeatureMode.FUSED, verbose=False)
        assert X.shape[0] == 30  # 3 subjects * 10 trials
        assert X.shape[1] > 0
        assert y.shape == (30,)
        assert subj.shape == (30,)

    def test_extract_eeg_only(self):
        X, y, _ = extract_dataset_features(self.dataset, mode=FeatureMode.EEG_ONLY, verbose=False)
        assert X.shape[0] == 30

    def test_extract_gaze_only(self):
        X, y, _ = extract_dataset_features(self.dataset, mode=FeatureMode.GAZE_ONLY, verbose=False)
        assert X.shape[0] == 30

    def test_fused_wider_than_unimodal(self):
        """Fused features should have more dimensions than either modality alone."""
        X_fused, _, _ = extract_dataset_features(self.dataset, mode=FeatureMode.FUSED, verbose=False)
        X_eeg, _, _ = extract_dataset_features(self.dataset, mode=FeatureMode.EEG_ONLY, verbose=False)
        X_gaze, _, _ = extract_dataset_features(self.dataset, mode=FeatureMode.GAZE_ONLY, verbose=False)
        assert X_fused.shape[1] > X_eeg.shape[1]
        assert X_fused.shape[1] > X_gaze.shape[1]
        assert X_fused.shape[1] == X_eeg.shape[1] + X_gaze.shape[1]

    def test_no_nans(self):
        X, _, _ = extract_dataset_features(self.dataset, mode=FeatureMode.FUSED, verbose=False)
        assert not np.any(np.isnan(X))


class TestArtifactRejection:
    """Tests for artifact rejection module."""

    def setup_method(self):
        self.n_channels = 16
        self.n_samples = 500
        self.sfreq = 500.0
        rng = np.random.RandomState(42)
        # Clean EEG: smooth, band-limited signal well within thresholds.
        # Using cumulative sum of small noise produces realistic-ish EEG
        # with small gradients (each sample differs by ~1-3 µV).
        raw = rng.randn(self.n_channels, self.n_samples) * 3.0
        # Low-pass via simple moving average to keep gradients small
        kernel = np.ones(5) / 5
        self.clean_eeg = np.array([
            np.convolve(ch, kernel, mode="same") for ch in raw
        ])
        # Clean gaze: (n_samples, 3) — x, y, pupil
        self.clean_gaze = np.column_stack([
            rng.uniform(0, 1920, self.n_samples),
            rng.uniform(0, 1080, self.n_samples),
            rng.uniform(2.0, 5.0, self.n_samples),
        ])

    def test_clean_trial_passes(self):
        result = reject_trial(self.clean_eeg, self.clean_gaze)
        assert result.keep is True
        assert len(result.reasons) == 0
        assert bool(result) is True

    def test_amplitude_rejection(self):
        """Trial with a 200 µV spike should be rejected."""
        bad_eeg = self.clean_eeg.copy()
        bad_eeg[3, 100] = 200.0  # Big spike on channel 3
        reason = check_amplitude(bad_eeg, threshold_uv=100.0)
        assert reason is not None
        assert "amplitude" in reason
        assert "ch 3" in reason

    def test_amplitude_clean(self):
        reason = check_amplitude(self.clean_eeg, threshold_uv=100.0)
        assert reason is None

    def test_gradient_rejection(self):
        """Trial with a sudden 80 µV jump should be rejected."""
        bad_eeg = self.clean_eeg.copy()
        bad_eeg[5, 200] = bad_eeg[5, 199] + 80.0  # Sudden jump
        reason = check_gradient(bad_eeg, gradient_uv=50.0)
        assert reason is not None
        assert "gradient" in reason

    def test_gradient_clean(self):
        reason = check_gradient(self.clean_eeg, gradient_uv=50.0)
        assert reason is None

    def test_flat_channel_rejection(self):
        """Trial with a dead (constant) channel should be rejected."""
        bad_eeg = self.clean_eeg.copy()
        bad_eeg[7, :] = 0.0  # Dead channel
        reason = check_flat_channels(bad_eeg, flat_threshold_uv=0.5)
        assert reason is not None
        assert "flat" in reason

    def test_flat_channel_clean(self):
        reason = check_flat_channels(self.clean_eeg, flat_threshold_uv=0.5)
        assert reason is None

    def test_gaze_loss_rejection(self):
        """Trial with >50% NaN gaze samples should be rejected."""
        bad_gaze = self.clean_gaze.copy()
        bad_gaze[: self.n_samples // 2 + 10, :2] = np.nan  # >50% NaN
        reason = check_gaze_loss(bad_gaze, loss_threshold=0.5)
        assert reason is not None
        assert "gaze loss" in reason

    def test_gaze_loss_clean(self):
        reason = check_gaze_loss(self.clean_gaze, loss_threshold=0.5)
        assert reason is None

    def test_gaze_loss_zero_coords(self):
        """Gaze (0,0) should be treated as lost data."""
        bad_gaze = self.clean_gaze.copy()
        bad_gaze[: self.n_samples // 2 + 10, :2] = 0.0  # >50% zeros
        reason = check_gaze_loss(bad_gaze, loss_threshold=0.5)
        assert reason is not None

    def test_reject_trial_multiple_reasons(self):
        """A badly corrupted trial can fail multiple checks."""
        bad_eeg = self.clean_eeg.copy()
        bad_eeg[0, 50] = 300.0  # amplitude violation
        bad_eeg[1, :] = 0.0  # flat channel
        result = reject_trial(bad_eeg, self.clean_gaze)
        assert result.keep is False
        assert len(result.reasons) >= 2

    def test_reject_trial_eeg_only(self):
        """check_gaze=False should skip gaze checks."""
        bad_gaze = np.full_like(self.clean_gaze, np.nan)
        result = reject_trial(self.clean_eeg, bad_gaze, check_gaze=False)
        assert result.keep is True

    def test_reject_dataset(self):
        """reject_dataset should filter out bad trials."""
        dataset = generate_synthetic_dataset(n_subjects=3, trials_per_subject=20)

        # Corrupt a few trials
        for trial in dataset.trials[:5]:
            trial.eeg_data[0, 50] = 500.0  # Massive spike

        clean, stats = reject_dataset(dataset.trials, amplitude_threshold=100.0)
        assert len(clean) < len(dataset.trials)
        assert stats.total_trials == len(dataset.trials)
        assert stats.kept_trials == len(clean)
        assert stats.rejected_amplitude >= 5
        assert stats.rejection_rate > 0

    def test_rejection_stats_summary(self):
        stats = RejectionStats(total_trials=100, kept_trials=85,
                               rejected_amplitude=10, rejected_gradient=5)
        summary = stats.summary()
        assert "85/100" in summary
        assert "85.0%" in summary

    def test_extract_with_rejection(self):
        """extract_dataset_features with reject_artifacts=True should work."""
        dataset = generate_synthetic_dataset(n_subjects=3, trials_per_subject=20)

        # Corrupt some trials with massive spikes that clearly exceed threshold
        for trial in dataset.trials[:5]:
            trial.eeg_data[0, 50] = 5000.0  # Way above any threshold

        # Use relaxed thresholds so only the spiked trials are rejected
        # (synthetic data has realistic gradients that we don't want to reject)
        X_clean, y_clean, _ = extract_dataset_features(
            dataset, mode=FeatureMode.FUSED, verbose=False,
            reject_artifacts=True,
            amplitude_threshold=500.0,
            gradient_threshold=500.0,
            flat_threshold=0.01,
            gaze_loss_threshold=0.9,
        )
        X_all, y_all, _ = extract_dataset_features(
            dataset, mode=FeatureMode.FUSED, verbose=False,
            reject_artifacts=False,
        )
        # Artifact rejection should produce fewer trials
        assert X_clean.shape[0] < X_all.shape[0]
        assert X_clean.shape[0] == y_clean.shape[0]
