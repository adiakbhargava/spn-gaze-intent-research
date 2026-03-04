"""Tests for data loading modules."""

import numpy as np
import pytest

from src.data.eegeyenet_loader import (
    EEGEyeNetDataset,
    EEGEyeNetTrial,
    generate_synthetic_dataset,
)


class TestSyntheticDataset:
    def test_generate_basic(self):
        dataset = generate_synthetic_dataset(n_subjects=3, trials_per_subject=10)
        assert dataset.n_trials == 30
        assert dataset.n_subjects == 3

    def test_trial_structure(self):
        dataset = generate_synthetic_dataset(n_subjects=2, trials_per_subject=10)
        trial = dataset.trials[0]
        assert isinstance(trial, EEGEyeNetTrial)
        assert trial.eeg_data.shape[0] == 128  # n_channels
        assert trial.eeg_data.shape[1] == 500  # n_samples
        assert trial.gaze_data.shape == (500, 3)  # (n_samples, [x,y,pupil])

    def test_label_balance(self):
        dataset = generate_synthetic_dataset(n_subjects=4, trials_per_subject=20)
        labels = dataset.get_labels()
        n_intent = np.sum(labels == 1)
        n_observe = np.sum(labels == 0)
        assert n_intent == n_observe  # Half and half

    def test_subject_split(self):
        dataset = generate_synthetic_dataset(n_subjects=5, trials_per_subject=10)
        for subj_id in range(5):
            trials = dataset.get_subject_trials(subj_id)
            assert len(trials) == 10

    def test_condition_split(self):
        dataset = generate_synthetic_dataset(n_subjects=2, trials_per_subject=20)
        splits = dataset.split_by_condition()
        assert "intent" in splits
        assert "observe" in splits
        assert len(splits["intent"]) + len(splits["observe"]) == dataset.n_trials

    def test_eeg_realistic_range(self):
        """EEG amplitudes should be in a realistic microvolt range."""
        dataset = generate_synthetic_dataset(n_subjects=2, trials_per_subject=10)
        for trial in dataset.trials[:5]:
            max_amp = np.max(np.abs(trial.eeg_data))
            assert max_amp < 500, f"EEG amplitude {max_amp} uV too large"
            assert max_amp > 1, f"EEG amplitude {max_amp} uV too small"

    def test_gaze_data_structure(self):
        """Gaze data should have x, y, pupil columns."""
        dataset = generate_synthetic_dataset(n_subjects=2, trials_per_subject=10)
        for trial in dataset.trials[:5]:
            assert trial.gaze_data.ndim == 2
            assert trial.gaze_data.shape[1] == 3

    def test_reproducibility(self):
        """Same random_state should produce identical datasets."""
        d1 = generate_synthetic_dataset(n_subjects=2, trials_per_subject=10, random_state=123)
        d2 = generate_synthetic_dataset(n_subjects=2, trials_per_subject=10, random_state=123)
        np.testing.assert_array_equal(d1.trials[0].eeg_data, d2.trials[0].eeg_data)

    def test_custom_params(self):
        dataset = generate_synthetic_dataset(
            n_subjects=2,
            trials_per_subject=5,
            n_channels=16,
            n_samples=256,
            sfreq=256.0,
        )
        assert dataset.trials[0].eeg_data.shape == (16, 256)
        assert dataset.trials[0].eeg_sfreq == 256.0
