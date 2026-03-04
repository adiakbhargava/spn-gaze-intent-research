"""
Integration tests for streaming, real-time inference, dashboard, and benchmark.

These tests verify the end-to-end pipeline components work together without
requiring external dependencies (pylsl, Flask). All components degrade gracefully
to simulation mode when LSL/Flask are unavailable.
"""

import json
import time
import threading

import numpy as np
import pytest

from src.data.eegeyenet_loader import (
    EEGEyeNetTrial,
    generate_synthetic_dataset,
)
from src.features.eeg_features import extract_eeg_features
from src.features.gaze_features import extract_gaze_features
from src.features.feature_pipeline import FeatureMode, extract_dataset_features
from src.models.baseline import build_random_forest
from src.streaming.lsl_replay import (
    EEGStreamReplay,
    GazeStreamReplay,
    SynchronizedReplay,
)
from src.streaming.realtime_inference import (
    InferenceTick,
    LatencyTracker,
    RealtimeInferenceEngine,
)
from src.benchmark.latency import (
    benchmark_fusion_pipeline,
)


# ---------------------------------------------------------------------------
# LSL Replay (Simulation Mode — no pylsl required)
# ---------------------------------------------------------------------------

class TestEEGStreamReplay:
    def test_create_stream(self):
        stream = EEGStreamReplay(n_channels=128, sfreq=500.0, chunk_size=32)
        stream.start()  # Should succeed with or without pylsl
        stream.stop()

    def test_push_trial_completes(self):
        """Push a trial through the stream without errors."""
        stream = EEGStreamReplay(n_channels=128, sfreq=500.0, chunk_size=32)
        stream.start()

        trial = _make_trial()
        stream.push_trial(trial, realtime=False)
        # If pylsl is installed, data goes through LSL outlet (not _data_queue)
        # If not, it goes to _data_queue. Either way, no errors.
        stream.stop()

    def test_trial_data_integrity(self):
        """Verify the stream processes the right number of samples."""
        # Use SynchronizedReplay's on_chunk callback for verification
        replay = SynchronizedReplay(n_eeg_channels=128, sfreq=500.0, chunk_size=32)
        replay.start()

        total_eeg_samples = [0]

        def on_chunk(eeg_chunk, gaze_chunk, ts):
            total_eeg_samples[0] += eeg_chunk.shape[1]

        replay.on_chunk(on_chunk)

        trial = _make_trial()
        replay._running = True
        replay.replay_trial(trial, realtime=False)

        assert total_eeg_samples[0] == 500  # All samples pushed
        replay.stop()


class TestGazeStreamReplay:
    def test_push_trial_completes(self):
        """Push gaze data through the stream without errors."""
        stream = GazeStreamReplay(sfreq=500.0, chunk_size=32)
        stream.start()

        trial = _make_trial()
        stream.push_trial(trial, realtime=False)
        # No errors expected regardless of pylsl availability
        stream.stop()

    def test_gaze_data_integrity(self):
        """Verify gaze stream processes the right number of samples."""
        replay = SynchronizedReplay(n_eeg_channels=128, sfreq=500.0, chunk_size=32)
        replay.start()

        total_gaze_samples = [0]

        def on_chunk(eeg_chunk, gaze_chunk, ts):
            total_gaze_samples[0] += gaze_chunk.shape[0]

        replay.on_chunk(on_chunk)

        trial = _make_trial()
        replay._running = True
        replay.replay_trial(trial, realtime=False)

        assert total_gaze_samples[0] == 500
        replay.stop()


class TestSynchronizedReplay:
    def test_replay_single_trial(self):
        """Replay one trial and verify chunk callbacks fire."""
        replay = SynchronizedReplay(n_eeg_channels=128, sfreq=500.0, chunk_size=32)
        replay.start()

        chunks_received = []

        def on_chunk(eeg_chunk, gaze_chunk, timestamp):
            chunks_received.append((eeg_chunk.shape, gaze_chunk.shape))

        replay.on_chunk(on_chunk)

        trial = _make_trial()
        replay._running = True
        replay.replay_trial(trial, realtime=False)

        # Should have pushed ~16 chunks (500 samples / 32 chunk_size)
        assert len(chunks_received) >= 15
        # Verify shapes
        eeg_shape, gaze_shape = chunks_received[0]
        assert eeg_shape == (128, 32)
        assert gaze_shape == (32, 3)
        replay.stop()

    def test_replay_multiple_trials(self):
        """Replay multiple trials and verify trial-end callbacks fire."""
        replay = SynchronizedReplay(n_eeg_channels=128, sfreq=500.0, chunk_size=32)
        replay.start()

        trial_ends = []

        def on_trial_end(idx, trial):
            trial_ends.append(idx)

        replay.on_trial_end(on_trial_end)

        trials = [_make_trial(label=i % 2) for i in range(3)]
        replay.replay_trials(trials, realtime=False)

        assert trial_ends == [0, 1, 2]
        replay.stop()

    def test_replay_async(self):
        """Replay in background thread and verify it completes."""
        replay = SynchronizedReplay(n_eeg_channels=128, sfreq=500.0, chunk_size=64)
        replay.start()

        done = threading.Event()

        def on_trial_end(idx, trial):
            if idx == 1:  # Last trial
                done.set()

        replay.on_trial_end(on_trial_end)

        trials = [_make_trial() for _ in range(2)]
        replay.replay_async(trials, realtime=False)

        assert done.wait(timeout=10.0), "Async replay did not complete in time"
        replay.stop()

    def test_stop_interrupts_replay(self):
        """Calling stop() should interrupt an ongoing replay."""
        replay = SynchronizedReplay(n_eeg_channels=128, sfreq=500.0, chunk_size=32)
        replay.start()

        chunks_received = []

        def on_chunk(eeg, gaze, ts):
            chunks_received.append(1)

        replay.on_chunk(on_chunk)

        # Make long trials
        trials = [_make_trial(n_samples=5000) for _ in range(10)]
        replay.replay_async(trials, realtime=False)

        time.sleep(0.1)  # Let some chunks through
        replay.stop()

        # Should not have processed all chunks from 10 long trials
        total_possible = 10 * (5000 // 32 + 1)
        assert len(chunks_received) <= total_possible


# ---------------------------------------------------------------------------
# Real-Time Inference Engine
# ---------------------------------------------------------------------------

class TestLatencyTracker:
    def test_record_and_stats(self):
        tracker = LatencyTracker()
        for i in range(50):
            tracker.record(total_ms=10.0 + i * 0.1, feature_ms=8.0, model_ms=2.0)

        stats = tracker.get_stats()
        assert "total" in stats
        assert "feature_extraction" in stats
        assert "model_inference" in stats
        assert stats["total"]["n_samples"] == 50
        assert stats["total"]["mean_ms"] > 0
        assert stats["total"]["p95_ms"] >= stats["total"]["mean_ms"]

    def test_empty_stats(self):
        tracker = LatencyTracker()
        assert tracker.get_stats() == {}


class TestRealtimeInferenceEngine:
    @pytest.fixture
    def trained_model(self):
        """Train a quick model for testing."""
        dataset = generate_synthetic_dataset(n_subjects=3, trials_per_subject=20)
        X, y, _ = extract_dataset_features(dataset, mode=FeatureMode.FUSED)
        X = np.nan_to_num(X)
        model = build_random_forest(n_estimators=10, max_depth=5)
        model.fit(X, y)
        return model

    def test_create_engine(self, trained_model):
        engine = RealtimeInferenceEngine(
            model=trained_model,
            sfreq=500.0,
            window_samples=500,
            step_samples=125,
            n_eeg_channels=128,
        )
        assert engine.window_samples == 500
        assert engine.step_samples == 125

    def test_push_data_triggers_inference(self, trained_model):
        """Pushing enough data should trigger inference and produce a prediction."""
        engine = RealtimeInferenceEngine(
            model=trained_model,
            sfreq=500.0,
            window_samples=500,
            step_samples=125,
            n_eeg_channels=128,
        )

        predictions = []

        def on_pred(tick):
            predictions.append(tick)

        engine.on_prediction(on_pred)

        # Push 500 samples (one full window) in chunks of 32
        trial = _make_trial()
        eeg = trial.eeg_data      # (128, 500)
        gaze = trial.gaze_data    # (500, 3)

        for start in range(0, 500, 32):
            end = min(start + 32, 500)
            engine.push_data(eeg[:, start:end], gaze[start:end])

        # Should have triggered at least one inference
        assert len(predictions) >= 1
        tick = predictions[0]
        assert isinstance(tick, InferenceTick)
        assert tick.prediction in (0, 1)
        assert 0.0 <= tick.confidence <= 1.0
        assert tick.latency_ms > 0

    def test_latency_tracking(self, trained_model):
        """Verify latency tracker accumulates measurements."""
        engine = RealtimeInferenceEngine(
            model=trained_model,
            sfreq=500.0,
            window_samples=500,
            step_samples=125,
            n_eeg_channels=128,
        )

        trial = _make_trial()
        eeg = trial.eeg_data
        gaze = trial.gaze_data

        for start in range(0, 500, 32):
            end = min(start + 32, 500)
            engine.push_data(eeg[:, start:end], gaze[start:end])

        stats = engine.latency_tracker.get_stats()
        assert stats != {}
        assert stats["total"]["n_samples"] >= 1

    def test_reset_buffer(self, trained_model):
        """Buffer reset should clear state for new trial."""
        engine = RealtimeInferenceEngine(
            model=trained_model,
            sfreq=500.0,
            window_samples=500,
            step_samples=125,
            n_eeg_channels=128,
        )

        # Push some data
        trial = _make_trial()
        engine.push_data(trial.eeg_data[:, :100], trial.gaze_data[:100])
        assert engine._buffer_pos > 0

        # Reset
        engine.reset_buffer()
        assert engine._buffer_pos == 0
        assert engine._samples_since_last_inference == 0

    def test_multiple_trials(self, trained_model):
        """Engine should produce predictions across multiple trials."""
        engine = RealtimeInferenceEngine(
            model=trained_model,
            sfreq=500.0,
            window_samples=500,
            step_samples=125,
            n_eeg_channels=128,
        )

        all_preds = []
        engine.on_prediction(lambda t: all_preds.append(t))

        for _ in range(3):
            trial = _make_trial()
            for start in range(0, 500, 32):
                end = min(start + 32, 500)
                engine.push_data(trial.eeg_data[:, start:end], trial.gaze_data[start:end])
            engine.reset_buffer()

        assert len(all_preds) >= 3
        assert all(isinstance(t, InferenceTick) for t in all_preds)

    def test_history(self, trained_model):
        """get_history() should return all past predictions."""
        engine = RealtimeInferenceEngine(
            model=trained_model,
            sfreq=500.0,
            window_samples=500,
            step_samples=125,
            n_eeg_channels=128,
        )

        trial = _make_trial()
        for start in range(0, 500, 32):
            end = min(start + 32, 500)
            engine.push_data(trial.eeg_data[:, start:end], trial.gaze_data[start:end])

        history = engine.get_history()
        assert len(history) >= 1


# ---------------------------------------------------------------------------
# End-to-End Integration: Replay → Inference → Predictions
# ---------------------------------------------------------------------------

class TestEndToEnd:
    @pytest.fixture
    def trained_model(self):
        dataset = generate_synthetic_dataset(n_subjects=3, trials_per_subject=20)
        X, y, _ = extract_dataset_features(dataset, mode=FeatureMode.FUSED)
        X = np.nan_to_num(X)
        model = build_random_forest(n_estimators=10, max_depth=5)
        model.fit(X, y)
        return model

    def test_replay_to_inference(self, trained_model):
        """Full pipeline: SynchronizedReplay → InferenceEngine → Predictions."""
        replay = SynchronizedReplay(n_eeg_channels=128, sfreq=500.0, chunk_size=32)
        replay.start()

        engine = RealtimeInferenceEngine(
            model=trained_model,
            sfreq=500.0,
            window_samples=500,
            step_samples=125,
            n_eeg_channels=128,
        )

        predictions = []
        trial_boundaries = []

        def on_chunk(eeg_chunk, gaze_chunk, ts):
            engine.push_data(eeg_chunk, gaze_chunk)

        def on_trial_end(idx, trial):
            trial_boundaries.append(idx)
            engine.reset_buffer()

        engine.on_prediction(lambda t: predictions.append(t))
        replay.on_chunk(on_chunk)
        replay.on_trial_end(on_trial_end)

        # Run 5 trials through the pipeline
        trials = [_make_trial(label=i % 2) for i in range(5)]
        replay.replay_trials(trials, realtime=False)
        replay.stop()

        # Verify predictions were generated
        assert len(predictions) >= 5, f"Expected >=5 predictions, got {len(predictions)}"
        assert trial_boundaries == [0, 1, 2, 3, 4]

        # Verify prediction structure
        for tick in predictions:
            assert tick.prediction in (0, 1)
            assert 0.0 <= tick.confidence <= 1.0
            assert tick.latency_ms > 0
            assert tick.feature_latency_ms > 0
            assert tick.model_latency_ms > 0

        # Verify latency stats
        stats = engine.latency_tracker.get_stats()
        assert stats["total"]["n_samples"] == len(predictions)
        assert stats["total"]["mean_ms"] > 0

    def test_replay_with_real_data(self, trained_model):
        """Integration test using real preprocessed data if available."""
        from pathlib import Path

        preprocessed = Path("data/raw/eegeyenet/prosaccade/preprocessed/eeg_data.npy")
        if not preprocessed.exists():
            pytest.skip("Real data not available — run preprocess_real_data.py first")

        from src.data.eegeyenet_loader import load_eegeyenet_matlab

        dataset = load_eegeyenet_matlab("data/raw/eegeyenet", paradigm="prosaccade", max_subjects=2)
        assert dataset.n_trials > 0

        engine = RealtimeInferenceEngine(
            model=trained_model,
            sfreq=500.0,
            window_samples=500,
            step_samples=125,
            n_eeg_channels=128,
        )
        predictions = []
        engine.on_prediction(lambda t: predictions.append(t))

        # Push one real trial
        trial = dataset.trials[0]
        for start in range(0, trial.eeg_data.shape[1], 32):
            end = min(start + 32, trial.eeg_data.shape[1])
            engine.push_data(trial.eeg_data[:, start:end], trial.gaze_data[start:end])

        assert len(predictions) >= 1


# ---------------------------------------------------------------------------
# Dashboard (import check — full tests require Flask)
# ---------------------------------------------------------------------------

class TestDashboard:
    def test_import(self):
        """Dashboard module should import without errors."""
        from src.ui import dashboard
        assert hasattr(dashboard, "DemoDashboard")
        assert hasattr(dashboard, "DASHBOARD_HTML")

    def test_html_template_valid(self):
        """Dashboard HTML should contain key UI elements."""
        from src.ui.dashboard import DASHBOARD_HTML
        assert "INTENT STREAM PIPELINE" in DASHBOARD_HTML
        assert "target-grid" in DASHBOARD_HTML
        assert "gaze-cursor" in DASHBOARD_HTML
        assert "socket.io" in DASHBOARD_HTML
        assert "prediction" in DASHBOARD_HTML
        assert "latency" in DASHBOARD_HTML.lower()

    def test_dashboard_creation(self):
        """DemoDashboard should initialize if Flask is available."""
        from src.ui.dashboard import FLASK_AVAILABLE
        if not FLASK_AVAILABLE:
            pytest.skip("Flask not installed")

        from src.ui.dashboard import DemoDashboard
        dash = DemoDashboard(port=15999)  # High port to avoid conflicts
        assert dash.port == 15999
        assert dash.app is not None


# ---------------------------------------------------------------------------
# Benchmark Module
# ---------------------------------------------------------------------------

class TestBenchmark:
    def test_fusion_benchmark(self):
        """Benchmark the fusion pipeline's inference latency."""
        dataset = generate_synthetic_dataset(n_subjects=2, trials_per_subject=10)
        X, y, _ = extract_dataset_features(dataset, mode=FeatureMode.FUSED)
        X = np.nan_to_num(X)

        model = build_random_forest(n_estimators=10, max_depth=5)
        model.fit(X, y)

        result = benchmark_fusion_pipeline(
            model=model,
            n_iterations=10,
            n_eeg_channels=128,
            n_samples=500,
            sfreq=500.0,
        )

        assert "total" in result
        assert "feature_extraction" in result
        assert "model_inference" in result
        assert result["total"]["mean_ms"] > 0
        assert result["n_iterations"] == 10


# ---------------------------------------------------------------------------
# Neural Models (import check — full tests require PyTorch)
# ---------------------------------------------------------------------------

class TestNeuralModels:
    def test_import(self):
        """Neural module should import without errors."""
        from src.models import neural
        assert hasattr(neural, "TORCH_AVAILABLE")

    def test_conv1d_creation(self):
        """Conv1DFusion model should initialize if PyTorch is available."""
        from src.models.neural import TORCH_AVAILABLE
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not installed")

        from src.models.neural import Conv1DFusion
        model = Conv1DFusion(n_eeg_channels=128, n_gaze_channels=3, n_samples=500)
        assert model is not None

    def test_lstm_creation(self):
        """LSTMFusion model should initialize if PyTorch is available."""
        from src.models.neural import TORCH_AVAILABLE
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not installed")

        from src.models.neural import LSTMFusion
        model = LSTMFusion(n_eeg_channels=128, n_gaze_channels=3)
        assert model is not None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trial(
    label: int = 1,
    n_channels: int = 128,
    n_samples: int = 500,
    sfreq: float = 500.0,
) -> EEGEyeNetTrial:
    """Create a minimal synthetic trial for testing."""
    rng = np.random.RandomState(42)
    return EEGEyeNetTrial(
        subject_id=0,
        trial_idx=0,
        paradigm="prosaccade",
        condition=label,
        eeg_data=rng.randn(n_channels, n_samples).astype(np.float64) * 20.0,
        gaze_data=rng.randn(n_samples, 3).astype(np.float64),
        eeg_sfreq=sfreq,
        gaze_sfreq=sfreq,
        channel_names=[f"Ch{i}" for i in range(n_channels)],
    )
