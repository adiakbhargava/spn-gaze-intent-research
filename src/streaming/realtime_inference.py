"""
Real-Time Inference Engine

Processes incoming EEG+gaze streams in real-time, extracting features and
running the trained fusion classifier to produce live intent predictions
with latency tracking.

This is the core of Module 4 — demonstrating that the offline-trained
pipeline can operate in a streaming context with bounded latency, which
is the critical engineering requirement for a product like Axion Click.
"""

from __future__ import annotations

import logging
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.features.eeg_features import extract_eeg_features
from src.features.gaze_features import extract_gaze_features

logger = logging.getLogger(__name__)


@dataclass
class InferenceTick:
    """A single inference result with timing information."""
    timestamp: float
    prediction: int       # 0 = observe, 1 = intent
    confidence: float     # probability of intent class
    gaze_x: float         # current gaze position
    gaze_y: float
    latency_ms: float     # end-to-end inference latency
    feature_latency_ms: float  # feature extraction time
    model_latency_ms: float    # model inference time


@dataclass
class LatencyTracker:
    """Tracks inference latency statistics in real-time."""
    _latencies: deque = field(default_factory=lambda: deque(maxlen=1000))
    _feature_latencies: deque = field(default_factory=lambda: deque(maxlen=1000))
    _model_latencies: deque = field(default_factory=lambda: deque(maxlen=1000))

    def record(self, total_ms: float, feature_ms: float, model_ms: float):
        self._latencies.append(total_ms)
        self._feature_latencies.append(feature_ms)
        self._model_latencies.append(model_ms)

    def get_stats(self) -> dict:
        if not self._latencies:
            return {}
        lats = np.array(self._latencies)
        feat_lats = np.array(self._feature_latencies)
        model_lats = np.array(self._model_latencies)
        return {
            "total": {
                "mean_ms": float(np.mean(lats)),
                "median_ms": float(np.median(lats)),
                "p95_ms": float(np.percentile(lats, 95)),
                "max_ms": float(np.max(lats)),
                "n_samples": len(lats),
            },
            "feature_extraction": {
                "mean_ms": float(np.mean(feat_lats)),
                "median_ms": float(np.median(feat_lats)),
            },
            "model_inference": {
                "mean_ms": float(np.mean(model_lats)),
                "median_ms": float(np.median(model_lats)),
            },
        }


class RealtimeInferenceEngine:
    """
    Real-time inference engine for EEG+gaze fusion.

    Maintains a sliding window buffer of incoming samples, extracts features
    at configurable intervals, and runs the trained classifier.

    Architecture:
        LSL streams → ring buffer → feature extraction → classifier → output
                                                                       ↓
                                                              latency tracker
    """

    def __init__(
        self,
        model,
        sfreq: float = 500.0,
        window_samples: int = 500,
        step_samples: int = 125,
        n_eeg_channels: int = 128,
        confidence_threshold: float = 0.7,
        eeg_method: str = "welch",
    ):
        """
        Args:
            model: Trained classifier (sklearn Pipeline or NeuralTrainer)
            sfreq: Sampling frequency
            window_samples: Size of the analysis window in samples
            step_samples: Step size between consecutive windows
            n_eeg_channels: Number of EEG channels
            confidence_threshold: Minimum confidence for "intent" prediction
            eeg_method: Feature extraction method
        """
        self.model = model
        self.sfreq = sfreq
        self.window_samples = window_samples
        self.step_samples = step_samples
        self.n_eeg_channels = n_eeg_channels
        self.confidence_threshold = confidence_threshold
        self.eeg_method = eeg_method

        # Ring buffers
        self._eeg_buffer = np.zeros((n_eeg_channels, window_samples * 2))
        self._gaze_buffer = np.zeros((window_samples * 2, 3))
        self._buffer_pos = 0
        self._samples_since_last_inference = 0

        # State
        self.latency_tracker = LatencyTracker()
        self._running = False
        self._callbacks: list = []
        self._history: list[InferenceTick] = []

    def on_prediction(self, callback):
        """Register callback: callback(InferenceTick)"""
        self._callbacks.append(callback)

    def reset_buffer(self):
        """Clear the ring buffer. Call this at trial boundaries to prevent
        cross-trial contamination — a window straddling two trials would
        mix intent and observe data, producing garbage features."""
        self._eeg_buffer[:] = 0
        self._gaze_buffer[:] = 0
        self._buffer_pos = 0
        self._samples_since_last_inference = 0

    def push_data(self, eeg_chunk: np.ndarray, gaze_chunk: np.ndarray):
        """
        Push new data into the buffer and run inference if enough samples.

        Args:
            eeg_chunk: (n_channels, n_samples)
            gaze_chunk: (n_samples, 3)
        """
        n_new = eeg_chunk.shape[1]
        buf_len = self._eeg_buffer.shape[1]

        # Shift buffer and append new data
        if self._buffer_pos + n_new > buf_len:
            shift = n_new
            self._eeg_buffer[:, :-shift] = self._eeg_buffer[:, shift:]
            self._gaze_buffer[:-shift] = self._gaze_buffer[shift:]
            self._buffer_pos = buf_len - shift

        end = self._buffer_pos + n_new
        self._eeg_buffer[:, self._buffer_pos:end] = eeg_chunk[:, :n_new]
        gaze_samples = min(n_new, gaze_chunk.shape[0])
        self._gaze_buffer[self._buffer_pos:self._buffer_pos + gaze_samples] = gaze_chunk[:gaze_samples]
        self._buffer_pos = end
        self._samples_since_last_inference += n_new

        # Run inference at step intervals
        if (self._samples_since_last_inference >= self.step_samples
                and self._buffer_pos >= self.window_samples):
            self._run_inference()
            self._samples_since_last_inference = 0

    def _run_inference(self):
        """Extract features and classify the current window."""
        t_start = time.perf_counter()

        # Extract window
        win_end = self._buffer_pos
        win_start = win_end - self.window_samples
        eeg_window = self._eeg_buffer[:, win_start:win_end].copy()
        gaze_window = self._gaze_buffer[win_start:win_end].copy()

        # Feature extraction
        t_feat_start = time.perf_counter()
        eeg_feats = extract_eeg_features(
            eeg_window, self.sfreq, method=self.eeg_method
        )
        gaze_feats = extract_gaze_features(gaze_window, sfreq=self.sfreq)
        features = np.concatenate([eeg_feats, gaze_feats]).reshape(1, -1)
        t_feat_end = time.perf_counter()

        # Handle NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # Classification
        t_model_start = time.perf_counter()
        try:
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(features)[0]
                confidence = float(proba[1]) if len(proba) > 1 else float(proba[0])
            else:
                confidence = 0.5
            prediction = 1 if confidence >= self.confidence_threshold else 0
        except Exception as e:
            logger.warning(f"Inference error: {e}")
            confidence = 0.5
            prediction = 0
        t_model_end = time.perf_counter()

        # Current gaze position (last sample)
        current_gaze = gaze_window[-1]

        # Timing
        feature_ms = (t_feat_end - t_feat_start) * 1000
        model_ms = (t_model_end - t_model_start) * 1000
        total_ms = (t_model_end - t_start) * 1000

        self.latency_tracker.record(total_ms, feature_ms, model_ms)

        tick = InferenceTick(
            timestamp=time.time(),
            prediction=prediction,
            confidence=confidence,
            gaze_x=float(current_gaze[0]),
            gaze_y=float(current_gaze[1]),
            latency_ms=total_ms,
            feature_latency_ms=feature_ms,
            model_latency_ms=model_ms,
        )
        self._history.append(tick)

        for cb in self._callbacks:
            cb(tick)

    def get_history(self) -> list[InferenceTick]:
        return self._history

    def get_accuracy(self, true_labels: Optional[list[int]] = None) -> Optional[float]:
        """Compute running accuracy if ground truth is available."""
        if true_labels is None or not self._history:
            return None
        predictions = [t.prediction for t in self._history]
        n = min(len(predictions), len(true_labels))
        if n == 0:
            return None
        correct = sum(p == t for p, t in zip(predictions[:n], true_labels[:n]))
        return correct / n
