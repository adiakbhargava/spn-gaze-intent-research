"""
Synthetic LSL Stream Replay

Replays EEGEyeNet trial data through Lab Streaming Layer (LSL) outlets at
realistic sample rates, simulating live EEG+eye-tracking acquisition.

This allows the full real-time pipeline to be tested end-to-end without
hardware, proving that the architecture works from acquisition through
feature extraction to live classification.

LSL (Lab Streaming Layer) is the standard middleware for real-time neural
signal streaming in BCI research, used by BCI2000, OpenViBE, and most
commercial EEG systems.
"""

from __future__ import annotations

import logging
import time
import threading
from typing import Optional

import numpy as np

from src.data.eegeyenet_loader import EEGEyeNetTrial

logger = logging.getLogger(__name__)

try:
    from pylsl import StreamInfo, StreamOutlet, local_clock
    LSL_AVAILABLE = True
except ImportError:
    LSL_AVAILABLE = False
    logger.warning("pylsl not available — LSL streaming disabled. Install with: pip install pylsl")


class EEGStreamReplay:
    """Replays EEG data as an LSL stream."""

    def __init__(
        self,
        n_channels: int = 128,
        sfreq: float = 500.0,
        chunk_size: int = 32,
        stream_name: str = "IntentPipeline_EEG",
    ):
        self.n_channels = n_channels
        self.sfreq = sfreq
        self.chunk_size = chunk_size
        self.stream_name = stream_name
        self._outlet = None
        self._thread = None
        self._running = False
        self._data_queue: list[np.ndarray] = []

    def start(self):
        """Create the LSL outlet."""
        if not LSL_AVAILABLE:
            logger.warning("LSL not available — running in simulation mode")
            return

        info = StreamInfo(
            name=self.stream_name,
            type="EEG",
            channel_count=self.n_channels,
            nominal_srate=self.sfreq,
            channel_format="float32",
            source_id=f"{self.stream_name}_replay",
        )
        self._outlet = StreamOutlet(info, chunk_size=self.chunk_size)
        logger.info(f"EEG LSL outlet created: {self.stream_name} ({self.n_channels}ch @ {self.sfreq}Hz)")

    def push_trial(self, trial: EEGEyeNetTrial, realtime: bool = True):
        """
        Push a trial's EEG data through the LSL outlet.

        Args:
            trial: Trial with EEG data to replay
            realtime: If True, pace output at realistic sample rate
        """
        eeg = trial.eeg_data  # (n_channels, n_samples)
        n_samples = eeg.shape[1]
        chunk_interval = self.chunk_size / self.sfreq

        for start in range(0, n_samples, self.chunk_size):
            end = min(start + self.chunk_size, n_samples)
            chunk = eeg[:, start:end].T.tolist()  # LSL expects (samples, channels)

            if self._outlet is not None:
                self._outlet.push_chunk(chunk)
            else:
                # Simulation mode — store for retrieval
                self._data_queue.append(eeg[:, start:end])

            if realtime:
                time.sleep(chunk_interval)

    def stop(self):
        """Stop the stream."""
        self._running = False
        self._outlet = None


class GazeStreamReplay:
    """Replays eye-tracking data as an LSL stream."""

    def __init__(
        self,
        sfreq: float = 500.0,
        chunk_size: int = 32,
        stream_name: str = "IntentPipeline_Gaze",
    ):
        self.sfreq = sfreq
        self.chunk_size = chunk_size
        self.stream_name = stream_name
        self._outlet = None
        self._data_queue: list[np.ndarray] = []

    def start(self):
        """Create the LSL outlet."""
        if not LSL_AVAILABLE:
            logger.warning("LSL not available — running in simulation mode")
            return

        info = StreamInfo(
            name=self.stream_name,
            type="Gaze",
            channel_count=3,  # x, y, pupil
            nominal_srate=self.sfreq,
            channel_format="float32",
            source_id=f"{self.stream_name}_replay",
        )
        self._outlet = StreamOutlet(info, chunk_size=self.chunk_size)
        logger.info(f"Gaze LSL outlet created: {self.stream_name} (3ch @ {self.sfreq}Hz)")

    def push_trial(self, trial: EEGEyeNetTrial, realtime: bool = True):
        """Push a trial's gaze data through the LSL outlet."""
        gaze = trial.gaze_data  # (n_samples, 3)
        n_samples = gaze.shape[0]
        chunk_interval = self.chunk_size / self.sfreq

        for start in range(0, n_samples, self.chunk_size):
            end = min(start + self.chunk_size, n_samples)
            chunk = gaze[start:end].tolist()

            if self._outlet is not None:
                self._outlet.push_chunk(chunk)
            else:
                self._data_queue.append(gaze[start:end])

            if realtime:
                time.sleep(chunk_interval)

    def stop(self):
        self._outlet = None


class SynchronizedReplay:
    """
    Synchronized replay of EEG+gaze streams.

    Replays both modalities in lock-step through separate LSL outlets,
    maintaining the temporal alignment required for multimodal fusion.
    """

    def __init__(
        self,
        n_eeg_channels: int = 128,
        sfreq: float = 500.0,
        chunk_size: int = 32,
    ):
        self.eeg_stream = EEGStreamReplay(
            n_channels=n_eeg_channels, sfreq=sfreq, chunk_size=chunk_size
        )
        self.gaze_stream = GazeStreamReplay(sfreq=sfreq, chunk_size=chunk_size)
        self.sfreq = sfreq
        self.chunk_size = chunk_size
        self._thread = None
        self._running = False
        self._current_trial_idx = 0
        self._trials: list[EEGEyeNetTrial] = []

        # Callbacks for real-time consumers
        self._on_chunk_callbacks: list = []
        self._on_trial_end_callbacks: list = []

    def start(self):
        """Initialize both LSL outlets."""
        self.eeg_stream.start()
        self.gaze_stream.start()

    def on_chunk(self, callback):
        """Register a callback for each pushed chunk: callback(eeg_chunk, gaze_chunk, timestamp)"""
        self._on_chunk_callbacks.append(callback)

    def on_trial_end(self, callback):
        """Register a callback for trial completion: callback(trial_idx, trial)"""
        self._on_trial_end_callbacks.append(callback)

    def replay_trial(self, trial: EEGEyeNetTrial, realtime: bool = True):
        """Replay a single trial synchronously."""
        eeg = trial.eeg_data
        gaze = trial.gaze_data
        n_samples = min(eeg.shape[1], gaze.shape[0])
        chunk_interval = self.chunk_size / self.sfreq

        for start in range(0, n_samples, self.chunk_size):
            if not self._running and realtime:
                break
            end = min(start + self.chunk_size, n_samples)

            eeg_chunk = eeg[:, start:end]
            gaze_chunk = gaze[start:end]

            # Push to LSL
            if self.eeg_stream._outlet is not None:
                self.eeg_stream._outlet.push_chunk(eeg_chunk.T.tolist())
            if self.gaze_stream._outlet is not None:
                self.gaze_stream._outlet.push_chunk(gaze_chunk.tolist())

            # Notify consumers
            for cb in self._on_chunk_callbacks:
                cb(eeg_chunk, gaze_chunk, time.time())

            if realtime:
                time.sleep(chunk_interval)

    def replay_trials(
        self,
        trials: list[EEGEyeNetTrial],
        realtime: bool = True,
        loop: bool = False,
    ):
        """Replay multiple trials in sequence."""
        self._trials = trials
        self._running = True
        self._current_trial_idx = 0

        while self._running:
            for idx, trial in enumerate(trials):
                if not self._running:
                    break
                self._current_trial_idx = idx
                self.replay_trial(trial, realtime=realtime)
                for cb in self._on_trial_end_callbacks:
                    cb(idx, trial)

            if not loop:
                break

        self._running = False

    def replay_async(
        self,
        trials: list[EEGEyeNetTrial],
        realtime: bool = True,
        loop: bool = False,
    ):
        """Start replay in a background thread."""
        self._thread = threading.Thread(
            target=self.replay_trials,
            args=(trials, realtime, loop),
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        """Stop replay."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.eeg_stream.stop()
        self.gaze_stream.stop()
