#!/usr/bin/env python3
"""
Real-Time Demo Runner

Runs the complete real-time pipeline:
1. Loads trained model
2. Generates/loads trial data for replay (synthetic or real EEGEyeNet data)
3. Starts Flask dashboard (auto-opens browser)
4. Replays trials through synthetic LSL streams
5. Runs real-time inference with visualization

Usage:
    # Run with synthetic data and auto-trained model
    python scripts/run_demo.py

    # Run with real EEGEyeNet data
    python scripts/run_demo.py --data-dir data/raw/eegeyenet --paradigm prosaccade

    # Run with a saved model
    python scripts/run_demo.py --model models/saved/best_model.pkl

    # Slow down for demo recording (0.5 = half speed)
    python scripts/run_demo.py --speed 0.5

    # Custom settings
    python scripts/run_demo.py --port 8080 --n-trials 50 --no-browser
"""

import argparse
import logging
import pickle
import sys
import time
import webbrowser
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.eegeyenet_loader import generate_synthetic_dataset, load_eegeyenet_matlab
from src.features.feature_pipeline import FeatureMode, extract_dataset_features
from src.models.baseline import build_random_forest
from src.streaming.lsl_replay import SynchronizedReplay
from src.streaming.realtime_inference import RealtimeInferenceEngine
from src.ui.dashboard import DemoDashboard

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_or_train_model(model_path: str | None = None):
    """Load a saved model or train a quick one on synthetic data."""
    if model_path and Path(model_path).exists():
        logger.info(f"Loading model from {model_path}")
        with open(model_path, "rb") as f:
            return pickle.load(f)

    logger.info("No saved model found — training a quick Random Forest on synthetic data...")
    dataset = generate_synthetic_dataset(n_subjects=10, trials_per_subject=60)
    X, y, _ = extract_dataset_features(dataset, mode=FeatureMode.FUSED)
    X = np.nan_to_num(X)

    model = build_random_forest(n_estimators=100, max_depth=8)
    model.fit(X, y)
    logger.info(f"Quick model trained on {len(y)} trials")

    # Save for future use
    save_dir = Path("models/saved")
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "best_model.pkl", "wb") as f:
        pickle.dump(model, f)

    return model


def main():
    parser = argparse.ArgumentParser(description="Run real-time demo")
    parser.add_argument("--model", type=str, default="models/saved/best_model.pkl",
                       help="Path to trained model")
    parser.add_argument("--data-dir", type=str, default=None,
                       help="Path to EEGEyeNet data directory (uses real data instead of synthetic)")
    parser.add_argument("--paradigm", default="prosaccade",
                       help="Which paradigm to load (default: prosaccade)")
    parser.add_argument("--port", type=int, default=5000,
                       help="Dashboard port")
    parser.add_argument("--n-trials", type=int, default=30,
                       help="Number of trials to replay")
    parser.add_argument("--speed", type=float, default=1.0,
                       help="Replay speed multiplier (0.5 = half speed, 2.0 = double)")
    parser.add_argument("--no-realtime", action="store_true",
                       help="Run as fast as possible (no pacing)")
    parser.add_argument("--no-dashboard", action="store_true",
                       help="Skip launching the web dashboard")
    parser.add_argument("--no-browser", action="store_true",
                       help="Don't auto-open the browser")
    parser.add_argument("--confidence-threshold", type=float, default=0.7,
                       help="Minimum confidence for intent prediction")
    parser.add_argument("--clean", action="store_true",
                       help="Clear saved models and output before running (forces retrain)")
    args = parser.parse_args()

    if args.clean:
        import shutil
        for d in [Path("models/saved"), Path("output"), Path("data/synthetic")]:
            if d.exists():
                shutil.rmtree(d)
                logger.info(f"Removed {d}/")

    # Load model
    model = load_or_train_model(args.model)

    # Load demo trials — real data or synthetic
    if args.data_dir:
        logger.info(f"Loading real data from {args.data_dir} ({args.paradigm})...")
        dataset = load_eegeyenet_matlab(args.data_dir, paradigm=args.paradigm)
        if dataset.n_trials == 0:
            logger.warning("No real data found — falling back to synthetic")
            dataset = generate_synthetic_dataset(
                n_subjects=3,
                trials_per_subject=max(args.n_trials // 3, 10),
            )
        else:
            # Randomly sample n_trials from the dataset
            rng = np.random.RandomState(42)
            if dataset.n_trials > args.n_trials:
                indices = rng.choice(dataset.n_trials, args.n_trials, replace=False)
                dataset.trials = [dataset.trials[i] for i in indices]
            logger.info(f"Loaded {dataset.n_trials} real trials for demo replay")
    else:
        logger.info(f"Generating {args.n_trials} synthetic demo trials...")
        dataset = generate_synthetic_dataset(
            n_subjects=3,
            trials_per_subject=max(args.n_trials // 3, 10),
        )

    demo_trials = dataset.trials[:args.n_trials]
    true_labels = [t.label for t in demo_trials]

    logger.info(
        f"Demo trials: {len(demo_trials)} total "
        f"({sum(true_labels)} intent, {len(true_labels) - sum(true_labels)} observe)"
    )

    # Start dashboard
    dashboard = None
    if not args.no_dashboard:
        try:
            dashboard = DemoDashboard(port=args.port)
            dashboard.start()
            time.sleep(1)  # Let server start

            url = f"http://127.0.0.1:{args.port}"
            logger.info(f"Dashboard: {url}")

            if not args.no_browser:
                logger.info("Opening dashboard in browser...")
                webbrowser.open(url)
                time.sleep(1)  # Let browser connect

        except Exception as e:
            logger.warning(f"Could not start dashboard: {e}")

    # Set up inference engine
    engine = RealtimeInferenceEngine(
        model=model,
        sfreq=500.0,
        window_samples=500,
        step_samples=125,
        n_eeg_channels=demo_trials[0].eeg_data.shape[0],
        confidence_threshold=args.confidence_threshold,
    )

    prediction_count = [0]
    # Track per-trial predictions for accurate accuracy calculation
    trial_predictions = []  # list of (prediction, true_label)

    def on_prediction(tick):
        prediction_count[0] += 1

        # Send to dashboard
        if dashboard is not None:
            dashboard.send_prediction(tick)

            # Send latency stats periodically
            if prediction_count[0] % 5 == 0:
                stats = engine.latency_tracker.get_stats()
                dashboard.send_latency_stats(stats)

            # Send accuracy (per-trial, not per-window)
            if trial_predictions:
                correct = sum(1 for p, t in trial_predictions if p == t)
                acc = correct / len(trial_predictions)
                dashboard.send_accuracy(acc, len(trial_predictions))

        # Console output
        label = "INTENT " if tick.prediction == 1 else "observe"
        logger.info(
            f"  [{prediction_count[0]:4d}] {label} "
            f"conf={tick.confidence:.3f} "
            f"gaze=({tick.gaze_x:+.1f}, {tick.gaze_y:+.1f}) "
            f"latency={tick.latency_ms:.1f}ms"
        )

    engine.on_prediction(on_prediction)

    # Set up synchronized replay
    replay = SynchronizedReplay(
        n_eeg_channels=demo_trials[0].eeg_data.shape[0],
        sfreq=500.0,
        chunk_size=32,
    )

    def on_chunk(eeg_chunk, gaze_chunk, timestamp):
        engine.push_data(eeg_chunk, gaze_chunk)

    current_trial_label = [None]

    def on_trial_end(trial_idx, trial):
        # Determine this trial's majority prediction from the inference history
        # (take the last few predictions made during this trial)
        history = engine.get_history()
        if history and current_trial_label[0] is not None:
            # Use the most confident prediction from this trial's windows
            # (predictions since last trial boundary)
            recent = history[-(prediction_count[0] - len(trial_predictions) * 3):]
            if recent:
                # Majority vote from recent predictions
                intent_votes = sum(1 for t in recent[-4:] if t.prediction == 1)
                trial_pred = 1 if intent_votes > len(recent[-4:]) / 2 else 0
                trial_predictions.append((trial_pred, current_trial_label[0]))

        current_trial_label[0] = trial.label

        # Reset the inference buffer at trial boundaries to prevent
        # cross-trial contamination
        engine.reset_buffer()

        label = "INTENT" if trial.label == 1 else "OBSERVE"
        logger.info(f"\n--- Trial {trial_idx + 1}/{len(demo_trials)} ({label}) ---")
        if dashboard is not None:
            dashboard.send_trial_info(trial_idx, len(demo_trials))

    replay.on_chunk(on_chunk)
    replay.on_trial_end(on_trial_end)

    # Apply speed multiplier to the replay chunk interval
    original_sfreq = 500.0
    if args.speed != 1.0 and not args.no_realtime:
        # Adjust the effective sample rate so chunks are pushed slower/faster
        replay.sfreq = original_sfreq * args.speed
        replay.eeg_stream.sfreq = original_sfreq * args.speed
        replay.gaze_stream.sfreq = original_sfreq * args.speed

    # Start replay
    replay.start()
    logger.info("\n" + "=" * 60)
    logger.info("REAL-TIME DEMO STARTING")
    if args.speed != 1.0:
        logger.info(f"Speed: {args.speed}x")
    logger.info("=" * 60 + "\n")

    try:
        replay.replay_trials(demo_trials, realtime=not args.no_realtime)
    except KeyboardInterrupt:
        logger.info("\nDemo interrupted by user")
    finally:
        replay.stop()

    # Final statistics
    stats = engine.latency_tracker.get_stats()

    logger.info("\n" + "=" * 60)
    logger.info("DEMO COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total predictions: {prediction_count[0]}")
    if trial_predictions:
        correct = sum(1 for p, t in trial_predictions if p == t)
        logger.info(f"Trial-level accuracy: {correct}/{len(trial_predictions)} = {correct/len(trial_predictions):.1%}")
    if stats:
        logger.info(f"Mean latency:      {stats['total']['mean_ms']:.1f} ms")
        logger.info(f"P95 latency:       {stats['total']['p95_ms']:.1f} ms")
        feat = stats.get("feature_extraction", {})
        model_lat = stats.get("model_inference", {})
        logger.info(f"  Feature extract:   {feat.get('mean_ms', 0):.1f} ms")
        logger.info(f"  Model inference:   {model_lat.get('mean_ms', 0):.1f} ms")

    # Keep dashboard alive so user can still view it
    if dashboard is not None:
        logger.info(f"\nDashboard still running at http://127.0.0.1:{args.port}")
        logger.info("Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("\nShutting down.")


if __name__ == "__main__":
    main()
