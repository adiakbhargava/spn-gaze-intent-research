#!/usr/bin/env python3
"""
Training Script

Trains baseline and neural classifiers on EEGEyeNet data (real or synthetic)
and runs the ablation study (EEG-only vs. gaze-only vs. fused).

Usage:
    # Train on synthetic data (for development)
    python scripts/train.py --synthetic

    # Train on real RSOD dataset (3 subjects for quick test)
    python scripts/train.py --dataset rsod --max-subjects 3

    # Train on full RSOD dataset (38 subjects)
    python scripts/train.py --dataset rsod

    # Train on ALS healthy dataset (20 subjects)
    python scripts/train.py --dataset als

    # Train on combined RSOD + ALS
    python scripts/train.py --dataset combined

    # Train on real EEGEyeNet data (legacy)
    python scripts/train.py --data-dir data/raw/eegeyenet

    # Run only ablation study
    python scripts/train.py --synthetic --ablation-only

    # Include neural models
    python scripts/train.py --synthetic --neural
"""

import argparse
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.eegeyenet_loader import (
    EEGEyeNetDataset,
    EEGEyeNetTrial,
    generate_synthetic_dataset,
    load_eegeyenet_matlab,
)
from src.data.eeget_loader import (
    load_rsod_dataset,
    load_als_spelling_dataset,
)
from src.features.eeg_features import extract_eeg_features
from src.features.gaze_features import extract_gaze_features
from src.features.feature_pipeline import (
    FeatureMode,
    extract_dataset_features,
)
from src.models.baseline import (
    evaluate_model,
    format_ablation_table,
    get_all_baselines,
    run_ablation,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(os.environ.get("INTENT_OUTPUT_DIR", "output"))
MODELS_DIR = Path(os.environ.get("INTENT_MODELS_DIR", "models/saved"))


# Default data paths (can be overridden via CLI arguments)
_DEFAULT_RSOD_DIR = "data/raw/eeget-rsod"
_DEFAULT_ALS_DIR  = "data/raw/eeget-als"


def load_dataset(args) -> EEGEyeNetDataset:
    """Load dataset based on arguments.

    Priority order:
    1. --synthetic: generate synthetic data
    2. --dataset {rsod,als,combined}: load real dataset(s)
    3. --data-dir: load legacy EEGEyeNet format
    4. fallback: pre-saved synthetic .npy or freshly generated synthetic
    """
    if args.synthetic:
        logger.info("Generating synthetic dataset...")
        return generate_synthetic_dataset(
            n_subjects=args.n_subjects,
            trials_per_subject=args.trials_per_subject,
        )

    # Real dataset dispatch
    dataset_choice = getattr(args, "dataset", None)
    if dataset_choice and dataset_choice != "synthetic":
        rsod_dir = getattr(args, "rsod_dir", None) or _DEFAULT_RSOD_DIR
        als_dir  = getattr(args, "als_dir",  None) or _DEFAULT_ALS_DIR
        max_sub  = getattr(args, "max_subjects", None)

        if dataset_choice == "rsod":
            logger.info(f"Loading RSOD dataset from {rsod_dir} ...")
            return load_rsod_dataset(
                rsod_dir,
                max_subjects=max_sub,
                preprocess=True,
            )

        elif dataset_choice == "als":
            logger.info(f"Loading ALS dataset from {als_dir} ...")
            return load_als_spelling_dataset(
                als_dir,
                max_subjects=max_sub,
                preprocess=True,
            )

        elif dataset_choice == "combined":
            logger.info("Loading combined RSOD + ALS dataset...")
            rsod_ds = load_rsod_dataset(
                rsod_dir, max_subjects=max_sub, preprocess=True,
            )
            als_ds = load_als_spelling_dataset(
                als_dir, max_subjects=max_sub, preprocess=True,
            )
            # Merge into a single dataset object
            combined = EEGEyeNetDataset(paradigm="combined")
            combined.trials = rsod_ds.trials + als_ds.trials
            combined.subjects = rsod_ds.subjects + [s + 100 for s in als_ds.subjects]
            logger.info(
                f"Combined: {combined.n_trials} trials "
                f"from {combined.n_subjects} subjects"
            )
            return combined

    elif args.data_dir:
        logger.info(f"Loading EEGEyeNet from {args.data_dir}...")
        return load_eegeyenet_matlab(args.data_dir, paradigm=args.paradigm)

    else:
        # Check for pre-saved synthetic data
        synth_dir = Path("data/synthetic")
        if (synth_dir / "eeg_data.npy").exists():
            logger.info("Loading pre-generated synthetic data...")
            eeg_all = np.load(synth_dir / "eeg_data.npy")
            gaze_all = np.load(synth_dir / "gaze_data.npy")
            labels = np.load(synth_dir / "labels.npy")
            subject_ids = np.load(synth_dir / "subject_ids.npy")

            dataset = EEGEyeNetDataset(paradigm="prosaccade")
            for i in range(len(labels)):
                trial = EEGEyeNetTrial(
                    subject_id=int(subject_ids[i]),
                    trial_idx=i,
                    paradigm="prosaccade",
                    condition=int(labels[i]),
                    eeg_data=eeg_all[i],
                    gaze_data=gaze_all[i],
                )
                dataset.trials.append(trial)
            return dataset
        else:
            logger.info("No data found. Generating synthetic dataset...")
            return generate_synthetic_dataset(n_subjects=20, trials_per_subject=80)


def train_baselines(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    cv_folds: int = 5,
    config: dict | None = None,
) -> dict:
    """Train and evaluate all baseline models."""
    models = get_all_baselines(config)
    results = {}

    for name, model in models.items():
        logger.info(f"\nTraining: {name}")
        t0 = time.time()
        metrics = evaluate_model(model, X, y, cv_folds=cv_folds, subject_ids=subject_ids)
        elapsed = time.time() - t0

        results[name] = metrics
        logger.info(
            f"  Accuracy: {metrics['accuracy']:.3f}, "
            f"F1: {metrics['f1']:.3f}, "
            f"AUC: {metrics.get('auc_roc', 'N/A')}"
        )
        if "per_subject" in metrics:
            ps = metrics["per_subject"]
            logger.info(
                f"  Per-subject CV: acc={ps['accuracy_mean']:.3f}±{ps['accuracy_std']:.3f}, "
                f"f1={ps['f1_mean']:.3f}±{ps['f1_std']:.3f}"
            )
        logger.info(f"  Training time: {elapsed:.1f}s")

    return results


def train_best_model(X: np.ndarray, y: np.ndarray, results: dict):
    """Train the best-performing model on all data and save it."""
    # Find best model by accuracy
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    logger.info(f"\nBest model: {best_name} (accuracy={results[best_name]['accuracy']:.3f})")

    models = get_all_baselines()
    best_pipeline = models[best_name]
    best_pipeline.fit(X, y)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "best_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(best_pipeline, f)
    logger.info(f"Best model saved to {model_path}")

    return best_pipeline, best_name


def run_ablation_study(
    dataset: EEGEyeNetDataset,
    cv_folds: int = 5,
    reject_artifacts: bool = False,
    amplitude_threshold: float = 100.0,
    gradient_threshold: float = 50.0,
) -> dict:
    """Run the full ablation study: EEG-only vs. gaze-only vs. fused."""
    logger.info("\n" + "=" * 60)
    logger.info("ABLATION STUDY: EEG-only vs. Gaze-only vs. Fused")
    logger.info("=" * 60)

    rej_kwargs = dict(
        reject_artifacts=reject_artifacts,
        amplitude_threshold=amplitude_threshold,
        gradient_threshold=gradient_threshold,
    )

    # Extract features for each modality
    logger.info("\nExtracting EEG-only features...")
    X_eeg, y, subject_ids = extract_dataset_features(
        dataset, mode=FeatureMode.EEG_ONLY, **rej_kwargs,
    )

    logger.info("Extracting gaze-only features...")
    X_gaze, _, _ = extract_dataset_features(
        dataset, mode=FeatureMode.GAZE_ONLY, **rej_kwargs,
    )

    # Run ablation
    results = run_ablation(X_eeg, X_gaze, y, subject_ids=subject_ids, cv_folds=cv_folds)

    # Print results
    table = format_ablation_table(results)
    logger.info("\n" + table)

    # Save results (filename includes dataset name if set via args)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset_tag = getattr(run_ablation_study, "_dataset_tag", "")
    result_filename = f"ablation_results{'_' + dataset_tag if dataset_tag else ''}.json"
    with open(OUTPUT_DIR / result_filename, "w") as f:
        # Convert numpy types for JSON serialization
        json.dump(_to_json_serializable(results), f, indent=2)
    logger.info(f"\nAblation results saved to {OUTPUT_DIR / result_filename}")

    return results


def _downsample_time(data: np.ndarray, factor: int) -> np.ndarray:
    """Downsample time dimension by averaging blocks.

    Args:
        data: Array of shape (N, C, T) — batch, channels, time
        factor: Downsample factor (e.g. 4 means 256→64 time steps)

    Returns:
        Downsampled array of shape (N, C, T // factor)
    """
    if factor <= 1:
        return data
    n, c, t = data.shape
    # Truncate to make evenly divisible
    t_new = (t // factor) * factor
    data = data[:, :, :t_new]
    return data.reshape(n, c, t_new // factor, factor).mean(axis=3)


def train_neural_models(
    dataset: EEGEyeNetDataset,
    epochs: int = 50,
    batch_size: int = 32,
    fast: bool = False,
) -> dict:
    """Train neural fusion models (1D-CNN and LSTM).

    Args:
        fast: If True, use CPU-optimized settings:
            - Temporal downsampling (4×) to reduce sequence length
            - Unidirectional LSTM (instead of bidirectional)
            - Smaller hidden size (32 instead of 64)
            - Early stopping (patience=7)
            - Fewer max epochs (30 instead of 50)
    """
    try:
        import torch
        from src.models.neural import Conv1DFusion, LSTMFusion, NeuralTrainer
    except ImportError:
        logger.warning("PyTorch not available — skipping neural models")
        return {}

    logger.info("\n" + "=" * 60)
    logger.info("NEURAL MODEL TRAINING" + (" (fast/CPU mode)" if fast else ""))
    logger.info("=" * 60)

    # Fast mode defaults
    if fast:
        epochs = min(epochs, 30)
        patience = 7
        downsample_factor = 4
        lstm_hidden = 32
        lstm_bidir = False
        lstm_layers = 1
        logger.info(
            f"  Fast mode: epochs≤{epochs}, patience={patience}, "
            f"downsample={downsample_factor}×, "
            f"LSTM: hidden={lstm_hidden}, unidirectional, {lstm_layers} layer"
        )
    else:
        patience = 0
        downsample_factor = 1
        lstm_hidden = 64
        lstm_bidir = True
        lstm_layers = 2

    # Prepare raw time-series data (not hand-crafted features)
    eeg_data = np.array([t.eeg_data for t in dataset.trials])  # (N, C_eeg, T)
    gaze_data = np.array([t.gaze_data.T for t in dataset.trials])  # (N, 3, T)
    labels = dataset.get_labels()
    subject_ids = dataset.get_subject_ids()

    # Simple train/val split (80/20)
    n = len(labels)
    rng = np.random.RandomState(42)
    indices = rng.permutation(n)
    split = int(0.8 * n)
    train_idx, val_idx = indices[:split], indices[split:]

    eeg_train, eeg_val = eeg_data[train_idx], eeg_data[val_idx]
    gaze_train, gaze_val = gaze_data[train_idx], gaze_data[val_idx]
    y_train, y_val = labels[train_idx], labels[val_idx]

    # Temporal downsampling for LSTM speed (applied to LSTM data only)
    if downsample_factor > 1:
        eeg_train_ds = _downsample_time(eeg_train, downsample_factor)
        eeg_val_ds = _downsample_time(eeg_val, downsample_factor)
        gaze_train_ds = _downsample_time(gaze_train, downsample_factor)
        gaze_val_ds = _downsample_time(gaze_val, downsample_factor)
        logger.info(
            f"  LSTM time steps: {eeg_data.shape[2]} → {eeg_train_ds.shape[2]}"
        )
    else:
        eeg_train_ds, eeg_val_ds = eeg_train, eeg_val
        gaze_train_ds, gaze_val_ds = gaze_train, gaze_val

    n_eeg_ch = eeg_data.shape[1]
    n_gaze_ch = gaze_data.shape[1]
    n_samples = eeg_data.shape[2]

    results = {}

    # 1D-CNN (Conv1D parallelizes over time — no downsampling needed)
    for mode in ["fused", "eeg_only", "gaze_only"]:
        model_name = f"Conv1D ({mode})"
        logger.info(f"\nTraining {model_name}...")

        model = Conv1DFusion(
            n_eeg_channels=n_eeg_ch,
            n_gaze_channels=n_gaze_ch,
            n_samples=n_samples,
            mode=mode,
        )
        trainer = NeuralTrainer(model, learning_rate=0.001)

        eeg_tr = eeg_train if mode != "gaze_only" else None
        gaze_tr = gaze_train if mode != "eeg_only" else None
        eeg_v = eeg_val if mode != "gaze_only" else None
        gaze_v = gaze_val if mode != "eeg_only" else None

        trainer.fit(
            eeg_tr, gaze_tr, y_train,
            eeg_v, gaze_v, y_val,
            epochs=epochs, batch_size=batch_size,
            patience=patience,
        )
        metrics = trainer.evaluate(eeg_v, gaze_v, y_val)
        results[model_name] = metrics
        logger.info(f"  Val accuracy: {metrics['accuracy']:.3f}, F1: {metrics['f1']:.3f}")

        # Save model weights for fused mode (needed for ONNX export to co-gateway)
        if mode == "fused":
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            pt_path = MODELS_DIR / "conv1d_fused.pt"
            torch.save(trainer.model.state_dict(), pt_path)
            logger.info(f"  Saved Conv1D (fused) weights to {pt_path}")

    # LSTM (use downsampled data for speed)
    for mode in ["fused", "eeg_only", "gaze_only"]:
        model_name = f"LSTM ({mode})"
        logger.info(f"\nTraining {model_name}...")

        model = LSTMFusion(
            n_eeg_channels=n_eeg_ch,
            n_gaze_channels=n_gaze_ch,
            hidden_size=lstm_hidden,
            n_layers=lstm_layers,
            mode=mode,
            bidirectional=lstm_bidir,
        )
        trainer = NeuralTrainer(model, learning_rate=0.001)

        eeg_tr = eeg_train_ds if mode != "gaze_only" else None
        gaze_tr = gaze_train_ds if mode != "eeg_only" else None
        eeg_v = eeg_val_ds if mode != "gaze_only" else None
        gaze_v = gaze_val_ds if mode != "eeg_only" else None

        trainer.fit(
            eeg_tr, gaze_tr, y_train,
            eeg_v, gaze_v, y_val,
            epochs=epochs, batch_size=batch_size,
            patience=patience,
        )
        metrics = trainer.evaluate(eeg_v, gaze_v, y_val)
        results[model_name] = metrics
        logger.info(f"  Val accuracy: {metrics['accuracy']:.3f}, F1: {metrics['f1']:.3f}")

        # Save model weights for fused mode (needed for ONNX export to co-gateway)
        if mode == "fused":
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            pt_path = MODELS_DIR / "lstm_fused.pt"
            torch.save(trainer.model.state_dict(), pt_path)
            logger.info(f"  Saved LSTM (fused) weights to {pt_path}")

    # Save neural results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "neural_results.json", "w") as f:
        json.dump(_to_json_serializable(results), f, indent=2)

    return results


def _to_json_serializable(obj):
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_json_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def clean_artifacts():
    """Remove all generated data, saved models, and output from previous runs."""
    import shutil
    dirs_to_clean = [
        Path("data/synthetic"),
        Path("models/saved"),
        Path("output"),
    ]
    for d in dirs_to_clean:
        if d.exists():
            shutil.rmtree(d)
            logger.info(f"Removed {d}/")
    logger.info("All previous run artifacts cleared.")


def main():
    parser = argparse.ArgumentParser(description="Train intent classifiers")
    parser.add_argument("--data-dir", type=str, default=None,
                       help="Path to EEGEyeNet data directory (legacy)")
    parser.add_argument("--paradigm", default="prosaccade")
    parser.add_argument("--synthetic", action="store_true",
                       help="Use synthetic data")

    # Real dataset selection
    parser.add_argument(
        "--dataset",
        choices=["synthetic", "rsod", "als", "combined"],
        default=None,
        help=(
            "Which real dataset to load. "
            "'rsod': EEGET-RSOD (38 subjects, 500Hz EEG, satellite images). "
            "'als': EEGET-ALS healthy (20 subjects, 128Hz EEG, spelling BCI). "
            "'combined': both datasets merged."
        ),
    )
    parser.add_argument(
        "--rsod-dir",
        type=str,
        default=None,
        help=f"Path to EEGET-RSOD dataset root (default: {_DEFAULT_RSOD_DIR})",
    )
    parser.add_argument(
        "--als-dir",
        type=str,
        default=None,
        help=f"Path to EEGET-ALS Dataset root (default: {_DEFAULT_ALS_DIR})",
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Cap number of subjects loaded (useful for quick sanity checks)",
    )

    parser.add_argument("--n-subjects", type=int, default=20)
    parser.add_argument("--trials-per-subject", type=int, default=80)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--ablation-only", action="store_true",
                       help="Only run ablation study")
    parser.add_argument("--neural", action="store_true",
                       help="Include neural model training")
    parser.add_argument("--neural-epochs", type=int, default=50)
    parser.add_argument("--fast-neural", action="store_true",
                       help="CPU-optimized neural training: fewer epochs, "
                       "early stopping, unidirectional LSTM, temporal downsampling")
    parser.add_argument("--reject-artifacts", action="store_true",
                       help="Apply artifact rejection before feature extraction "
                       "(amplitude, gradient, flat-channel, and gaze-loss checks)")
    parser.add_argument("--amplitude-threshold", type=float, default=100.0,
                       help="Max absolute EEG amplitude in µV (default: 100)")
    parser.add_argument("--gradient-threshold", type=float, default=50.0,
                       help="Max sample-to-sample EEG jump in µV (default: 50)")
    parser.add_argument("--clean", action="store_true",
                       help="Clear all generated data, models, and output before running")
    args = parser.parse_args()

    if args.clean:
        clean_artifacts()

    # Load data
    dataset = load_dataset(args)
    logger.info(f"Dataset: {dataset.n_trials} trials, {dataset.n_subjects} subjects")

    # Tag ablation output file with dataset name for easy identification
    dataset_tag = getattr(args, "dataset", None) or ("synthetic" if args.synthetic else "")
    run_ablation_study._dataset_tag = dataset_tag or ""

    # Run ablation study
    ablation_results = run_ablation_study(
        dataset,
        cv_folds=args.cv_folds,
        reject_artifacts=args.reject_artifacts,
        amplitude_threshold=args.amplitude_threshold,
        gradient_threshold=args.gradient_threshold,
    )

    if not args.ablation_only:
        # Train fused baselines
        logger.info("\n" + "=" * 60)
        logger.info("BASELINE MODEL TRAINING (Fused Features)")
        logger.info("=" * 60)

        X, y, subject_ids = extract_dataset_features(
            dataset, mode=FeatureMode.FUSED,
            reject_artifacts=args.reject_artifacts,
            amplitude_threshold=args.amplitude_threshold,
            gradient_threshold=args.gradient_threshold,
        )
        logger.info(f"Feature matrix: {X.shape}")

        baseline_results = train_baselines(X, y, subject_ids, cv_folds=args.cv_folds)

        # Save best model
        best_model, best_name = train_best_model(X, y, baseline_results)

        # Save all results
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_DIR / "baseline_results.json", "w") as f:
            json.dump(_to_json_serializable(baseline_results), f, indent=2)

    # Neural models
    if args.neural or args.fast_neural:
        neural_results = train_neural_models(
            dataset, epochs=args.neural_epochs, fast=args.fast_neural,
        )

    logger.info("\nTraining complete!")


if __name__ == "__main__":
    main()
