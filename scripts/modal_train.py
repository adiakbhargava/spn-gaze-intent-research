#!/usr/bin/env python3
"""
Modal Cloud Training for spn-gaze-intent-research.

Runs the ablation study and model training on Modal's serverless
infrastructure instead of local compute. Synthetic data is generated
in-memory on the remote machine — no data upload needed.

Setup (one-time):
    pip install modal
    modal setup

Usage:
    # Baseline ablation (CPU, ~2 min)
    modal run scripts/modal_train.py

    # More subjects
    modal run scripts/modal_train.py --n-subjects 50

    # Include neural models (GPU)
    modal run scripts/modal_train.py --neural

    # Fast neural (CPU-optimized, no GPU needed)
    modal run scripts/modal_train.py --fast-neural

    # Full run: 50 subjects + neural on GPU
    modal run scripts/modal_train.py --n-subjects 50 --neural

Results are saved to a Modal Volume and downloaded to local output/.
"""

import modal

# ---------------------------------------------------------------------------
# Modal configuration
# ---------------------------------------------------------------------------

app = modal.App("spn-gaze-intent-research")

# Persistent volume for training output (survives across runs)
output_volume = modal.Volume.from_name("intent-output", create_if_missing=True)

# Container image with training deps + project source code
training_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "mne>=1.5.0",
        "scikit-learn>=1.3.0",
        "torch>=2.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "h5py>=3.9.0",
    )
    # Add project source code to the container
    .add_local_dir("src", remote_path="/app/src")
    .add_local_dir("scripts", remote_path="/app/scripts")
    .add_local_dir("configs", remote_path="/app/configs")
)


# ---------------------------------------------------------------------------
# Remote training function
# ---------------------------------------------------------------------------


@app.function(
    image=training_image,
    volumes={"/results": output_volume},
    gpu="T4",
    timeout=3600,
    memory=8192,
)
def train_remote(
    n_subjects: int = 20,
    trials_per_subject: int = 80,
    cv_folds: int = 5,
    neural: bool = False,
    fast_neural: bool = False,
    neural_epochs: int = 50,
    reject_artifacts: bool = False,
    amplitude_threshold: float = 100.0,
    gradient_threshold: float = 50.0,
) -> dict:
    """Run training on Modal's cloud infrastructure.

    Generates synthetic data in-memory, runs ablation study and
    optionally trains neural models. Results are written to the
    Modal Volume at /results/ and returned as a dict.
    """
    import json
    import logging
    import os
    import sys
    import time

    # Set up paths so imports work
    sys.path.insert(0, "/app")
    os.environ["INTENT_OUTPUT_DIR"] = "/results"
    os.environ["INTENT_MODELS_DIR"] = "/results/models"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("modal_train")

    # Import project modules (inside function so they run in the container)
    from src.data.eegeyenet_loader import generate_synthetic_dataset
    from src.features.feature_pipeline import FeatureMode, extract_dataset_features
    from src.models.baseline import (
        evaluate_model,
        format_ablation_table,
        get_all_baselines,
        run_ablation,
    )

    t_start = time.time()
    all_results = {}

    # ── Generate synthetic data ──────────────────────────────────────────
    logger.info(f"Generating synthetic dataset: {n_subjects} subjects, "
                f"{trials_per_subject} trials each")
    dataset = generate_synthetic_dataset(
        n_subjects=n_subjects,
        trials_per_subject=trials_per_subject,
    )
    logger.info(f"Dataset: {dataset.n_trials} trials, {dataset.n_subjects} subjects")

    # ── Ablation study ───────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("ABLATION STUDY: EEG-only vs. Gaze-only vs. Fused")
    logger.info("=" * 60)

    rej_kwargs = dict(
        reject_artifacts=reject_artifacts,
        amplitude_threshold=amplitude_threshold,
        gradient_threshold=gradient_threshold,
    )

    logger.info("Extracting EEG-only features...")
    X_eeg, y, subject_ids = extract_dataset_features(
        dataset, mode=FeatureMode.EEG_ONLY, **rej_kwargs,
    )

    logger.info("Extracting gaze-only features...")
    X_gaze, _, _ = extract_dataset_features(
        dataset, mode=FeatureMode.GAZE_ONLY, **rej_kwargs,
    )

    ablation_results = run_ablation(
        X_eeg, X_gaze, y, subject_ids=subject_ids, cv_folds=cv_folds,
    )

    table = format_ablation_table(ablation_results)
    logger.info("\n" + table)
    all_results["ablation"] = _to_json_serializable(ablation_results)

    # ── Neural models (optional) ─────────────────────────────────────────
    if neural or fast_neural:
        logger.info("\n" + "=" * 60)
        logger.info("NEURAL MODEL TRAINING" + (" (fast mode)" if fast_neural else ""))
        logger.info("=" * 60)

        # Import train.py's neural training function
        from scripts.train import train_neural_models
        neural_results = train_neural_models(
            dataset, epochs=neural_epochs, fast=fast_neural,
        )
        all_results["neural"] = _to_json_serializable(neural_results)

    # ── Save results to volume ───────────────────────────────────────────
    os.makedirs("/results", exist_ok=True)

    with open("/results/ablation_results.json", "w") as f:
        json.dump(all_results.get("ablation", {}), f, indent=2)
    logger.info("Saved /results/ablation_results.json")

    if "neural" in all_results:
        with open("/results/neural_results.json", "w") as f:
            json.dump(all_results["neural"], f, indent=2)
        logger.info("Saved /results/neural_results.json")

    # Commit volume writes
    output_volume.commit()

    elapsed = time.time() - t_start
    logger.info(f"\nTraining complete in {elapsed:.1f}s")

    all_results["_meta"] = {
        "n_subjects": n_subjects,
        "trials_per_subject": trials_per_subject,
        "cv_folds": cv_folds,
        "neural": neural or fast_neural,
        "elapsed_seconds": round(elapsed, 1),
    }

    return all_results


def _to_json_serializable(obj):
    """Convert numpy types to Python types for JSON serialization."""
    import numpy as np

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


# ---------------------------------------------------------------------------
# Local entrypoint — runs on your machine, dispatches to Modal
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(
    n_subjects: int = 20,
    trials_per_subject: int = 80,
    cv_folds: int = 5,
    neural: bool = False,
    fast_neural: bool = False,
    neural_epochs: int = 50,
    reject_artifacts: bool = False,
    amplitude_threshold: float = 100.0,
    gradient_threshold: float = 50.0,
):
    """Run training on Modal and download results locally."""
    import json
    from pathlib import Path

    print(f"Dispatching to Modal cloud...")
    print(f"  Subjects: {n_subjects}")
    print(f"  Trials/subject: {trials_per_subject}")
    print(f"  CV folds: {cv_folds}")
    print(f"  Neural: {neural or fast_neural}")
    print()

    # Call the remote function
    results = train_remote.remote(
        n_subjects=n_subjects,
        trials_per_subject=trials_per_subject,
        cv_folds=cv_folds,
        neural=neural,
        fast_neural=fast_neural,
        neural_epochs=neural_epochs,
        reject_artifacts=reject_artifacts,
        amplitude_threshold=amplitude_threshold,
        gradient_threshold=gradient_threshold,
    )

    # Download results to local output/
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    if "ablation" in results:
        with open(output_dir / "ablation_results.json", "w") as f:
            json.dump(results["ablation"], f, indent=2)
        print(f"Saved {output_dir / 'ablation_results.json'}")

    if "neural" in results:
        with open(output_dir / "neural_results.json", "w") as f:
            json.dump(results["neural"], f, indent=2)
        print(f"Saved {output_dir / 'neural_results.json'}")

    # Print summary
    meta = results.get("_meta", {})
    print(f"\nDone in {meta.get('elapsed_seconds', '?')}s on Modal")

    # Print ablation table if available
    if "ablation" in results:
        print("\nAblation Results:")
        print("-" * 70)
        for model_name, modalities in results["ablation"].items():
            if isinstance(modalities, dict):
                for modality, metrics in modalities.items():
                    if isinstance(metrics, dict) and "accuracy" in metrics:
                        print(
                            f"  {model_name:25s} {modality:10s}  "
                            f"acc={metrics['accuracy']:.3f}  "
                            f"f1={metrics.get('f1', 0):.3f}  "
                            f"auc={metrics.get('auc_roc', 0):.3f}"
                        )
