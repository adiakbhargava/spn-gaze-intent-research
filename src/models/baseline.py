"""
Baseline Classification Models

Implements traditional ML classifiers for intent vs. observation classification
using hand-crafted EEG+gaze features:

- Logistic Regression — interpretable baseline, good for feature importance
- Support Vector Machine (RBF kernel) — strong on moderate-dimensional data
- Random Forest — handles nonlinear feature interactions, feature importance

All models support scikit-learn's standard interface for easy comparison
and per-subject cross-validation.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

logger = logging.getLogger(__name__)


def build_logistic_regression(C: float = 1.0, max_iter: int = 1000) -> Pipeline:
    """Build a Logistic Regression pipeline with standardization and PCA."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95, random_state=42)),
        ("clf", LogisticRegression(
            C=C, max_iter=max_iter, solver="lbfgs", random_state=42
        )),
    ])


def build_svm(C: float = 1.0, kernel: str = "rbf", gamma: str = "scale") -> Pipeline:
    """Build an SVM pipeline with standardization and PCA."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95, random_state=42)),
        ("clf", SVC(
            C=C, kernel=kernel, gamma=gamma, probability=True, random_state=42
        )),
    ])


def build_random_forest(
    n_estimators: int = 200,
    max_depth: Optional[int] = 10,
) -> Pipeline:
    """Build a Random Forest pipeline (no standardization needed)."""
    return Pipeline([
        ("clf", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1,
        )),
    ])


def get_all_baselines(config: Optional[dict] = None) -> dict[str, Pipeline]:
    """Build all baseline models with optional config overrides."""
    config = config or {}
    lr_cfg = config.get("logistic_regression", {})
    svm_cfg = config.get("svm", {})
    rf_cfg = config.get("random_forest", {})

    return {
        "Logistic Regression": build_logistic_regression(
            C=lr_cfg.get("C", 1.0),
            max_iter=lr_cfg.get("max_iter", 1000),
        ),
        "SVM (RBF)": build_svm(
            C=svm_cfg.get("C", 1.0),
            kernel=svm_cfg.get("kernel", "rbf"),
            gamma=svm_cfg.get("gamma", "scale"),
        ),
        "Random Forest": build_random_forest(
            n_estimators=rf_cfg.get("n_estimators", 200),
            max_depth=rf_cfg.get("max_depth", 10),
        ),
    }


def evaluate_model(
    model: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5,
    subject_ids: Optional[np.ndarray] = None,
) -> dict:
    """
    Evaluate a model using cross-validation.

    Supports two CV strategies:
    1. Standard stratified K-fold (default)
    2. Leave-one-subject-out (if subject_ids provided and n_subjects >= cv_folds)

    Per-subject evaluation is critical for BCI research — Raj's CHI 2024
    paper explicitly uses per-subject analysis because neural signals vary
    dramatically across individuals.

    Args:
        model: scikit-learn Pipeline to evaluate
        X: Feature matrix (n_samples, n_features)
        y: Labels (n_samples,)
        cv_folds: Number of CV folds
        subject_ids: Subject IDs for per-subject CV

    Returns:
        Dict with evaluation metrics
    """
    results = {}

    # Per-subject cross-validation (leave-subjects-out)
    if subject_ids is not None:
        unique_subjects = np.unique(subject_ids)
        if len(unique_subjects) >= cv_folds:
            results["per_subject"] = _per_subject_cv(
                model, X, y, subject_ids, cv_folds
            )

    # Standard stratified K-fold
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, X, y, cv=cv, method="predict")

    try:
        y_proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")
        auc = roc_auc_score(y, y_proba[:, 1])
    except Exception:
        auc = None

    results["accuracy"] = float(accuracy_score(y, y_pred))
    results["f1"] = float(f1_score(y, y_pred, average="binary"))
    if auc is not None:
        results["auc_roc"] = float(auc)
    results["classification_report"] = classification_report(
        y, y_pred, target_names=["observe", "intent"], output_dict=True
    )

    return results


def _per_subject_cv(
    model: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    cv_folds: int,
) -> dict:
    """Leave-subjects-out cross-validation."""
    unique_subjects = np.unique(subject_ids)
    rng = np.random.RandomState(42)
    rng.shuffle(unique_subjects)

    # Split subjects into folds
    fold_size = len(unique_subjects) // cv_folds
    subject_folds = []
    for i in range(cv_folds):
        start = i * fold_size
        end = start + fold_size if i < cv_folds - 1 else len(unique_subjects)
        subject_folds.append(unique_subjects[start:end])

    fold_accs = []
    fold_f1s = []
    fold_aucs = []

    for fold_idx in range(cv_folds):
        test_subjects = set(subject_folds[fold_idx])
        test_mask = np.array([s in test_subjects for s in subject_ids])
        train_mask = ~test_mask

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        from sklearn.base import clone
        fold_model = clone(model)
        fold_model.fit(X_train, y_train)
        y_pred = fold_model.predict(X_test)

        fold_accs.append(accuracy_score(y_test, y_pred))
        fold_f1s.append(f1_score(y_test, y_pred, average="binary"))

        try:
            y_proba = fold_model.predict_proba(X_test)[:, 1]
            fold_aucs.append(roc_auc_score(y_test, y_proba))
        except Exception:
            pass

    return {
        "accuracy_mean": float(np.mean(fold_accs)) if fold_accs else 0.0,
        "accuracy_std": float(np.std(fold_accs)) if fold_accs else 0.0,
        "f1_mean": float(np.mean(fold_f1s)) if fold_f1s else 0.0,
        "f1_std": float(np.std(fold_f1s)) if fold_f1s else 0.0,
        "auc_mean": float(np.mean(fold_aucs)) if fold_aucs else 0.0,
        "auc_std": float(np.std(fold_aucs)) if fold_aucs else 0.0,
        "n_folds": len(fold_accs),
    }


def run_ablation(
    X_eeg: np.ndarray,
    X_gaze: np.ndarray,
    y: np.ndarray,
    subject_ids: Optional[np.ndarray] = None,
    cv_folds: int = 5,
    config: Optional[dict] = None,
) -> dict:
    """
    Run ablation study: EEG-only vs. gaze-only vs. fused.

    This is THE key analysis for the Axion Click thesis — demonstrating
    that multimodal EEG+gaze fusion outperforms either modality alone.

    Args:
        X_eeg: EEG feature matrix (n_samples, n_eeg_features)
        X_gaze: Gaze feature matrix (n_samples, n_gaze_features)
        y: Labels (n_samples,)
        subject_ids: Subject IDs for per-subject CV
        cv_folds: Number of CV folds
        config: Model configuration

    Returns:
        Nested dict: {model_name: {modality: {metrics}}}
    """
    X_fused = np.hstack([X_eeg, X_gaze])
    modalities = {
        "EEG-only": X_eeg,
        "Gaze-only": X_gaze,
        "Fused (EEG+Gaze)": X_fused,
    }

    models = get_all_baselines(config)
    results = {}

    for model_name, model in models.items():
        results[model_name] = {}
        for modality_name, X in modalities.items():
            logger.info(f"Evaluating {model_name} on {modality_name}...")
            from sklearn.base import clone
            m = clone(model)
            results[model_name][modality_name] = evaluate_model(
                m, X, y, cv_folds=cv_folds, subject_ids=subject_ids
            )
            logger.info(
                f"  {modality_name}: accuracy={results[model_name][modality_name]['accuracy']:.3f}, "
                f"f1={results[model_name][modality_name]['f1']:.3f}"
            )

    return results


def format_ablation_table(results: dict) -> str:
    """Format ablation results as a readable table."""
    lines = []
    lines.append(f"{'Model':<25} {'Modality':<20} {'Accuracy':>10} {'F1':>10} {'AUC-ROC':>10}")
    lines.append("-" * 77)

    for model_name, modalities in results.items():
        for modality_name, metrics in modalities.items():
            acc = f"{metrics['accuracy']:.3f}"
            f1 = f"{metrics['f1']:.3f}"
            auc = f"{metrics.get('auc_roc', 0):.3f}" if metrics.get('auc_roc') else "N/A"
            lines.append(f"{model_name:<25} {modality_name:<20} {acc:>10} {f1:>10} {auc:>10}")
        lines.append("")

    return "\n".join(lines)
