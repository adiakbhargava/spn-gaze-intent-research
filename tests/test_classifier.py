"""Tests for classification models."""

import numpy as np
import pytest

from src.data.eegeyenet_loader import generate_synthetic_dataset
from src.features.feature_pipeline import FeatureMode, extract_dataset_features
from src.models.baseline import (
    build_logistic_regression,
    build_random_forest,
    build_svm,
    evaluate_model,
    get_all_baselines,
    run_ablation,
    format_ablation_table,
)


@pytest.fixture(scope="module")
def feature_data():
    """Generate features once for all tests in this module."""
    dataset = generate_synthetic_dataset(
        n_subjects=4, trials_per_subject=20,
        n_channels=16, n_samples=256,
    )
    X_fused, y, subj = extract_dataset_features(dataset, mode=FeatureMode.FUSED, verbose=False)
    X_eeg, _, _ = extract_dataset_features(dataset, mode=FeatureMode.EEG_ONLY, verbose=False)
    X_gaze, _, _ = extract_dataset_features(dataset, mode=FeatureMode.GAZE_ONLY, verbose=False)
    X_fused = np.nan_to_num(X_fused)
    X_eeg = np.nan_to_num(X_eeg)
    X_gaze = np.nan_to_num(X_gaze)
    return X_fused, X_eeg, X_gaze, y, subj


class TestBaselineModels:
    def test_logistic_regression(self, feature_data):
        X, _, _, y, _ = feature_data
        model = build_logistic_regression()
        model.fit(X, y)
        preds = model.predict(X)
        assert set(preds).issubset({0, 1})

    def test_svm(self, feature_data):
        X, _, _, y, _ = feature_data
        model = build_svm()
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(y), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_random_forest(self, feature_data):
        X, _, _, y, _ = feature_data
        model = build_random_forest(n_estimators=20)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(y)

    def test_get_all_baselines(self):
        models = get_all_baselines()
        assert len(models) == 3
        assert "Logistic Regression" in models
        assert "SVM (RBF)" in models
        assert "Random Forest" in models


class TestEvaluation:
    def test_evaluate_model_metrics(self, feature_data):
        X, _, _, y, subj = feature_data
        model = build_logistic_regression()
        results = evaluate_model(model, X, y, cv_folds=3, subject_ids=subj)
        assert "accuracy" in results
        assert "f1" in results
        assert 0 <= results["accuracy"] <= 1
        assert 0 <= results["f1"] <= 1

    def test_evaluate_with_auc(self, feature_data):
        X, _, _, y, _ = feature_data
        model = build_logistic_regression()
        results = evaluate_model(model, X, y, cv_folds=3)
        if "auc_roc" in results:
            assert 0 <= results["auc_roc"] <= 1

    def test_per_subject_cv(self, feature_data):
        X, _, _, y, subj = feature_data
        model = build_logistic_regression()
        results = evaluate_model(model, X, y, cv_folds=3, subject_ids=subj)
        if "per_subject" in results:
            ps = results["per_subject"]
            assert "accuracy_mean" in ps
            assert "f1_mean" in ps


class TestAblation:
    def test_run_ablation(self, feature_data):
        _, X_eeg, X_gaze, y, subj = feature_data
        results = run_ablation(X_eeg, X_gaze, y, subject_ids=subj, cv_folds=3)
        # Should have results for each model
        assert len(results) > 0
        for model_name, modalities in results.items():
            assert "EEG-only" in modalities
            assert "Gaze-only" in modalities
            assert "Fused (EEG+Gaze)" in modalities

    def test_format_ablation_table(self, feature_data):
        _, X_eeg, X_gaze, y, _ = feature_data
        results = run_ablation(X_eeg, X_gaze, y, cv_folds=3)
        table = format_ablation_table(results)
        assert isinstance(table, str)
        assert "EEG-only" in table
        assert "Gaze-only" in table
        assert "Fused" in table
