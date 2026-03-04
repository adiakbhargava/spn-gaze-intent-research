"""
Neural Fusion Models

Implements deep learning classifiers that operate on raw windowed EEG+gaze
time series for intent decoding:

1. **Conv1DFusion:** 1D-CNN that learns temporal patterns in each modality
   separately before fusing at the classification layer. Inspired by
   EEGNet (Lawhern et al., 2018) and adapted for multimodal input.

2. **LSTMFusion:** Bidirectional LSTM that captures sequential dependencies
   in the time series. Useful when temporal ordering of neural events
   (e.g., SPN buildup → saccade → post-saccadic activity) carries
   discriminative information.

Both models support modality ablation (EEG-only, gaze-only, fused) for
direct comparison with the baseline classifiers.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available — neural models disabled")


def check_torch():
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for neural models. Install with: pip install torch"
        )


class Conv1DFusion(nn.Module):
    """
    1D-CNN for EEG+gaze time series classification.

    Architecture:
        EEG branch:  Conv1D → BatchNorm → ReLU → Conv1D → BatchNorm → ReLU → GlobalAvgPool
        Gaze branch: Conv1D → BatchNorm → ReLU → Conv1D → BatchNorm → ReLU → GlobalAvgPool
        Fusion:      Concatenate → FC → ReLU → Dropout → FC → Sigmoid
    """

    def __init__(
        self,
        n_eeg_channels: int = 128,
        n_gaze_channels: int = 3,
        n_samples: int = 500,
        n_filters: tuple[int, ...] = (32, 64),
        kernel_sizes: tuple[int, ...] = (5, 3),
        dropout: float = 0.3,
        mode: str = "fused",  # "fused", "eeg_only", "gaze_only"
    ):
        super().__init__()
        self.mode = mode

        # EEG branch
        if mode in ("fused", "eeg_only"):
            self.eeg_branch = nn.Sequential(
                nn.Conv1d(n_eeg_channels, n_filters[0], kernel_sizes[0], padding="same"),
                nn.BatchNorm1d(n_filters[0]),
                nn.ReLU(),
                nn.Conv1d(n_filters[0], n_filters[1], kernel_sizes[1], padding="same"),
                nn.BatchNorm1d(n_filters[1]),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
            )
            self.eeg_out_dim = n_filters[1]
        else:
            self.eeg_branch = None
            self.eeg_out_dim = 0

        # Gaze branch
        if mode in ("fused", "gaze_only"):
            self.gaze_branch = nn.Sequential(
                nn.Conv1d(n_gaze_channels, n_filters[0], kernel_sizes[0], padding="same"),
                nn.BatchNorm1d(n_filters[0]),
                nn.ReLU(),
                nn.Conv1d(n_filters[0], n_filters[1], kernel_sizes[1], padding="same"),
                nn.BatchNorm1d(n_filters[1]),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
            )
            self.gaze_out_dim = n_filters[1]
        else:
            self.gaze_branch = None
            self.gaze_out_dim = 0

        # Classification head
        fusion_dim = self.eeg_out_dim + self.gaze_out_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        eeg: torch.Tensor,
        gaze: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass. Accepts positional args for ONNX tracing compatibility.

        For single-modality modes, pass a dummy tensor for the unused input.
        """
        features = []

        if self.eeg_branch is not None:
            features.append(self.eeg_branch(eeg))

        if self.gaze_branch is not None:
            features.append(self.gaze_branch(gaze))

        if not features:
            raise ValueError("At least one modality must be provided")

        x = torch.cat(features, dim=1)
        return self.classifier(x)


class LSTMFusion(nn.Module):
    """
    LSTM for EEG+gaze time series classification.

    Processes each modality through separate LSTM encoders, then fuses
    the final hidden states for classification. Supports both
    unidirectional and bidirectional modes.
    """

    def __init__(
        self,
        n_eeg_channels: int = 128,
        n_gaze_channels: int = 3,
        hidden_size: int = 64,
        n_layers: int = 2,
        dropout: float = 0.3,
        mode: str = "fused",
        bidirectional: bool = True,
    ):
        super().__init__()
        self.mode = mode
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1

        if mode in ("fused", "eeg_only"):
            self.eeg_lstm = nn.LSTM(
                input_size=n_eeg_channels,
                hidden_size=hidden_size,
                num_layers=n_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if n_layers > 1 else 0,
            )
            self.eeg_out_dim = hidden_size * num_directions
        else:
            self.eeg_lstm = None
            self.eeg_out_dim = 0

        if mode in ("fused", "gaze_only"):
            self.gaze_lstm = nn.LSTM(
                input_size=n_gaze_channels,
                hidden_size=hidden_size,
                num_layers=n_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if n_layers > 1 else 0,
            )
            self.gaze_out_dim = hidden_size * num_directions
        else:
            self.gaze_lstm = None
            self.gaze_out_dim = 0

        fusion_dim = self.eeg_out_dim + self.gaze_out_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        eeg: torch.Tensor,
        gaze: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass. Accepts positional args for ONNX tracing compatibility."""
        features = []

        if self.eeg_lstm is not None:
            # eeg: (batch, n_channels, n_samples) → (batch, n_samples, n_channels)
            eeg_t = eeg.transpose(1, 2)
            _, (h_n, _) = self.eeg_lstm(eeg_t)
            if self.bidirectional:
                # Take last layer's forward+backward hidden states
                h_fwd = h_n[-2]
                h_bwd = h_n[-1]
                features.append(torch.cat([h_fwd, h_bwd], dim=1))
            else:
                features.append(h_n[-1])

        if self.gaze_lstm is not None:
            # gaze: (batch, n_channels, n_samples) → (batch, n_samples, n_channels)
            gaze_t = gaze.transpose(1, 2)
            _, (h_n, _) = self.gaze_lstm(gaze_t)
            if self.bidirectional:
                h_fwd = h_n[-2]
                h_bwd = h_n[-1]
                features.append(torch.cat([h_fwd, h_bwd], dim=1))
            else:
                features.append(h_n[-1])

        if not features:
            raise ValueError("At least one modality must be provided")

        x = torch.cat(features, dim=1)
        return self.classifier(x)


class NeuralTrainer:
    """Trainer for neural fusion models."""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        device: Optional[str] = None,
    ):
        check_torch()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}

    def fit(
        self,
        X_eeg_train: Optional[np.ndarray],
        X_gaze_train: Optional[np.ndarray],
        y_train: np.ndarray,
        X_eeg_val: Optional[np.ndarray] = None,
        X_gaze_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32,
        patience: int = 0,
    ) -> dict:
        """Train the neural model.

        Args:
            patience: Early stopping patience. If > 0, training stops when
                val_loss hasn't improved for this many epochs. 0 = disabled.
        """
        train_loader = self._make_loader(X_eeg_train, X_gaze_train, y_train, batch_size, shuffle=True)

        best_val_loss = float("inf")
        epochs_without_improvement = 0

        for epoch in range(epochs):
            # Training
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                eeg_batch, gaze_batch, y_batch = self._unpack_batch(batch)
                self.optimizer.zero_grad()
                logits = self.model(eeg_batch, gaze_batch).squeeze(-1)
                loss = self.criterion(logits, y_batch)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)
            self.history["train_loss"].append(avg_train_loss)

            # Validation
            if y_val is not None:
                val_metrics = self.evaluate(X_eeg_val, X_gaze_val, y_val, batch_size)
                self.history["val_loss"].append(val_metrics["loss"])
                self.history["val_acc"].append(val_metrics["accuracy"])

                if (epoch + 1) % 10 == 0:
                    logger.info(
                        f"Epoch {epoch + 1}/{epochs}: "
                        f"train_loss={avg_train_loss:.4f}, "
                        f"val_loss={val_metrics['loss']:.4f}, "
                        f"val_acc={val_metrics['accuracy']:.3f}"
                    )

                # Early stopping
                if patience > 0:
                    if val_metrics["loss"] < best_val_loss:
                        best_val_loss = val_metrics["loss"]
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1
                        if epochs_without_improvement >= patience:
                            logger.info(
                                f"Early stopping at epoch {epoch + 1} "
                                f"(no improvement for {patience} epochs)"
                            )
                            break

        return self.history

    def evaluate(
        self,
        X_eeg: Optional[np.ndarray],
        X_gaze: Optional[np.ndarray],
        y: np.ndarray,
        batch_size: int = 32,
    ) -> dict:
        """Evaluate model on a dataset."""
        loader = self._make_loader(X_eeg, X_gaze, y, batch_size, shuffle=False)
        self.model.eval()

        all_logits = []
        all_labels = []
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in loader:
                eeg_batch, gaze_batch, y_batch = self._unpack_batch(batch)
                logits = self.model(eeg_batch, gaze_batch).squeeze(-1)
                loss = self.criterion(logits, y_batch)
                total_loss += loss.item()
                n_batches += 1
                all_logits.append(logits.cpu().numpy())
                all_labels.append(y_batch.cpu().numpy())

        all_logits = np.concatenate(all_logits)
        all_labels = np.concatenate(all_labels)
        preds = (all_logits > 0).astype(int)
        probs = 1.0 / (1.0 + np.exp(-all_logits))

        metrics = {
            "loss": total_loss / max(n_batches, 1),
            "accuracy": float(np.mean(preds == all_labels)),
            "f1": float(f1_score_np(all_labels, preds)),
        }

        try:
            from sklearn.metrics import roc_auc_score
            metrics["auc_roc"] = float(roc_auc_score(all_labels, probs))
        except Exception:
            pass

        return metrics

    def predict_proba(
        self,
        X_eeg: Optional[np.ndarray],
        X_gaze: Optional[np.ndarray],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Get prediction probabilities."""
        dummy_y = np.zeros(
            len(X_eeg) if X_eeg is not None else len(X_gaze)
        )
        loader = self._make_loader(X_eeg, X_gaze, dummy_y, batch_size, shuffle=False)
        self.model.eval()

        all_probs = []
        with torch.no_grad():
            for batch in loader:
                eeg_batch, gaze_batch, _ = self._unpack_batch(batch)
                logits = self.model(eeg_batch, gaze_batch).squeeze(-1)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)

        return np.concatenate(all_probs)

    def _make_loader(
        self,
        X_eeg: Optional[np.ndarray],
        X_gaze: Optional[np.ndarray],
        y: np.ndarray,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        tensors = []

        if X_eeg is not None:
            tensors.append(torch.FloatTensor(X_eeg))
        else:
            # Placeholder
            n = len(y)
            tensors.append(torch.zeros(n, 1, 1))

        if X_gaze is not None:
            tensors.append(torch.FloatTensor(X_gaze))
        else:
            n = len(y)
            tensors.append(torch.zeros(n, 1, 1))

        tensors.append(torch.FloatTensor(y))
        dataset = TensorDataset(*tensors)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def _unpack_batch(self, batch):
        eeg = batch[0].to(self.device)
        gaze = batch[1].to(self.device)
        y = batch[2].to(self.device)

        if self.model.mode == "eeg_only":
            gaze = None
        elif self.model.mode == "gaze_only":
            eeg = None

        return eeg, gaze, y


def f1_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute F1 score without sklearn dependency."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    return float(2 * precision * recall / (precision + recall + 1e-12))
