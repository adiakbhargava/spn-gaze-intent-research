#!/usr/bin/env python3
"""
Export trained neural models to ONNX format.

Loads a trained Conv1DFusion or LSTMFusion model from a .pt state_dict
file and exports it as an ONNX model with named inputs ("eeg", "gaze")
and output ("logit") for consumption by co-gateway's Rust inference engine.

Usage:
    # Export Conv1D (default)
    python scripts/export_to_onnx.py

    # Export with explicit paths
    python scripts/export_to_onnx.py \\
        --model-path models/saved/conv1d_fused.pt \\
        --output models/saved/conv1d_fused.onnx

    # Export LSTM model
    python scripts/export_to_onnx.py \\
        --model-type lstm \\
        --model-path models/saved/lstm_fused.pt \\
        --output models/saved/lstm_fused.onnx

    # Verify exported model matches PyTorch output
    python scripts/export_to_onnx.py --verify
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def export_conv1d(
    model_path: Path,
    output_path: Path,
    n_eeg_channels: int = 128,
    n_gaze_channels: int = 3,
    n_samples: int = 500,
) -> None:
    import torch
    from src.models.neural import Conv1DFusion

    model = Conv1DFusion(
        n_eeg_channels=n_eeg_channels,
        n_gaze_channels=n_gaze_channels,
        n_samples=n_samples,
        mode="fused",
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    dummy_eeg = torch.randn(1, n_eeg_channels, n_samples)
    dummy_gaze = torch.randn(1, n_gaze_channels, n_samples)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (dummy_eeg, dummy_gaze),
        str(output_path),
        input_names=["eeg", "gaze"],
        output_names=["logit"],
        dynamic_axes={
            "eeg": {0: "batch"},
            "gaze": {0: "batch"},
            "logit": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,  # Use legacy TorchScript exporter (stable on Windows)
    )
    print(f"Exported Conv1D (fused) to {output_path}")
    print(f"  Input: eeg ({n_eeg_channels}, {n_samples}), gaze ({n_gaze_channels}, {n_samples})")
    print(f"  Output: logit (1,) -> sigmoid -> confidence [0, 1]")


def export_lstm(
    model_path: Path,
    output_path: Path,
    n_eeg_channels: int = 128,
    n_gaze_channels: int = 3,
    n_samples: int = 500,
    hidden_size: int = 32,
    n_layers: int = 1,
    bidirectional: bool = False,
) -> None:
    import torch
    from src.models.neural import LSTMFusion

    model = LSTMFusion(
        n_eeg_channels=n_eeg_channels,
        n_gaze_channels=n_gaze_channels,
        hidden_size=hidden_size,
        n_layers=n_layers,
        mode="fused",
        bidirectional=bidirectional,
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    dummy_eeg = torch.randn(1, n_eeg_channels, n_samples)
    dummy_gaze = torch.randn(1, n_gaze_channels, n_samples)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (dummy_eeg, dummy_gaze),
        str(output_path),
        input_names=["eeg", "gaze"],
        output_names=["logit"],
        dynamic_axes={
            "eeg": {0: "batch"},
            "gaze": {0: "batch"},
            "logit": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,  # Use legacy TorchScript exporter (stable on Windows)
    )
    print(f"Exported LSTM (fused) to {output_path}")
    print(f"  Input: eeg ({n_eeg_channels}, {n_samples}), gaze ({n_gaze_channels}, {n_samples})")
    print(f"  Output: logit (1,) -> sigmoid -> confidence [0, 1]")


def verify_onnx(
    pt_model_path: Path,
    onnx_path: Path,
    model_type: str = "conv1d",
    n_eeg_channels: int = 128,
    n_gaze_channels: int = 3,
    n_samples: int = 500,
    n_tests: int = 50,
) -> None:
    import torch
    import onnxruntime as ort

    if model_type == "conv1d":
        from src.models.neural import Conv1DFusion
        model = Conv1DFusion(
            n_eeg_channels=n_eeg_channels,
            n_gaze_channels=n_gaze_channels,
            n_samples=n_samples,
            mode="fused",
        )
    else:
        from src.models.neural import LSTMFusion
        model = LSTMFusion(
            n_eeg_channels=n_eeg_channels,
            n_gaze_channels=n_gaze_channels,
            hidden_size=32,
            n_layers=1,
            mode="fused",
            bidirectional=False,
        )

    model.load_state_dict(torch.load(pt_model_path, map_location="cpu", weights_only=True))
    model.eval()

    session = ort.InferenceSession(str(onnx_path))
    max_diff = 0.0

    for i in range(n_tests):
        eeg = np.random.randn(1, n_eeg_channels, n_samples).astype(np.float32)
        gaze = np.random.randn(1, n_gaze_channels, n_samples).astype(np.float32)

        with torch.no_grad():
            pt_out = model(torch.from_numpy(eeg), torch.from_numpy(gaze)).numpy()

        onnx_out = session.run(None, {"eeg": eeg, "gaze": gaze})[0]

        diff = np.max(np.abs(pt_out - onnx_out))
        max_diff = max(max_diff, diff)

        np.testing.assert_allclose(pt_out, onnx_out, rtol=1e-4, atol=1e-5)

    print(f"Verification passed: {n_tests} tests, max diff = {max_diff:.2e}")


def main():
    parser = argparse.ArgumentParser(description="Export neural models to ONNX")
    parser.add_argument(
        "--model-type", choices=["conv1d", "lstm"], default="conv1d",
        help="Model architecture to export (default: conv1d)",
    )
    parser.add_argument(
        "--model-path", type=Path, default=None,
        help="Path to .pt state_dict (default: models/saved/{model_type}_fused.pt)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output .onnx path (default: models/saved/{model_type}_fused.onnx)",
    )
    parser.add_argument(
        "--n-eeg-channels", type=int, default=128,
        help="Number of EEG channels (default: 128, matching synthetic data)",
    )
    parser.add_argument(
        "--n-gaze-channels", type=int, default=3,
        help="Number of gaze channels (default: 3: x, y, pupil)",
    )
    parser.add_argument(
        "--n-samples", type=int, default=500,
        help="Number of time samples per window (default: 500)",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="After export, verify ONNX output matches PyTorch output",
    )
    args = parser.parse_args()

    # Resolve defaults
    if args.model_path is None:
        args.model_path = Path(f"models/saved/{args.model_type}_fused.pt")
    if args.output is None:
        args.output = Path(f"models/saved/{args.model_type}_fused.onnx")

    if not args.model_path.exists():
        print(f"Error: {args.model_path} not found.")
        print(f"Run training first: python scripts/train.py --synthetic --neural --fast-neural")
        sys.exit(1)

    # Export
    if args.model_type == "conv1d":
        export_conv1d(
            args.model_path, args.output,
            args.n_eeg_channels, args.n_gaze_channels, args.n_samples,
        )
    else:
        export_lstm(
            args.model_path, args.output,
            args.n_eeg_channels, args.n_gaze_channels, args.n_samples,
        )

    # Verify
    if args.verify:
        print("\nVerifying ONNX output matches PyTorch...")
        verify_onnx(
            args.model_path, args.output, args.model_type,
            args.n_eeg_channels, args.n_gaze_channels, args.n_samples,
        )


if __name__ == "__main__":
    main()
