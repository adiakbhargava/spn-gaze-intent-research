# SPN + Gaze Intent Research

**Research validation pipeline for EEG+eye-tracking fusion intent decoding, based on Stimulus-Preceding Negativity (SPN).**

This repository contains the offline training, ablation study, and evaluation code that validates the central thesis: multimodal fusion of EEG and gaze signals improves intent detection over either modality alone, addressing the Midas touch problem in gaze-based interfaces.

Based on [Reddy et al. (CHI 2024)](https://dl.acm.org/doi/10.1145/3613904.3641925): *"Towards an Eye-Brain-Computer Interface: Combining Gaze with the Stimulus-Preceding Negativity for Target Selections in XR."*

> **Companion repository:** [intent-gateway](https://github.com/adiakbhargava/intent-gateway) — the production Rust implementation of real-time EEG+gaze fusion.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    OFFLINE TRAINING PIPELINE                      │
│                                                                   │
│  Dataset ──▶ Feature Extraction ──▶ Fusion Classifier ──▶ JSON   │
│  (synthetic,   (EEG band power +     (LR / SVM / RF /    results │
│   EEGET-ALS,    gaze kinematics +      1D-CNN / LSTM)            │
│   EEGET-RSOD)   SPN amplitude)                                   │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                    REAL-TIME REPLAY DEMO (optional)               │
│                                                                   │
│  Synthetic LSL ──▶ Feature Pipeline ──▶ Trained Model ──▶ UI     │
│  Streams (replay)                        (live inference)         │
│                    Latency Dashboard ◀─── Timing Tracker          │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                    LATENCY BENCHMARK                              │
│                                                                   │
│  BCI2000 .dat ──▶ Roundtrip/SourceTime ──▶ Comparison Report     │
│  Parser             Extraction               (vs. fusion demo)   │
└──────────────────────────────────────────────────────────────────┘
```

## Modules

| Module | Purpose | Key Files |
|--------|---------|-----------|
| **Data Loading** | EEGEyeNet (synthetic), EEGET-RSOD/ALS (real), BCI2000 .dat parser | `src/data/` |
| **Feature Extraction** | EEG band power (Welch/AR/FFT), SPN amplitude, ERD/ERS, gaze fixation/saccade/pupil | `src/features/` |
| **Artifact Rejection** | Trial-level amplitude, gradient, flat-channel, and gaze-loss filtering | `src/features/artifact_rejection.py` |
| **Classification** | Baseline models (LR, SVM, RF), neural models (1D-CNN, BiLSTM), ablation framework | `src/models/` |
| **Streaming** | LSL replay, synchronized EEG+gaze streams, real-time inference engine | `src/streaming/` |
| **Benchmark** | BCI2000 latency analysis, fusion pipeline latency benchmarking | `src/benchmark/` |
| **Demo UI** | Flask+WebSocket dashboard with target grid, gaze cursor, live classification *(optional)* | `src/ui/` |

## Dataset Expectations

| Dataset | Subjects | EEG | Eye Tracking | Labels | Use |
|---------|----------|-----|--------------|--------|-----|
| **Synthetic** | Configurable (default 20) | 32ch simulated @ 128 Hz | Simulated x,y,pupil | Binary (intent/observe) | Ablation study, development |
| **[EEGET-ALS](https://springernature.figshare.com/articles/dataset/EEGET-ALS_Dataset/24485689)** | 26 (6 ALS + 20 healthy) | 32ch @ 128 Hz (Emotiv) | 30 Hz Tobii | Behavioral ground truth (dwell selection) | **Primary validation** |
| **[EEGET-RSOD](https://figshare.com/articles/dataset/EEGET-RSOD/26943565)** | 38 | 32ch @ 500 Hz (Enobio) | 250 Hz SMI RED250 | Heuristic only (at chance) | Needs Tier 2 labeling |
| **BCI2000 .dat** | Any | Any paradigm | Not required | N/A | Latency benchmarking only |

The synthetic data generator models the Midas touch problem with calibrated difficulty: ~60-70% EEG-only, ~65-75% gaze-only, ~75-85% fused accuracy, matching published BCI literature.

**Real dataset setup:** Download EEGET-ALS and/or EEGET-RSOD from the links above. Override paths with `--als-dir` / `--rsod-dir`, or configure in `configs/default.yaml`.

## Quick Start

### Install

```bash
python -m venv venv
source venv/bin/activate      # Linux/macOS
# or: venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### Run Ablation Study

```bash
# Synthetic data (default: 20 subjects, 80 trials, 5-fold CV)
python scripts/train.py --synthetic

# Include neural models (requires PyTorch)
python scripts/train.py --synthetic --neural --fast-neural

# Real data — EEGET-ALS (leave-one-subject-out CV)
python scripts/train.py --dataset als

# Real data — EEGET-RSOD
python scripts/train.py --dataset rsod

# Both datasets combined
python scripts/train.py --dataset combined

# With artifact rejection (recommended for real data)
python scripts/train.py --dataset als --reject-artifacts --amplitude-threshold 200.0

# Quick sanity check
python scripts/train.py --dataset als --max-subjects 3
```

Results are saved to `output/{dataset}_full_ablation.json`.

### Run Latency Benchmark

```bash
python scripts/benchmark.py --bci2000-dir data/raw/bci2000
```

### Run Real-Time Demo (Optional)

Requires Flask and Flask-SocketIO (`pip install flask flask-socketio`).

```bash
python scripts/run_demo.py
# Opens dashboard at http://127.0.0.1:5000

# Without dashboard (headless inference)
python scripts/run_demo.py --no-dashboard
```

### Cloud Training (Modal)

```bash
pip install modal && modal setup
modal run scripts/modal_train.py              # CPU baseline
modal run scripts/modal_train.py --neural     # GPU neural models
```

### Run Tests

```bash
pytest tests/ -v
```

## Test Results

**81 passed, 2 skipped** (Python 3.12, 60s)

| Suite | Tests | Status |
|-------|-------|--------|
| `test_loader` | 9 | All pass |
| `test_features` | 32 | All pass |
| `test_classifier` | 9 | All pass |
| `test_bci2000` | 6 | All pass |
| `test_integration` | 25 | 23 pass, 2 skipped |

Skipped tests:
- `test_replay_with_real_data` — requires downloaded dataset files (not included in repo)
- `test_dashboard_creation` — requires Flask (optional dependency)

## Ablation Results

Results are validated at three tiers, each serving a distinct purpose:

| Tier | Dataset | Who Can Run | Purpose |
|------|---------|-------------|---------|
| 1 | Synthetic (fixed seed) | Anyone — no download | Reproducible baseline; confirms pipeline code is correct |
| 2 | EEGET-ALS (real, behavioral labels) | Anyone with ~2 GB download | Confirms real neural signal extraction |
| 3 | EEGET-RSOD (real, heuristic labels) | Anyone with ~4 GB download | Documents labeling limitation honestly |

Tier 1 is the **reproducible artifact**: `python scripts/train.py --synthetic` produces numbers that match this README exactly on any machine (fixed seed, no external data). Tiers 2–3 are **author-validated**; dataset files are not bundled due to size.

### Why the Synthetic Data Is Credible

Synthetic EEG+gaze data is credible when it is (a) deterministic, (b) calibrated against real experiments, and (c) faithfully models the difficulty structure of the problem being studied. This generator satisfies all three criteria.

**1. Deterministic reproduction.**
The generator uses a fixed NumPy random seed (`seed=42`). Every run on every machine produces bit-for-bit identical data, making all synthetic results exactly reproducible without downloading any dataset.

**2. Calibrated to published BCI difficulty ranges.**
Signal-to-noise ratios and class-conditional distributions are tuned so classification difficulty matches ranges reported across EEG intent-decoding and BCI literature:

| Modality | Target range (literature) | Achieved |
|----------|--------------------------|---------|
| EEG-only | 60–70% accuracy | 69–71% |
| Gaze-only | 65–75% accuracy | 68–73% |
| Fused | 75–85% accuracy | 75–77% |

These ranges are intentionally conservative — real paradigms vary widely, but these numbers reflect the signal quality achievable with offline epoch-based decoding under realistic SNR.

**3. Faithful Midas touch modeling.**
The Midas touch problem is that both intent and observe conditions involve looking at the target — the eye naturally moves to objects of interest regardless of whether you want to select them. The generator encodes this correctly:
- Both conditions produce saccades toward and fixations on the target
- Intent trials carry a simulated pre-stimulus EEG signature (SPN-like negativity at occipitoparietal channels, −500ms to 0ms) and marginally longer fixation durations
- Observe trials produce the same gaze trajectory but weaker or absent EEG modulation
- This means neither modality alone fully separates the classes — gaze is ambiguous by construction, and EEG is noisy — so fusion provides the reliable combined signal

**4. Per-subject generalization confirmed by CV variance.**
The 5-fold per-subject cross-validation produces meaningful variance estimates. Across all three classifiers in the fused condition (from `output/ablation_results_synthetic.json`):

| Model | Accuracy (mean ± std) | AUC-ROC (mean ± std) |
|-------|----------------------|----------------------|
| Logistic Regression | 0.748 ± 0.072 | 0.820 ± 0.086 |
| SVM (RBF) | 0.742 ± 0.073 | 0.819 ± 0.083 |
| Random Forest | 0.761 ± 0.091 | 0.835 ± 0.094 |

The ±0.07–0.09 standard deviation reflects realistic individual-subject variability: simulated subjects span a range of EEG SNR levels, not a single easy-to-classify distribution. Fold-to-fold stability at these variance levels is consistent with published within-subject BCI classification benchmarks.

**5. Cross-validated against real neural signals.**
On EEGET-ALS (Tier 2 below), the identical feature extraction pipeline achieves EEG-only AUC = 0.795 on real recordings — well above chance — using the same Welch band-power + SPN features as the synthetic evaluation. This confirms the features extract genuine neural information, not synthetic artifacts. The synthetic results are an honest preview of what the pipeline can achieve; the real-data results confirm the pipeline is not overfit to synthetic structure.

### Tier 1 — Synthetic (20 subjects × 80 trials, 5-fold CV)

*Fully reproducible: `python scripts/train.py --synthetic`*

| Model | Modality | Accuracy | F1 | AUC-ROC |
|-------|----------|----------|----|---------|
| Logistic Regression | EEG-only | 0.701 | 0.702 | 0.771 |
| Logistic Regression | Gaze-only | 0.684 | 0.671 | 0.759 |
| Logistic Regression | **Fused** | **0.752** | **0.753** | **0.826** |
| SVM (RBF) | EEG-only | 0.694 | 0.695 | 0.766 |
| SVM (RBF) | Gaze-only | 0.687 | 0.659 | 0.751 |
| SVM (RBF) | **Fused** | **0.756** | **0.759** | **0.831** |
| Random Forest | EEG-only | 0.704 | 0.699 | 0.763 |
| Random Forest | Gaze-only | 0.726 | 0.713 | 0.797 |
| Random Forest | **Fused** | **0.767** | **0.766** | **0.844** |

**Fusion improves accuracy by 4–7pp over the best unimodal baseline and AUC-ROC by 5–8pp across all classifiers.**

### Tier 2 — EEGET-ALS: Author-Validated (Real Data)

*20 healthy subjects, spelling BCI, Emotiv 32ch @ 128 Hz, Tobii gaze @ 30 Hz, leave-one-subject-out CV.*
*To reproduce: download from [Figshare](https://springernature.figshare.com/articles/dataset/EEGET-ALS_Dataset/24485689) (~2 GB), then `python scripts/train.py --dataset als`.*

| Modality | Balanced Accuracy | AUC-ROC | Observe Recall | Intent Recall |
|----------|------------------|---------|----------------|---------------|
| EEG-only | 0.722 | 0.795 | 0.707 | 0.738 |
| Gaze-only | **0.935** | **0.973** | 0.915 | 0.955 |
| Fused | 0.933 | 0.972 | 0.908 | 0.959 |

Key findings:
- **EEG alone achieves AUC = 0.795** — well above chance, confirming the feature pipeline extracts genuine neural signals from real recordings
- **Gaze dominates on this paradigm** — dwell-based spelling BCI selection creates highly discriminable fixation durations that directly encode intent, leaving less room for EEG to contribute
- **Fusion value is paradigm-dependent** — multimodal fusion provides its largest gain when intent and observe produce similar gaze trajectories (the Midas touch problem proper); in spelling BCIs with explicit dwell selection, gaze already carries the majority of the discriminative signal

### Tier 3 — EEGET-RSOD: Negative Result (Author-Validated)

*38 subjects, remote sensing imagery, Enobio 32ch @ 500 Hz, SMI RED250 gaze @ 250 Hz, leave-one-subject-out CV.*
*To reproduce: download from [Figshare](https://figshare.com/articles/dataset/EEGET-RSOD/26943565) (~4 GB), then `python scripts/train.py --dataset rsod`.*

| Modality | Balanced Accuracy | AUC-ROC |
|----------|------------------|---------|
| EEG-only | 0.500 | 0.447 |
| Gaze-only | 0.497 | 0.487 |
| Fused | 0.500 | 0.456 |

This is an **honest negative result, not a pipeline failure.** All modalities produce chance-level performance because the Tier 1 heuristic labels (target-present/absent derived from image position, ~90/10 class imbalance) do not encode real intent/observe neural-state differences. The confusion matrices confirm it: EEG-only and Fused predict the majority class for every trial (observe recall = 0.0); Gaze-only achieves only 1.6% observe recall. All classifiers learn the class imbalance, not the intended signal.

Meaningful results on RSOD require Tier 2 gaze-informed labeling: deriving ground-truth intent labels from fixation density maps and dwell-time thresholds, rather than image-position heuristics.

## EEG Features

- **Band Power:** Theta (4-7 Hz), Alpha (8-12 Hz), Mu (8-13 Hz), Beta (13-30 Hz) via Welch, AR, or FFT
- **SPN Amplitude:** Stimulus-Preceding Negativity from occipitoparietal channels (-500ms to 0ms pre-stimulus) — the anticipatory signal that confirms intent
- **ERD/ERS:** Event-Related Desynchronization/Synchronization in Mu/Beta bands — motor preparation signatures

## Gaze Features

- **Fixation Metrics:** Duration, count, spatial dispersion (I-VT algorithm, velocity threshold)
- **Saccade Kinematics:** Amplitude, peak velocity, direction (skipped for gaze rates < 60 Hz)
- **Pupillometry:** Mean diameter, dilation rate, range (cognitive load proxy)
- **Spatial Dispersion:** RMS, BCEA (Bivariate Contour Ellipse Area)
- **AOI Analysis:** Dwell time, first entry latency, re-entry count

## Project Structure

```
spn-gaze-intent-research/
├── configs/
│   └── default.yaml                # Pipeline configuration (rates, channels, paths)
├── scripts/
│   ├── train.py                    # Training & ablation study entry point
│   ├── download_data.py            # Synthetic data generation
│   ├── run_demo.py                 # Real-time replay demo (optional dashboard)
│   ├── benchmark.py                # BCI2000 latency benchmarking
│   ├── modal_train.py              # Modal cloud training
│   ├── export_to_onnx.py           # Export trained model to ONNX format
│   ├── preprocess_eegeyenet.py     # EEGEyeNet NPZ/BIDS preprocessing
│   └── preprocess_real_data.py     # Position task preprocessing
├── docs/
│   ├── real-data-integration-plan.md
│   ├── robustness-feedback-validation.md
│   ├── website-article-intent-stream-pipeline.md
│   └── website-demo-plan-intent-stream-pipeline.md
├── src/
│   ├── data/
│   │   ├── eegeyenet_loader.py     # EEGEyeNet dataset loader (synthetic + real)
│   │   ├── eeget_loader.py         # EEGET-RSOD & EEGET-ALS loaders
│   │   └── bci2000_parser.py       # BCI2000 .dat binary parser
│   ├── features/
│   │   ├── eeg_features.py         # EEG spectral features + SPN + ERD/ERS
│   │   ├── gaze_features.py        # Eye-tracking feature extraction
│   │   ├── feature_pipeline.py     # Combined EEG+gaze feature orchestration
│   │   └── artifact_rejection.py   # Trial-level artifact filtering
│   ├── models/
│   │   ├── baseline.py             # LR, SVM, RF classifiers + ablation
│   │   └── neural.py               # 1D-CNN, BiLSTM fusion models
│   ├── streaming/
│   │   ├── lsl_replay.py           # Synthetic LSL stream replay
│   │   └── realtime_inference.py   # Real-time classification engine
│   ├── benchmark/
│   │   └── latency.py              # Latency benchmarking & comparison
│   └── ui/
│       └── dashboard.py            # Flask+WebSocket demo dashboard (optional)
├── tests/                          # 83 tests (81 pass, 2 skip)
├── output/                         # Ablation result JSONs
├── requirements.txt                # Core dependencies
├── requirements-modal.txt          # Modal cloud training dependencies
└── pyproject.toml                  # Project metadata & pytest config
```

## Known Limitations

### Synthetic vs. Real Data Gap
The synthetic data generator calibrates difficulty against BCI literature ranges, but does not replicate the non-stationarity, electrode drift, or cross-session variability of real EEG. The EEGET-ALS validation on real data confirms the pipeline works on genuine neural signals, but the synthetic ablation numbers should not be interpreted as expected production accuracy.

### Artifact Rejection State
Artifact rejection (`--reject-artifacts`) is implemented with amplitude, gradient, flat-channel, and gaze-loss checks. However, rejection thresholds are currently static per-dataset defaults. Adaptive thresholds (e.g., percentile-based per-subject) would improve robustness across recording conditions.

### Event-Locking Constraints
SPN extraction requires knowing when the stimulus is about to appear (the -500ms to 0ms pre-stimulus window). This makes the pipeline **epoch-based by design**. It cannot validate SPN on continuous streaming data without an external event marker that defines stimulus onset. This is the primary gap between this research pipeline and the production gateway, which processes continuous streams.

### Dashboard Optionality
The demo dashboard (`src/ui/`) is optional visualization tooling. It has zero coupling to training, evaluation, or metrics code. All ablation results and benchmarks are produced without it. It requires Flask and Flask-SocketIO, which are not in `requirements.txt` by default.

### EEGET-RSOD Labeling
The RSOD dataset produces chance-level results because Tier 1 heuristic labels (image-position-based) do not capture real intent/observe conditions. Meaningful results require Tier 2 gaze-informed labeling using fixation patterns to derive ground-truth labels.

## Roadmap

### Real-Data Integration
- Build an epoch segmenter for the production gateway that produces event-locked `.npz` windows compatible with this pipeline's loaders
- Add a `gateway_loader.py` to `src/data/` that reads gateway-exported epochs with resampling and channel mapping
- Validate SPN detection on real streaming data processed through the full production stack

### Adaptive Normalization
- Per-subject z-score normalization of EEG features (currently global)
- Session-adaptive baselines for ERD/ERS calculation
- Online artifact threshold estimation

### Spatial Filtering
- Common Spatial Patterns (CSP) for EEG channel combination
- Laplacian montage as alternative to CAR
- Region-of-interest optimization for SPN electrodes

### Robustness Hardening
- Cross-session transfer learning evaluation
- Electrode dropout simulation and graceful degradation
- Subject-independent model evaluation with domain adaptation

## References

1. Reddy et al. "Towards an Eye-Brain-Computer Interface: Combining Gaze with the Stimulus-Preceding Negativity for Target Selections in XR." ACM CHI 2024.
2. Kastrati et al. "EEGEyeNet: a Simultaneous Electroencephalography and Eye-tracking Dataset and Benchmark." 2021.
3. Schalk et al. "BCI2000: A General-Purpose Brain-Computer Interface (BCI) System." IEEE TBME, 2004.
4. Pan et al. "EEGET: An EEG and Eye Tracking Synchronized Dataset for ALS Patients." Scientific Data, 2024.
5. Wang et al. "EEGET-RSOD: An EEG and Eye Tracking Synchronized Dataset for Remote Sensing Object Detection." Scientific Data, 2024.
6. Remsik et al. "Behavioral outcomes following brain-computer interface intervention for upper extremity rehabilitation in stroke." 2018.
7. Remsik et al. "Ipsilesional Mu Rhythm Desynchronization and Changes in Motor Behavior Following BCI Intervention for Stroke Rehabilitation." 2019.

## License

MIT
