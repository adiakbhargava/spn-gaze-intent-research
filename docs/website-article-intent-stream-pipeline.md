# Intent Stream Pipeline: Building a Real-Time EEG + Eye-Tracking Intent Decoder

*How I built a multimodal brain-computer interface research stack, why it matters for the “Midas touch” problem, and where it goes next.*

---

## 1) Why this project exists

Eye tracking is fast and intuitive, but it has a known failure mode: **the Midas touch problem**. In gaze-driven interfaces, people look at many things they do **not** intend to select. If a system triggers selection from gaze alone, it confuses visual attention with user intent.

This project asks a practical question:

> Can we combine gaze behavior with non-invasive neural signals to distinguish “I looked at it” from “I want to click it” in real time?

The core inspiration comes from Reddy et al. (CHI 2024), which explores an eye-brain-computer interface that uses **Stimulus-Preceding Negativity (SPN)** as an intent-related neural signature before selection events.

Reference paper: [Towards an Eye-Brain-Computer Interface: Combining Gaze with the Stimulus-Preceding Negativity for Target Selections in XR](https://dl.acm.org/doi/pdf/10.1145/3613904.3641925)

---

## 2) The high-level solution

Intent Stream Pipeline is a **multimodal fusion system**:

- **Input A: EEG** (brain activity)
- **Input B: Eye tracking** (where and how the user looks)
- **Output:** A real-time prediction of likely intent vs passive observation

The system has two complementary tracks:

1. **Python research pipeline** for data loading, feature extraction, model training, ablation, and replay-based real-time inference.
2. **Rust real-time engine** that demonstrates low-latency, systems-oriented pipeline execution and telemetry under performance constraints.

This split is intentional: Python for research velocity, Rust for production realism.

---

## 3) Architecture at a glance

### Python stack (research + experimentation)

- Data loaders for synthetic and real datasets
- EEG feature extraction (band power, SPN, ERD/ERS)
- Gaze feature extraction (fixations, saccades, pupil, dispersion)
- Baseline and neural models (e.g., LR/SVM/RF, 1D-CNN/BiLSTM)
- Real-time replay engine with latency tracking and dashboard

### Rust stack (performance + reliability path)

- Lock-free SPSC ring buffer
- Zero-copy decode path
- Rolling statistics + FFT-based features
- Fault injection and telemetry reporting
- Throughput/latency tooling for real-time constraints

Together, these components answer two different questions:

- **Research question:** does fusion improve intent decoding?
- **Engineering question:** can this architecture sustain real-time operation with predictable latency?

---

## 4) What the pipeline actually does

## 4.1 Data and synchronization

The pipeline supports synthetic and real EEG/eye-tracking datasets, plus a BCI2000 parser for latency-focused analysis. The streaming path assumes synchronized windows across modalities and can be replayed to evaluate online behavior.

## 4.2 EEG features

EEG processing currently includes:

- Spectral band-power estimates (Welch / AR / FFT variants)
- SPN amplitude features in pre-stimulus windows
- ERD/ERS-style contrast features in sensorimotor bands

The key idea is to capture both ongoing rhythms (frequency-domain) and anticipatory state changes (time-domain SPN-like behavior).

## 4.3 Gaze features

Gaze processing includes:

- Fixation statistics (count, duration, dispersion)
- Saccade kinematics (amplitude, velocity)
- Pupil-derived statistics
- Spatial dispersion measures (e.g., BCEA-like summaries)

These features model visual behavior and attention dynamics around candidate targets.

## 4.4 Fusion and classification

The system supports multiple modeling regimes:

- Traditional baselines for interpretability and benchmarking
- Neural temporal models for sequence dynamics
- Ablation studies to compare EEG-only vs gaze-only vs fused performance

This allows the project to test the central thesis directly:

> Does multimodal fusion outperform unimodal decoding in intent-relevant scenarios?

## 4.5 Real-time path

A replay-driven inference engine uses sliding windows to extract features and run predictions continuously while tracking feature/model/end-to-end latency.

---

## 5) Why the Rust port matters

Many BCI prototypes stop at offline notebooks. I wanted a second proof:

- Low-overhead packet movement
- Bounded behavior under load
- Deterministic-ish latency characteristics
- Explicit failure-mode testing (drops/corruption/backpressure)

The Rust implementation demonstrates real-time systems thinking: memory discipline, lock-free queues, and telemetry-first profiling.

In other words, the Rust path is not “just a rewrite”—it is part of validating feasibility for interactive systems where latency and jitter directly affect user experience.

---

## 6) Scientific context and inspiration

The project is motivated by eye-brain fusion research, especially the SPN framing in the CHI paper above. Rather than claiming direct replication, the goal here is:

1. Adopt the same *problem framing* (intent confirmation beyond gaze),
2. Build a robust engineering scaffold around it,
3. Evaluate where today’s pipeline is strong and where it still needs hardening for real-world non-invasive EEG.

---

## 7) Current strengths

- End-to-end multimodal architecture from data to real-time inference
- Multiple model families + ablation support
- Practical benchmarking and latency instrumentation
- Clear path from research code to systems implementation (Rust)
- Transparent documentation of known gaps and prioritized improvements

---

## 8) Current limitations (and what I’m improving)

Non-invasive EEG is noisy and non-stationary. The hard part is not writing a classifier; it is controlling confounds.

### Highest-priority hardening items

1. **Artifact robustness upgrades**
   - Move beyond threshold-only rejection toward stronger EOG-aware correction paths
2. **Adaptive normalization**
   - Session-aware or rolling normalization to reduce drift over time
3. **Event-locking clarity**
   - Tighten event alignment contracts between gaze fixation events and EEG epoching in real-time
4. **Abstention policy**
   - Add an uncertainty rejection zone instead of forcing every decision into a binary label
5. **Spatial denoising enhancements**
   - Expand preprocessing options (e.g., Laplacian/xDAWN-style additions where appropriate)

These are exactly the changes that move a promising prototype toward robust human-in-the-loop behavior.

---

## 9) Who this is for

This project is designed to be legible to two audiences:

- **Newcomers** interested in how EEG + gaze can improve interaction reliability
- **Experts** who care about signal quality, confounds, event timing, and deployment realism

If you are new: think of this as an intent “double-check layer” for gaze interfaces.

If you are experienced: think of it as an engineering-forward fusion scaffold with explicit acknowledgment of single-trial EEG challenges and a concrete hardening roadmap.

---

## 10) Roadmap snapshot

### Near term

- Implement confidence abstention
- Strengthen artifact handling defaults
- Add adaptive feature normalization hooks
- Publish clearer event-locking/dataset protocol notes

### Medium term

- Integrate stronger spatial filtering options
- Add calibration-aware denoising workflows
- Expand real-headset validation scenarios

### Longer term

- Online adaptation/recalibration lifecycle
- More robust domain shift handling across subjects and sessions
- Production-focused evaluation under realistic UI task constraints

---

## 11) Final thought

The main thesis is simple:

> Gaze tells you *where* a person is looking; neural context helps tell you *whether they mean to act*.

Intent Stream Pipeline is my attempt to turn that thesis into an end-to-end, testable system—first as a research platform, and progressively as a real-time engineering artifact.

If you’d like, I also published a companion **demo walkthrough** that explains exactly how to showcase the system live, what to measure, and how to narrate results to technical and non-technical audiences.
