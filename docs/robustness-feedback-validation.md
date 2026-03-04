# Robustness Feedback Validation (EEG+Gaze Intent Pipeline)

This note evaluates external feedback against the current codebase and estimates implementation effort before outreach.

## Verdict

The feedback is **mostly reputable and technically valid** for real-world non-invasive EEG deployment. The pipeline has strong architecture and useful baseline safeguards, but several signal-processing hardening steps are either absent or only partially implemented.

## Claim-by-Claim Validation

### 1) "SPN detection is fragile in real conditions"

**Assessment: Largely valid.**

- The pipeline computes SPN as a mean in a pre-stimulus window, but does **not** subtract a dedicated baseline window for SPN itself.
- SPN channels default to first N channels unless channel groups are passed, which can reduce physiological specificity in some contexts.
- A CAR step exists in the EEGET loader preprocessing path, but no Laplacian/xDAWN-like sharpening is integrated into the core feature extraction/realtime inference path.

**Code evidence:**
- `compute_spn_amplitude()` computes a pre-stimulus mean and returns it directly (no SPN baseline subtraction).  
- `extract_eeg_features()` includes SPN mean/min, but there is no denoising/spatial filter stage before SPN extraction.

### 2) "No artifact rejection is a serious problem"

**Assessment: Partially valid (overstated as absolute).**

- The project **does include** artifact rejection, but it is threshold-based (amplitude/gradient/flat/gaze-loss) and optional during dataset feature extraction.
- There is **no ICA/EOG regression** implementation in the online or offline default path.

**Code evidence:**
- `src/features/artifact_rejection.py` implements amplitude, gradient, flat-channel, and gaze-loss checks.
- `extract_dataset_features(..., reject_artifacts=False)` defaults to disabled rejection.
- No source module exposes ICA or regression-based EOG subtraction.

### 3) "Band features assume stationarity; no adaptive normalization"

**Assessment: Valid.**

- Welch/AR/FFT band-power features are extracted per window/trial.
- There is no rolling z-score or session-adaptive normalization in real-time inference.

**Code evidence:**
- `extract_eeg_features()` builds spectral features directly from current data.
- `RealtimeInferenceEngine._run_inference()` extracts current-window features and predicts, with no adaptive baseline or drift compensation.

### 4) "BiLSTM needs physiologically grounded, event-locked windowing"

**Assessment: Valid concern.**

- Real-time inference is sliding-window based, not explicitly event-locked to fixation onset.
- A buffer reset method exists for trial boundaries, but event-locking logic is not enforced in the real-time path.

**Code evidence:**
- `RealtimeInferenceEngine` uses `window_samples` and `step_samples` on a rolling buffer.
- `reset_buffer()` is available, but external orchestration must call it correctly.

### 5) "Needs abstention/rejection zone"

**Assessment: Valid and high impact.**

- Current output forces binary label by thresholding confidence (`prediction = 1 if conf >= threshold else 0`), with no explicit "uncertain" class.

## Practical Effort Estimate (Before Outreach)

## Quick wins (1-3 days)

1. **Enable artifact rejection by default for training scripts** with documented thresholds and rejection stats.
2. **Add confidence abstention** (e.g., emit `abstain` when confidence in [0.4, 0.6] or top-probability below threshold).
3. **Add per-session/rolling normalization** for band and SPN features (running mean/std z-scoring).
4. **Document event-locking contract** for real-time use (what event starts a window; when `reset_buffer()` must be called).

## Medium effort (3-10 days)

1. **CAR or surface Laplacian preprocessing** integrated into EEG feature extraction.
2. **Regression-based EOG subtraction** (lighter than ICA, likely first practical step).
3. **Calibration-based spatial filters** (xDAWN/CSP-like) for single-trial robustness.

## Higher effort (>1-2 weeks)

1. **Online ICA with quality checks** in real-time path.
2. **Full adaptive drift-handling + periodic recalibration workflow** with subject/session persistence.

## Outreach Recommendation

Given current state:

- **Do not block outreach** waiting for all neuroscience hardening steps.
- Send the email now, but frame the project as:
  - architecturally mature,
  - validated on synthetic + available real datasets,
  - and currently executing a prioritized hardening roadmap for real-headset robustness.

Suggested positioning sentence:

> "I have a working multimodal intent pipeline with real-time inference and ablations, and I am currently implementing the next robustness layer (artifact suppression, adaptive normalization, and event-locked inference contracts) to harden performance under real non-invasive EEG conditions."

This is usually stronger than claiming production robustness prematurely.
