# Demo Plan: Intent Stream Pipeline in Action

*A detailed, publishable walkthrough you can attach to your website as companion media (video/script/live demo guide).* 

---

## Demo goals

By the end of this demo, viewers should understand:

1. The **problem** (Midas touch in gaze interfaces)
2. Why **EEG + gaze fusion** is a meaningful approach
3. How your system runs from streams to predictions in real time
4. What your **current limitations** are and how your roadmap addresses them

---

## Audience modes

Use two layers of narration:

- **Layer A (beginner-friendly):** focus on intuition, avoid jargon-heavy claims
- **Layer B (expert add-ons):** include timing windows, feature families, confound controls, and drift limitations

Tip: In the video, use “Beginner View” overlays and optional “Technical Deep Dive” side panels.

---

## Suggested demo length

- **8-12 minutes** for website default video
- Optional **20-30 minute extended technical cut** for researchers

---

## Assets to prepare

1. **Architecture slide** (Python research path + Rust real-time path)
2. **Feature slide** (EEG, gaze, fusion)
3. **Live dashboard capture** (cursor, prediction/confidence, latency)
4. **Ablation chart** (EEG-only vs gaze-only vs fused)
5. **Roadmap slide** (quick wins + medium + long-term)
6. **Paper inspiration slide** with citation:
   - Reddy et al. CHI 2024 (use your provided link)

---

## End-to-end demo script (website-ready)

## Segment 1 — Hook (0:00-0:45)

**Visual:** one slide with “Why eye tracking alone fails” + Midas touch example.

**Talk track:**

- “When you use gaze as a pointer, you look at many things you don’t intend to select.”
- “This project adds a neural confirmation signal so gaze can become more intentional.”
- “The system is inspired by recent eye-brain-computer interface research using SPN-related signals.”

---

## Segment 2 — Problem framing (0:45-2:00)

**Visual:** timeline graphic showing fixation vs true selection intent.

**Beginner explanation:**

- Gaze = attention, not always action
- Need a second signal to reduce false activations

**Expert add-on:**

- Mention anticipatory neural dynamics in pre-selection windows
- Mention confound risk from ocular artifacts and non-stationarity

---

## Segment 3 — Architecture overview (2:00-3:30)

**Visual:** your architecture diagram.

**Talk track:**

- “Python pipeline handles data loading, feature extraction, model training, and replay inference.”
- “Rust path stress-tests real-time systems behavior: lock-free buffers, decode path, telemetry, and fault handling.”
- “This gives both scientific iteration speed and systems-level confidence.”

---

## Segment 4 — Feature extraction walkthrough (3:30-5:00)

**Visual:** side-by-side EEG and gaze feature cards.

**Talk track:**

- EEG: band-power families, SPN-related summary, ERD/ERS-style contrasts
- Gaze: fixation/saccade/pupil/dispersion features
- Fusion: combines modalities and compares against unimodal baselines

**Expert add-on:**

- Explain why ablations matter for identifying where signal value truly comes from

---

## Segment 5 — Live real-time replay demo (5:00-7:30)

**Visual:** running dashboard + terminal with inference loop output.

**Steps:**

1. Start demo stream/replay
2. Show live predictions and confidence changes
3. Show latency panel (feature + model + total)
4. Point out thresholded decision behavior

**Narration cues:**

- “Here you can see the model processing synchronized windows continuously.”
- “Latency tracking shows whether this can support interactive UX.”
- “Current decision logic is binary thresholding; uncertainty abstention is next on roadmap.”

---

## Segment 6 — Results and interpretation (7:30-9:00)

**Visual:** ablation results plot(s).

**Talk track:**

- Explain relative performance of EEG-only / gaze-only / fused
- Highlight paradigm dependency (e.g., some tasks are gaze-dominant)
- Stress that robust validation depends on label quality and protocol

---

## Segment 7 — Honest limitations + roadmap (9:00-10:30)

**Visual:** roadmap slide in 3 columns.

### Quick wins

- Confidence abstention zone
- Stronger default artifact handling settings
- Adaptive normalization hooks
- Explicit event-locking protocol docs

### Medium

- Enhanced spatial filtering options
- Calibration-driven denoising

### Long-term

- Online adaptation/recalibration
- Better cross-session robustness

**Talk track:**

- “This is architecturally complete as a prototype, but signal processing hardening determines real-headset reliability.”
- “I’m actively implementing this roadmap and will publish updates.”

---

## Segment 8 — Close and call-to-action (10:30-11:00)

**Visual:** one page with repo, paper link, and contact.

**Talk track:**

- “If you work in BCIs, XR, or neuroadaptive interfaces, I’d love feedback and collaboration.”
- “I’m especially interested in applied work around robust intent confirmation in real-world conditions.”

---

## Suggested website package

Publish three items together:

1. **Primary article** (concept + architecture + roadmap)
2. **10-minute demo video** (above script)
3. **Technical appendix page** with:
   - data sources,
   - known limitations,
   - reproducibility commands,
   - and changelog for roadmap updates.

This gives visitors multiple ways to engage: skim, watch, or deep dive.

---

## Optional appendix: recording checklist

- Clean desktop + large fonts
- Deterministic script run order (no dead air)
- Pre-open all slides/plots
- Record one backup take with shorter wording
- Include subtitles for accessibility
- End with one concrete “what I’m shipping next” statement

---

## Optional appendix: outreach follow-up template

After publishing a roadmap improvement, send a concise follow-up:

> “I implemented [specific robustness improvement] from my published roadmap and updated the demo with before/after behavior. I’d value your critique on the next highest-impact step.”

This shows consistent execution and genuine interest in the field.
