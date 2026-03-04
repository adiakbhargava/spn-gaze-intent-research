# Dataset Integration Plan for intent-stream-pipeline

## Document Purpose

This plan describes how to integrate two real EEG+eye-tracking datasets into the intent-stream-pipeline, which currently runs on synthetic data. It is written for a Claude Code instance to execute. The plan covers dataset selection, condition mapping, trial segmentation, code changes, feature pipeline adaptations, evaluation methodology, and success criteria.

---

## 1. Dataset Selection and Subject Counts

### 1.1 Primary Dataset: EEGET-RSOD (Remote Sensing Object Detection)

- **Location:** `C:\AAKB\0-Projects\co-gateway\data\eeget-rsod\`
- **Subjects:** 38 (P01 through P38)
- **EEG:** 32 channels (NE Enobio 32) at 500 Hz, stored as EDF files in `EEG/P{XX}.edf`
- **Eye-tracking:** SMI RED250 at 250 Hz, stored as comma-delimited text in `ET/P{XX}.txt` (BeGaze export format)
- **Merged data:** Pre-aligned EDF+ET files in `merged/P{XX}/`
- **Task:** Subjects view 500 satellite images (3 seconds each, 0.5s gray fixation cross between images), searching for airplanes. 90% of images contain 1-6 airplanes (target-present), 10% contain no airplanes (target-absent). Images are organized in 5 blocks of 100 images each.
- **Use all 38 subjects.** This dataset is the best fit for the pipeline because the task naturally produces intent (active visual search and fixation on detected targets) vs observe (scanning images without target objects) conditions.

**EEG channel names (32 channels):** P7, P4, Cz, Pz, P3, P8, O1, O2, T8, F8, C4, F4, Fp2, Fz, C3, F3, Fp1, T7, F7, Oz, PO4, FC6, FC2, AF4, CP6, CP2, CP1, CP5, FC1, FC5, AF3, PO3

**ET columns (key fields from BeGaze export):** Time, Type (MSG/SMP), Trial, L POR X [px], L POR Y [px], R POR X [px], R POR Y [px], L Pupil Diameter [mm], R Pupil Diameter [mm], L Event Info, R Event Info, Stimulus

### 1.2 Secondary Dataset: EEGET-ALS (Spelling BCI)

- **Location:** `C:\AAKB\0-Projects\co-gateway\data\eeget-als\EEGET-ALS Dataset\`
- **Subjects:** 26 total (6 ALS patients: ALS01-ALS06, 20 healthy: id1-id20)
- **EEG:** 32 channels (Emotiv EPOC Flex) at 128 Hz, stored as EDF in `{subject}/time1/scenario{N}/EEG.edf`
- **Eye-tracking:** Tobii at ~30 Hz, stored as CSV in `{subject}/time1/scenario{N}/ET.csv`
- **Metadata:** `eeg.json` (sampling rate, channel count), `scenario.json` (task description), `info.json` (age, gender)
- **Task:** 9 scenarios per subject. Scenarios 1-7 are motor imagery tasks; scenarios 8-9 are functional demand tasks. Each includes a spelling sub-task where the subject uses gaze to type Vietnamese text on a virtual keyboard.
- **Use 20 healthy subjects (id1 through id20) initially.** The ALS patients have variable data quality and very few subjects (n=6) -- insufficient for reliable cross-subject statistics. The healthy cohort provides a cleaner signal for establishing baselines. ALS patients can be added later for a clinical-population analysis.
- **Only use the spelling sub-task portions (Task iii)** from all 9 scenarios per subject, as these contain the intent (selecting a letter by dwelling on it) vs observe (scanning the keyboard for the next letter) conditions.

### 1.3 Excluded Dataset: EEGEyeNet (OpenNeuro ds005872)

- Only 1 subject (sub-EP10), 129 channels at 500 Hz. This is a gaze-prediction regression benchmark, not an intent-classification paradigm. Excluded because: (a) n=1 provides zero cross-subject generalization, (b) no intent/observe condition labels, (c) the task is a saccade/fixation prediction regression problem.

---

## 2. Condition Mapping to Intent/Observe Binary Classification

### 2.1 EEGET-RSOD Condition Mapping

The RSOD task maps to intent/observe as follows:

| Condition | Label | Definition | Rationale |
|-----------|-------|------------|-----------|
| **Target detected** | 1 (intent) | Satellite image contains airplane(s) AND the subject fixates within the target region | Active target engagement -- the subject has identified and is attending to a specific object with recognition-driven gaze behavior |
| **No target / target not fixated** | 0 (observe) | Image contains no airplane, OR image contains airplane but subject's gaze never dwells on the target region | Passive scanning -- the subject is in visual search mode without target engagement |

**Implementation approach (two-tier, from simple to precise):**

**Tier 1 (image-level labeling, implement first):**
- Label all target-absent images (10% of images, ~50 per subject) as condition=0 (observe)
- Label all target-present images (90% of images, ~450 per subject) as condition=1 (intent)
- This is imprecise because a subject may not have detected the target in a target-present image, but it provides a strong starting point with 500 trials per subject and is trivially implementable from the event markers
- To identify target-absent images: the EEGET-RSOD paper states each block has 90 target-present and 10 target-absent images. If no ground-truth label file is available in the dataset, use a heuristic: target-absent images can potentially be identified by the absence of fixation-on-target gaze patterns (shorter mean fixation duration, higher gaze dispersion). Alternatively, check the Figshare repository for a supplementary label CSV, or contact dataset authors. For now, proceed with Tier 1 using the 90/10 split.

**Tier 2 (gaze-informed labeling, implement second):**
- For target-present images, further sub-classify based on gaze behavior during the 3-second viewing window:
  - If the subject shows a sustained fixation (>300ms) in the central region of the image (suggesting they found the target): condition=1 (intent)
  - If the subject's gaze is diffuse/scanning throughout the 3s window: condition=0 (observe, target missed)
- This creates a more ecologically valid Midas-touch problem: subject is looking at images in both conditions, but only actively engaging with targets in the intent condition

**Critical note on class balance:** The 90/10 target-present/absent ratio creates a class imbalance problem. Address this by:
- Oversampling the minority class (observe) using SMOTE or random oversampling
- Using stratified cross-validation
- Reporting F1 and AUC-ROC (not just accuracy) as primary metrics
- With Tier 2 labeling, the balance will improve since many target-present trials will be re-classified as observe (subject missed the target)

### 2.2 EEGET-ALS Condition Mapping

The ALS spelling task maps to intent/observe as follows:

| Condition | Label | Definition | Rationale |
|-----------|-------|------------|-----------|
| **Letter selection (dwell)** | 1 (intent) | The 1-2 second window ending at each character-typing event in the ET CSV | The subject has decided to select a specific letter and is dwelling on it to trigger selection |
| **Scanning/searching** | 0 (observe) | Gaze samples between selections where no character is being typed | The subject is scanning the virtual keyboard to find the next letter |

**Implementation approach:**
- Parse the ET.csv `character typing` column. When a new character appears (e.g., 'n', 'a', etc.), the time of that event marks a "selection" moment.
- Extract a 1.5-second window ending at each selection event as an intent trial (condition=1).
- Extract 1.5-second windows from periods where no selection is occurring (at least 0.5s before the next selection) as observe trials (condition=0).
- Match the EEG timestamps using the EEGTimeStamp.txt file for temporal alignment between EEG and ET streams.
- Expected yield per scenario: approximately 10-20 letter selections (typing a Vietnamese sentence), so roughly 10-20 intent trials per scenario, with 9 scenarios per subject giving ~90-180 intent trials per subject.

---

## 3. Trial Segmentation

### 3.1 EEGET-RSOD Trial Segmentation

**Step 1: Parse event markers from the ET file.**
- Read `ET/P{XX}.txt`, skipping lines starting with `##` (header comments).
- Extract MSG lines to identify image onsets. Each satellite image onset is a MSG event containing a `.jpg` filename (not `gary.jpg`).
- Each image is shown for exactly 3000ms, followed by a 500ms gray fixation cross (`gary.jpg`).

**Step 2: Compute trial boundaries.**
- For each satellite image MSG event at timestamp `t_img`:
  - EEG epoch: from `t_img` to `t_img + 3000ms` (500 Hz, so 1500 EEG samples per trial)
  - ET epoch: from `t_img` to `t_img + 3000ms` (250 Hz, so 750 ET samples per trial)
  - Set `stimulus_onset_sample = 0` (the image onset is the beginning of the trial)

**Step 3: Align EEG and ET time bases.**
- The RSOD dataset has pre-aligned merged files in `merged/P{XX}/` containing both EEG and ET data. Use these if they provide synchronized timestamps. If not:
  - The EEG EDF timestamps start from 0 (in EDF+ annotation format, each data record has a `+{seconds}` annotation).
  - The ET timestamps are in microseconds from some system clock.
  - Synchronization approach: identify shared events (e.g., the first image onset) in both streams and compute the time offset between them. Then use this offset to align all subsequent events.

**Step 4: Extract trial data.**
- EEG: read 32 EEG channels (exclude X, Y, Z accelerometer and EDF Annotations channels) for the 3-second window. Shape: (32, 1500).
- ET: read gaze position (L POR X, L POR Y or averaged binocular) and pupil diameter (L Pupil Diameter) for the 3-second window. Shape: (750, 3) -- [gaze_x, gaze_y, pupil_diameter].
- Resample ET from 250Hz to 500Hz (linear interpolation) to match EEG sample rate, producing shape (1500, 3). Alternatively, keep them at different rates and adjust `gaze_sfreq` in the `EEGEyeNetTrial` object.

**Expected trial counts per subject:** ~500 trials (100 images per block x 5 blocks). Total across 38 subjects: ~19,000 trials.

### 3.2 EEGET-ALS Trial Segmentation

**Step 1: Parse ET.csv for selection events.**
- Read `ET.csv` with UTF-8 encoding. The columns are: TimeStamp, Data, x, y, character typing, sentence.
- A "selection event" occurs when the `character typing` column transitions from empty to a new accumulated string (each new character appended to the string indicates a selection).
- Detect selection moments by finding rows where the accumulated `character typing` string grows by one character compared to the previous non-empty value.

**Step 2: Align EEG and ET.**
- Read `EEGTimeStamp.txt` which contains per-sample timestamps for the EEG stream.
- The ET.csv TimeStamp column contains timestamps in seconds (with fractional precision).
- Map ET timestamps to EEG sample indices using the EEGTimeStamp.txt lookup.

**Step 3: Extract trial epochs.**
- **Intent trials (condition=1):** For each selection event at time `t_select`, extract a 1.5-second window centered on or preceding the selection: EEG samples from `t_select - 1.5s` to `t_select`, shape (32, 192) at 128 Hz. ET samples from the same window, shape (45, 3) at 30 Hz. Use x, y from the Data/x/y columns and pupil data if available (check if Tobii provides pupil -- if not, fill with zeros).
- **Observe trials (condition=0):** For periods between selections where gaze is actively scanning (moving), extract 1.5-second windows. Ensure at least 0.5s separation from any selection event to avoid contamination.
- Downsample or handle the rate difference: EEG at 128 Hz, ET at 30 Hz. Store the actual rates in `eeg_sfreq` and `gaze_sfreq` fields.

**Expected trial counts per subject:** ~90-180 intent trials (10-20 selections x 9 scenarios), matched with an equal number of observe trials. Total across 20 healthy subjects: ~3,600-7,200 trials.

**Critical note on ET quality:** The Tobii eye tracker in this dataset operates at only 30 Hz, which is extremely low for saccade detection. The gaze feature pipeline's `velocity_threshold` of 30 deg/s may not work reliably at 30 Hz because saccades can complete within a single sample interval (33ms). The pipeline must either (a) lower the velocity threshold, (b) use only fixation/pupil features and skip saccade detection, or (c) interpolate gaze to a higher rate.

---

## 4. Data Loading Code Changes

### 4.1 New Loader Module: `src/data/eeget_loader.py`

Create a new loader module (do NOT modify `eegeyenet_loader.py`, which handles the original EEGEyeNet format and synthetic data). The new loader should:

1. **Reuse `EEGEyeNetTrial` and `EEGEyeNetDataset` dataclasses** -- these are generic enough to hold any EEG+gaze trial. The `paradigm` field can be set to `"rsod"` or `"als_spelling"`.

2. **Implement `load_rsod_dataset(data_dir, subjects=None, max_subjects=None, use_merged=True) -> EEGEyeNetDataset`:**
   - Read EDF files using `mne-python` (already a project dependency).
   - Parse ET text files using manual CSV parsing (comma-separated SMI BeGaze format).
   - Segment trials based on image onset MSG events.
   - Assign condition labels (Tier 1: image-level target-present/absent).
   - Set `channel_names` to the 32-channel list from the EDF header.
   - Set `eeg_sfreq=500.0`, `gaze_sfreq=250.0` (or 500.0 if resampled).

3. **Implement `load_als_spelling_dataset(data_dir, subjects=None, max_subjects=None, scenarios=None) -> EEGEyeNetDataset`:**
   - Walk the directory tree: `{subject}/time1/scenario{N}/`.
   - Read `eeg.json` for sampling rate and channel count.
   - Read EDF files for EEG data, parse ET.csv for gaze and selection events.
   - Segment trials around selection events (intent) and scanning periods (observe).
   - Set `eeg_sfreq=128.0`, `gaze_sfreq=30.0`.

### 4.2 Modifications to `scripts/train.py`

Add a new `--dataset` argument to select which dataset to use:

```
--dataset {synthetic,rsod,als,combined}
--rsod-dir PATH  (default: data/eeget-rsod)
--als-dir PATH   (default: data/eeget-als/EEGET-ALS Dataset)
```

Update `load_dataset()` to dispatch to the appropriate loader based on the `--dataset` argument.

### 4.3 Modifications to `configs/default.yaml`

Add new data source configurations:

```yaml
data:
  rsod:
    root_dir: "C:/AAKB/0-Projects/co-gateway/data/eeget-rsod"
    sampling_rate: 500
    n_channels: 32
    trial_duration_ms: 3000
    et_sampling_rate: 250
  als:
    root_dir: "C:/AAKB/0-Projects/co-gateway/data/eeget-als/EEGET-ALS Dataset"
    sampling_rate: 128
    n_channels: 32
    trial_duration_ms: 1500
    et_sampling_rate: 30
    use_healthy_only: true
```

---

## 5. Feature Pipeline Changes

### 5.1 Channel Group Resolution

The current `CHANNEL_GROUP_NAMES` in `feature_pipeline.py` are already compatible with the 32-channel montages in both datasets. The RSOD EEG channels include: O1, O2, Oz, Pz, P3, P4, PO3, PO4 (occipitoparietal), Cz, C3, C4 (central), Fz, FCz-equivalent (frontal_midline), P7, P8 (parietal). The `resolve_channel_groups()` function will correctly resolve these by name.

No changes needed to channel group definitions -- the existing groups will resolve a subset of channels from each group, and groups with no matches will be omitted with a warning.

### 5.2 Handling Different EEG Sampling Rates

The current pipeline assumes 500 Hz throughout. The ALS dataset is at 128 Hz.

**Changes needed:**
- The `eeg_sfreq` field on each `EEGEyeNetTrial` already stores the sampling rate and is passed to all feature extraction functions. Verify that all feature functions use this field rather than hardcoding 500 Hz.
- **Welch PSD estimation:** The `nperseg` default of `sfreq // 2` will give 64 samples for 128 Hz data, which is only 0.5 seconds. This may be too short for reliable theta/alpha estimation. Consider using `nperseg = min(int(sfreq), n_samples)` (already the current logic) and verify it produces stable spectra.
- **AR model order:** The default order of 16 may need to be reduced for 128 Hz data (rule of thumb: order = 2 * sfreq / lowest_frequency_of_interest). For 128 Hz with 4 Hz theta: order = 2 * 128/4 = 64, which is actually fine. But for short epochs (192 samples), order 16 is appropriate.
- **SPN window:** The SPN is measured in a 500ms pre-stimulus window. For 128 Hz data at 1.5s epoch length, this means samples 0-64 of a 192-sample epoch (if stimulus is at sample 96). This is valid.

### 5.3 Handling Different Gaze Sampling Rates

The current pipeline assumes `gaze_sfreq = 500.0`. The RSOD ET is at 250 Hz, the ALS ET is at 30 Hz.

**Changes needed:**
- The `gaze_sfreq` field on `EEGEyeNetTrial` already stores the gaze sampling rate.
- Verify that `extract_gaze_features()` uses the `sfreq` parameter (it does -- line 336 of `gaze_features.py`).
- **Saccade detection at 30 Hz (ALS):** At 30 Hz, each sample spans 33ms. A typical saccade lasts 20-80ms, meaning saccades may span only 1-2 samples. The velocity-based I-VT algorithm will produce unreliable saccade detections. **Solution:** For ALS data at 30 Hz, skip saccade features (set them to 0) or only use fixation/pupil features and skip saccade detection. Add a check in `extract_gaze_features()`: if `sfreq < 60`, skip saccade detection and return zeros for saccade features.
- **Fixation detection at 30 Hz (ALS):** Minimum fixation duration of 100ms maps to only 3 samples at 30 Hz. This is marginal but usable. Reduce `min_fixation_ms` to 133ms (4 samples) to ensure at least 4 samples are used for dispersion estimation.

### 5.4 Gaze Data Format Differences

The current pipeline expects `gaze_data` in shape (n_samples, 3) with columns [x_degrees, y_degrees, pupil_diameter].

**RSOD gaze data:** The ET file provides gaze in pixels (L POR X [px], L POR Y [px]) and pupil diameter in mm. Convert pixel positions to degrees visual angle using the geometry specified in the ET header: stimulus 344x194mm at 700mm distance, displayed on 1920x1080 pixel screen. Conversion: `deg = arctan(px * (344/1920) / 700) * 180/pi`.

**ALS gaze data:** The ET.csv provides normalized coordinates (0-1 range, representing screen proportions). Convert to degrees visual angle using approximate monitor geometry (assume standard BCI setup: ~50cm viewing distance, 24-inch monitor at 1920x1080). If exact geometry is unavailable, use the normalized coordinates directly -- the relative gaze dispersion features will still be informative.

### 5.5 Handling Missing Pupil Data

The ALS dataset's Tobii ET.csv may not include pupil diameter (the CSV columns are: TimeStamp, Data, x, y, character typing, sentence). If pupil data is unavailable, fill the third column of `gaze_data` with zeros. The pupil features will all be zero, and the feature pipeline already handles this gracefully.

### 5.6 EEG Preprocessing

The current pipeline does not apply EEG preprocessing (filtering, re-referencing) before feature extraction. For real data, add minimal preprocessing:

1. **Bandpass filter:** 1-45 Hz (as specified in `default.yaml`). Use `scipy.signal.butter` and `scipy.signal.filtfilt` for zero-phase filtering.
2. **Notch filter:** 50 Hz for both datasets (both were collected in Vietnam/China, which use 50 Hz mains power, NOT 60 Hz as in the current config).
3. **Common average reference:** Re-reference all channels to the mean of all channels.
4. **Baseline correction:** Subtract the mean of each channel within the trial.

Add this as a `preprocess_trial()` function in the new loader module, applied to each trial before it is added to the dataset.

---

## 6. Expected Outcomes and Success Criteria

### 6.1 Accuracy Targets

Based on published literature for EEG+gaze intent detection:

| Metric | RSOD (Tier 1) | RSOD (Tier 2) | ALS Spelling | Combined |
|--------|---------------|---------------|--------------|----------|
| EEG-only accuracy | 55-65% | 60-70% | 55-65% | 55-65% |
| Gaze-only accuracy | 60-70% | 65-75% | 70-80% | 65-75% |
| Fused accuracy | 65-75% | 70-80% | 75-85% | 70-80% |
| Fusion improvement | >5% over best unimodal | >5% | >5% | >5% |

**Notes:**
- RSOD Tier 1 accuracy may be artificially high because image-level labels create a trivially separable problem for gaze features (target-absent images have very different gaze patterns -- diffuse scanning vs. target fixation).
- RSOD Tier 2 will be harder and more meaningful because both conditions involve viewing images with targets.
- ALS spelling may have higher gaze-only accuracy because the dwell-to-select mechanism produces very distinctive gaze patterns during selection.
- These numbers are rough estimates. The real test is whether fusion > best unimodal.

### 6.2 Primary Success Criterion

**Fusion provides statistically significant improvement over the best unimodal approach.** Specifically:
- Paired t-test (or Wilcoxon signed-rank test) across subjects comparing fused accuracy vs best-unimodal accuracy should yield p < 0.05.
- The improvement should be at least 3 percentage points in absolute accuracy.

### 6.3 Secondary Success Criteria

1. Cross-subject generalization: leave-one-subject-out (LOSO) accuracy should be above chance (>55%) for all three modalities.
2. The pipeline runs end-to-end on real data without errors.
3. Artifact rejection removes a reasonable fraction of trials (5-30%), not 0% or >50%.
4. Feature distributions from real data should look qualitatively similar to synthetic data (same order of magnitude, no all-zero features).

### 6.4 What Would Indicate Failure

- Fused accuracy is no better than (or worse than) the best unimodal approach -- would indicate the fusion strategy needs redesign.
- All accuracies near chance (50%) -- would indicate the condition labels are not capturing a real neural/behavioral distinction.
- Artifact rejection removes >50% of trials -- would indicate the preprocessing or threshold settings need adjustment.
- Features are dominated by NaN/Inf values -- would indicate the sampling rate or data format handling has bugs.

---

## 7. Statistical Requirements

### 7.1 Cross-Validation Strategy

Use **leave-one-subject-out (LOSO) cross-validation** as the primary evaluation method:
- For RSOD (38 subjects): 38-fold LOSO. Train on 37 subjects, test on 1. This provides 38 per-subject accuracy values.
- For ALS (20 subjects): 20-fold LOSO. Train on 19 subjects, test on 1. This provides 20 per-subject accuracy values.
- For combined analysis: can also use 5-fold group-stratified CV (ensuring no subject appears in both train and test) for faster iteration during development.

**LOSO is essential** because the goal is to demonstrate that the model generalizes to new subjects (a requirement for any practical BCI). Within-subject CV would overestimate performance.

### 7.2 Subject Counts for Statistical Tests

**RSOD (n=38):** 38 subjects is sufficient for:
- Paired t-test (power > 0.80 to detect a 5% accuracy difference with SD=10%, alpha=0.05 requires n >= 34).
- Wilcoxon signed-rank test (nonparametric alternative, slightly less power, but n=38 is still adequate).

**ALS healthy (n=20):** 20 subjects provides:
- Paired t-test power of 0.68 to detect a 5% difference with SD=10%. This is marginal. A 7% difference would be detectable with power > 0.80.
- If results are borderline, use the Wilcoxon signed-rank test and report confidence intervals.

**Combined (n=58):** Using both datasets together provides excellent statistical power. However, combining requires careful handling of dataset effects (different hardware, sampling rates, tasks). Use dataset as a covariate in the analysis or stratify by dataset.

### 7.3 Statistical Tests to Report

For each ablation comparison (EEG-only vs. Fused, Gaze-only vs. Fused), report:

1. **Per-subject mean and standard deviation** of accuracy, F1, and AUC-ROC.
2. **Paired t-test** (or Wilcoxon signed-rank if normality is violated per Shapiro-Wilk test) comparing per-subject accuracy values.
3. **Effect size:** Cohen's d for the accuracy difference.
4. **95% confidence interval** for the mean accuracy difference.

### 7.4 Multiple Comparisons Correction

With 3 pairwise comparisons (EEG vs Fused, Gaze vs Fused, EEG vs Gaze), apply Bonferroni correction: use alpha = 0.05/3 = 0.0167 as the significance threshold. Alternatively, report both corrected and uncorrected p-values.

---

## 8. Implementation Order

The executing Claude Code instance should implement these changes in the following order:

1. **Install dependencies:** Ensure `mne-python` is in the project's dependency list (it should already be there).

2. **Create `src/data/eeget_loader.py`** with the RSOD and ALS loading functions. Include inline tests (assert statements or a `if __name__ == "__main__"` block) that load 1 subject from each dataset and print trial counts, shapes, and condition distributions.

3. **Modify `src/features/gaze_features.py`** to handle low-frequency gaze data (30 Hz). Add a check: if `sfreq < 60`, set saccade features to 0 and skip saccade detection.

4. **Update `configs/default.yaml`** with RSOD and ALS data source configurations. Change notch filter from 60 Hz to 50 Hz for these datasets.

5. **Modify `scripts/train.py`** to add the `--dataset` argument and dispatch to the new loader.

6. **Run a quick sanity test:** Load 3 subjects from RSOD, extract features in all 3 modes, train a Random Forest, and verify that accuracy is above chance. This should take under 5 minutes.

7. **Run the full LOSO evaluation** on RSOD (38 subjects) with the ablation study. Save results to `output/rsod_ablation_results.json`.

8. **Run the full LOSO evaluation** on ALS healthy (20 subjects) with the ablation study. Save results to `output/als_ablation_results.json`.

9. **Combine results** and compute statistical tests. Save a summary table to `output/real_data_summary.json`.

---

## 9. Key Technical Details for Implementation

### 9.1 Reading EDF Files

```python
import mne

def read_edf_eeg(edf_path, exclude_non_eeg=True):
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    if exclude_non_eeg:
        # Drop non-EEG channels (accelerometers, annotations)
        non_eeg = [ch for ch in raw.ch_names if ch in ['X', 'Y', 'Z', 'EDF Annotations']]
        raw.drop_channels(non_eeg)
    return raw.get_data(), raw.info['sfreq'], raw.ch_names
```

### 9.2 Parsing RSOD ET Files

```python
def parse_rsod_et(et_path):
    """Returns list of (timestamp_us, event_type, payload) tuples."""
    events = []
    with open(et_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.startswith('##'):
                continue
            parts = line.strip().split(',')
            if len(parts) < 3:
                continue
            timestamp = int(parts[0])
            event_type = parts[1].strip()
            if event_type == 'MSG':
                payload = parts[3] if len(parts) > 3 else ''
                events.append((timestamp, 'MSG', payload))
            elif event_type == 'SMP':
                events.append((timestamp, 'SMP', parts))
    return events
```

### 9.3 Gaze Coordinate Conversion (RSOD)

The ET header specifies: Stimulus Dimension 344x194mm, Head Distance 700mm, Screen 1920x1080 pixels.

```python
import numpy as np

SCREEN_W_PX = 1920
SCREEN_H_PX = 1080
STIM_W_MM = 344
STIM_H_MM = 194
HEAD_DIST_MM = 700

def px_to_deg(px_x, px_y):
    mm_x = (px_x / SCREEN_W_PX) * STIM_W_MM - STIM_W_MM / 2
    mm_y = (px_y / SCREEN_H_PX) * STIM_H_MM - STIM_H_MM / 2
    deg_x = np.degrees(np.arctan(mm_x / HEAD_DIST_MM))
    deg_y = np.degrees(np.arctan(mm_y / HEAD_DIST_MM))
    return deg_x, deg_y
```

### 9.4 EEG-ET Synchronization (RSOD)

The merged files in `merged/P{XX}/` contain both EEG (P{XX}.edf) and ET (P{XX}.txt) with presumably aligned timestamps. Use these instead of separately aligning the raw files.

If using the separate files:
- EEG timestamps: seconds from recording start (from EDF annotation channel).
- ET timestamps: microseconds from system clock.
- Find the first image onset in both streams and compute offset: `offset = et_first_image_us / 1e6 - eeg_first_image_s`.
- Apply this offset to convert all ET timestamps to EEG time: `eeg_time = (et_us / 1e6) - offset`.

---

## 10. File Listing Summary

**Files to create:**
- `src/data/eeget_loader.py` -- new loader for RSOD and ALS datasets

**Files to modify:**
- `scripts/train.py` -- add `--dataset` argument
- `configs/default.yaml` -- add RSOD/ALS configurations
- `src/features/gaze_features.py` -- handle low-frequency gaze data
- `requirements.txt` -- ensure `mne` and `pyedflib` are listed

**Files NOT to modify:**
- `src/data/eegeyenet_loader.py` -- keep for synthetic data and original EEGEyeNet format
- `src/features/eeg_features.py` -- no changes needed; it already parameterizes on `sfreq`
- `src/features/feature_pipeline.py` -- no changes needed; channel groups will resolve correctly

---

## 11. Data Paths Quick Reference

The datasets live in the co-gateway project (already downloaded):

```
C:\AAKB\0-Projects\co-gateway\data\eeget-rsod\
  EEG\P01.edf ... P38.edf     (32ch EEG @ 500 Hz)
  ET\P01.txt ... P38.txt       (250 Hz SMI RED250 eye tracking)
  merged\P01\ ... P38\         (paired EDF+TXT per subject)

C:\AAKB\0-Projects\co-gateway\data\eeget-als\EEGET-ALS Dataset\
  id1\ ... id20\               (healthy subjects)
    time1\scenario1\           (first recording session)
      EEG.edf                  (32ch EEG @ 128 Hz)
      ET.csv                   (30 Hz Tobii eye tracking)
      EEGTimeStamp.txt         (per-sample EEG timestamps)
      eeg.json, scenario.json  (metadata)
  ALS01\ ... ALS06\            (ALS patients, same structure but with 10 time sessions)
```

---

This plan provides all the information needed for a Claude Code instance to implement the real-data integration end-to-end. The key risks are: (1) EEG-ET synchronization in the RSOD dataset may require debugging if the merged files have alignment issues, and (2) the ALS dataset's 30 Hz eye-tracking rate severely limits gaze feature quality. Both are addressed with fallback strategies in the plan.
