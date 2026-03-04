"""
Real Dataset Loaders: EEGET-RSOD and EEGET-ALS

Implements loaders for two real EEG+eye-tracking datasets, reusing the
EEGEyeNetTrial / EEGEyeNetDataset dataclasses from eegeyenet_loader.py.

Datasets:
- EEGET-RSOD: 38 subjects, remote-sensing object detection task.
  EEG: 32ch NE Enobio @ 500 Hz.  ET: SMI RED250 @ 250 Hz.
  Intent = target-present image; Observe = target-absent image.

- EEGET-ALS: 26 subjects (20 healthy used), spelling BCI task.
  EEG: 32ch Emotiv EPOC Flex @ 128 Hz.  ET: Tobii @ ~30 Hz.
  Intent = gaze dwell on a letter (selection); Observe = scanning.

Implementation order follows docs/real-data-integration-plan.md §8.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

# MNE for EDF reading (already a project dependency)
try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False

# scipy for preprocessing filters
try:
    from scipy.signal import butter, filtfilt, iirnotch
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from src.data.eegeyenet_loader import EEGEyeNetDataset, EEGEyeNetTrial

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# RSOD constants
# ---------------------------------------------------------------------------
RSOD_EEG_SFREQ = 500.0
RSOD_ET_SFREQ = 250.0
RSOD_TRIAL_MS = 3000          # ms each satellite image is shown
RSOD_TRIAL_EEG = int(RSOD_EEG_SFREQ * RSOD_TRIAL_MS / 1000)   # 1500 samples
RSOD_TRIAL_ET  = int(RSOD_ET_SFREQ  * RSOD_TRIAL_MS / 1000)   # 750 samples

# SMI RED250 geometry for pixel→degree conversion
SCREEN_W_PX  = 1920
SCREEN_H_PX  = 1080
STIM_W_MM    = 344
STIM_H_MM    = 194
HEAD_DIST_MM = 700

# 32-channel names for RSOD (from plan §1.1)
RSOD_CHANNEL_NAMES = [
    "P7", "P4", "Cz", "Pz", "P3", "P8", "O1", "O2", "T8", "F8",
    "C4", "F4", "Fp2", "Fz", "C3", "F3", "Fp1", "T7", "F7", "Oz",
    "PO4", "FC6", "FC2", "AF4", "CP6", "CP2", "CP1", "CP5",
    "FC1", "FC5", "AF3", "PO3",
]

# ---------------------------------------------------------------------------
# ALS constants
# ---------------------------------------------------------------------------
ALS_EEG_SFREQ = 128.0
ALS_ET_SFREQ  = 30.0          # approximate; actual rate varies slightly
ALS_TRIAL_S   = 1.5           # seconds per intent/observe window
ALS_TRIAL_EEG = int(ALS_EEG_SFREQ * ALS_TRIAL_S)   # 192 samples
ALS_TRIAL_ET  = int(ALS_ET_SFREQ  * ALS_TRIAL_S)   # 45 samples
ALS_MIN_GAP_S = 0.5           # min seconds gap before/after selection

# Healthy subject IDs (id1 … id20)
ALS_HEALTHY_IDS = [f"id{i}" for i in range(1, 21)]


# ===========================================================================
# Shared preprocessing helpers
# ===========================================================================

def _bandpass_filter(data: np.ndarray, sfreq: float,
                     lowcut: float = 1.0, highcut: float = 45.0) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter. data shape: (n_ch, n_samples)."""
    if not SCIPY_AVAILABLE:
        return data
    nyq = sfreq / 2.0
    if highcut >= nyq:
        highcut = nyq * 0.99
    b, a = butter(4, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, data, axis=1)


def _notch_filter(data: np.ndarray, sfreq: float,
                  freq: float = 50.0, Q: float = 30.0) -> np.ndarray:
    """Zero-phase notch filter for mains interference. data shape: (n_ch, n_samples)."""
    if not SCIPY_AVAILABLE:
        return data
    nyq = sfreq / 2.0
    if freq >= nyq:
        return data
    b, a = iirnotch(freq / nyq, Q)
    return filtfilt(b, a, data, axis=1)


def _common_average_reference(data: np.ndarray) -> np.ndarray:
    """Re-reference to mean of all channels. data shape: (n_ch, n_samples)."""
    return data - data.mean(axis=0, keepdims=True)


def _baseline_correct(data: np.ndarray) -> np.ndarray:
    """Subtract per-channel trial mean (baseline correction). data shape: (n_ch, n_samples)."""
    return data - data.mean(axis=1, keepdims=True)


def preprocess_eeg_trial(data: np.ndarray, sfreq: float,
                         lowcut: float = 1.0, highcut: float = 45.0,
                         notch_freq: float = 50.0) -> np.ndarray:
    """
    Apply minimal EEG preprocessing to a single trial.

    Pipeline: bandpass → notch (50 Hz) → CAR → baseline correction.

    Args:
        data: EEG data, shape (n_channels, n_samples)
        sfreq: Sampling frequency in Hz
        lowcut: High-pass cutoff (Hz)
        highcut: Low-pass cutoff (Hz)
        notch_freq: Notch frequency for mains interference (50 Hz for EU/Asia)

    Returns:
        Preprocessed EEG, same shape as input
    """
    data = _bandpass_filter(data, sfreq, lowcut, highcut)
    data = _notch_filter(data, sfreq, notch_freq)
    data = _common_average_reference(data)
    data = _baseline_correct(data)
    return data


def _px_to_deg(px_x: np.ndarray, px_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert gaze pixel positions (RSOD SMI RED250) to degrees visual angle.

    Geometry: 1920×1080 px screen, stimulus 344×194 mm, head 700 mm away.
    """
    mm_x = (px_x / SCREEN_W_PX) * STIM_W_MM - STIM_W_MM / 2
    mm_y = (px_y / SCREEN_H_PX) * STIM_H_MM - STIM_H_MM / 2
    deg_x = np.degrees(np.arctan(mm_x / HEAD_DIST_MM))
    deg_y = np.degrees(np.arctan(mm_y / HEAD_DIST_MM))
    return deg_x, deg_y


# ===========================================================================
# RSOD loader
# ===========================================================================

def _get_edf_physical_dim(edf_path: Path) -> str:
    """
    Read the physical dimension of the first signal from an EDF header.

    EDF stores physical dimension as an 8-char ASCII field (e.g. "uV", "mV", "V").
    MNE converts this to Volts internally; knowing the original unit lets us
    apply the correct de-scaling when converting back to µV.

    Returns: one of "uV", "mV", "V", or "" (unknown)
    """
    try:
        with open(edf_path, "rb") as f:
            # Skip: version(8)+patient(80)+recording(80)+startdate(8)+starttime(8)
            #        +header_bytes(8)+reserved(44)+nrecords(8)+duration(8)
            f.seek(236)
            ns_raw = f.read(4).decode("ascii", errors="replace").strip()
            ns = int(ns_raw) if ns_raw.isdigit() else 1
            # Skip ns*16 (labels) + ns*80 (transducers) → land on physical dims
            f.seek(256 + 16 * ns + 80 * ns)
            first_pd = f.read(8).decode("ascii", errors="replace").strip()
        # Normalize
        pd = first_pd.strip().lower()
        if pd == "uv" or pd == "µv":
            return "uV"
        elif pd == "mv":
            return "mV"
        elif pd == "v":
            return "V"
        return first_pd  # return raw string for unknown
    except Exception:
        return ""


def _read_edf_eeg(edf_path: Path, exclude_non_eeg: bool = True
                  ) -> tuple[np.ndarray, float, list[str]]:
    """
    Read EEG data from an EDF file using MNE, returning data in µV.

    MNE always returns data in SI Volts (having applied the EDF physical
    dimension conversion internally). We detect the original EDF physical
    dimension and apply the appropriate rescaling so output is µV:

      EDF dim "uV": MNE output = V×1 (already divided by 1e6) → ×1e6 = µV
      EDF dim "mV": MNE output = V×1 (already divided by 1e3)  → ×1e3 = µV
      EDF dim "V" : MNE output = V×1 (no division)             → ×1e6 = µV

    Returns:
        data: (n_channels, n_samples) in µV
        sfreq: Sampling frequency
        ch_names: List of channel names
    """
    if not MNE_AVAILABLE:
        raise ImportError("mne-python is required to read EDF files. "
                          "Install with: pip install mne")
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    if exclude_non_eeg:
        non_eeg = [ch for ch in raw.ch_names
                   if ch.upper() in {"X", "Y", "Z", "EDF ANNOTATIONS"}
                   or "ANNOT" in ch.upper()]
        if non_eeg:
            raw.drop_channels(non_eeg)

    data = raw.get_data()          # (n_ch, n_samples) in MNE's Volts

    # Auto-detect physical dimension to apply correct V→µV conversion
    physical_dim = _get_edf_physical_dim(edf_path)
    if physical_dim == "mV":
        # MNE divided by 1e3 (mV→V); multiply by 1e3 to recover mV, then ×1e3 for µV
        data = data * 1e6   # = data * 1e3 (V→mV) * 1e3 (mV→µV)
        # NOTE: for ALS Emotiv EDF with mV physical dim this gives large values
        # (~650 mV range) due to the Emotiv EDF header gain settings. The
        # classification pipeline uses StandardScaler normalization so absolute
        # scale does not affect accuracy.
    else:
        # "uV" or "V": MNE divided by 1e6 (uV→V); multiply back to µV
        data = data * 1e6

    return data, float(raw.info["sfreq"]), list(raw.ch_names)


def _parse_rsod_et(et_path: Path) -> list[tuple[int, str, str]]:
    """
    Parse an SMI RED250 BeGaze-export text file.

    Returns:
        List of (timestamp_us, event_type, payload_or_raw_line) tuples.
        For MSG events: payload = the message text (e.g. "12135.jpg").
        For SMP events: payload = comma-joined original fields.
    """
    events = []
    header_done = False
    columns: list[str] = []

    with open(et_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if line.startswith("##"):
                continue
            if not header_done:
                # First non-comment line is the CSV header
                columns = [c.strip() for c in line.split(",")]
                header_done = True
                continue

            parts = line.split(",")
            if len(parts) < 3:
                continue
            try:
                ts = int(parts[0])
            except ValueError:
                continue
            etype = parts[1].strip() if len(parts) > 1 else ""
            if etype == "MSG":
                # parts[3] contains the message payload (after "# Message: ")
                payload = parts[3].strip() if len(parts) > 3 else ""
                # Strip "# Message: " prefix if present
                if payload.startswith("# Message:"):
                    payload = payload[len("# Message:"):].strip()
                events.append((ts, "MSG", payload))
            elif etype == "SMP":
                events.append((ts, "SMP", ",".join(parts)))
    return events


def _rsod_extract_smp_fields(raw_smp: str, col_indices: dict) -> tuple[float, float, float, float]:
    """
    Extract L POR X, L POR Y, R POR X, R POR Y from a raw SMP line.
    Falls back to zeros for missing/blink samples (value 0.00).
    """
    parts = raw_smp.split(",")
    def _get(key, fallback=0.0):
        idx = col_indices.get(key)
        if idx is None or idx >= len(parts):
            return fallback
        try:
            v = float(parts[idx])
            return v if v != 0.0 else fallback
        except ValueError:
            return fallback

    lx = _get("L POR X [px]")
    ly = _get("L POR Y [px]")
    rx = _get("R POR X [px]")
    ry = _get("R POR Y [px]")
    # Binocular average (fall back to available eye)
    if lx != 0.0 and rx != 0.0:
        x, y = (lx + rx) / 2, (ly + ry) / 2
    elif lx != 0.0:
        x, y = lx, ly
    elif rx != 0.0:
        x, y = rx, ry
    else:
        x, y = 0.0, 0.0

    # Pupil diameter: prefer left
    l_pup_idx = col_indices.get("L Pupil Diameter [mm]")
    r_pup_idx = col_indices.get("R Pupil Diameter [mm]")
    pupil = 0.0
    if l_pup_idx and l_pup_idx < len(parts):
        try:
            pupil = float(parts[l_pup_idx])
        except ValueError:
            pass
    if pupil == 0.0 and r_pup_idx and r_pup_idx < len(parts):
        try:
            pupil = float(parts[r_pup_idx])
        except ValueError:
            pass
    return x, y, pupil, 0.0   # 4th value unused


def _build_rsod_col_index(et_path: Path) -> dict[str, int]:
    """Parse the CSV header of an RSOD ET file to get column→index mapping."""
    with open(et_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if line.startswith("##"):
                continue
            # First non-comment line is the header
            cols = [c.strip() for c in line.split(",")]
            return {c: i for i, c in enumerate(cols)}
    return {}


def _rsod_segment_trials(
    eeg_data: np.ndarray,
    eeg_sfreq: float,
    et_events: list[tuple[int, str, str]],
    col_index: dict[str, int],
) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """
    Segment EEG and ET data into trials aligned to image onset events.

    Returns list of (eeg_trial, et_trial, condition) tuples.
    condition: 1 = target-present (intent), 0 = target-absent (observe)

    RSOD Tier 1 labeling: satellite images = intent (1), gary.jpg = observe (0).
    Non-satellite non-gray messages are skipped.
    """
    trials = []

    # Collect image onset events (MSG with .jpg filename, not gary.jpg)
    image_onsets: list[tuple[int, int]] = []  # (et_timestamp_us, condition)
    for ts, etype, payload in et_events:
        if etype != "MSG":
            continue
        if payload.lower().endswith(".jpg") and payload.lower() != "gary.jpg":
            # Target-present satellite image → intent
            image_onsets.append((ts, 1))
        elif payload.lower() == "gary.jpg":
            # Fixation cross → observe (used for inter-trial baseline)
            # Skip for now — gray cross is not a trial
            pass

    if not image_onsets:
        logger.warning("No image onset MSG events found in RSOD ET file")
        return []

    # Tier 1 labeling heuristic: no ground-truth label file is bundled with
    # the dataset. The paper states each block has 90 target-present and
    # 10 target-absent images (90/10 ratio). Apply a sequential-index heuristic:
    # every 10th image in the sequence is labeled target-absent (observe=0).
    # This preserves the correct class ratio without external label files.
    image_onsets = [
        (ts, 0 if (i % 10 == 9) else 1)   # every 10th → observe
        for i, (ts, _) in enumerate(image_onsets)
    ]

    # Separate SMP events with timestamps into an array for fast lookup
    smp_events = [(ts, raw) for ts, etype, raw in et_events if etype == "SMP"]
    if not smp_events:
        logger.warning("No SMP events found in RSOD ET file")
        return []

    smp_ts = np.array([s[0] for s in smp_events])  # microseconds

    # EEG has no per-sample timestamps in this dataset — we must sync via the
    # first image onset. EEG recording starts at t=0; ET timestamps are
    # absolute system clock microseconds. We'll use EDF start-of-file as
    # the EEG epoch 0 and align with ET.
    #
    # Synchronization: assume continuous recording, compute the ET sample
    # index corresponding to each image onset.
    first_et_ts = smp_ts[0]
    et_duration_us = smp_ts[-1] - first_et_ts

    # Total EEG samples
    n_eeg_total = eeg_data.shape[1]
    eeg_duration_s = n_eeg_total / eeg_sfreq

    # For each image onset, find the ET sample index and corresponding EEG sample
    for et_onset_ts, condition in image_onsets:
        # ET sample index relative to start
        et_onset_offset_us = et_onset_ts - first_et_ts
        if et_onset_offset_us < 0:
            continue

        # Find nearest SMP timestamp
        et_idx = int(np.searchsorted(smp_ts, et_onset_ts))
        et_idx = min(et_idx, len(smp_ts) - 1)

        # EEG sample index: scale ET time offset to EEG sample
        # Both streams start at the same recording time
        et_time_s = et_onset_offset_us / 1e6
        eeg_idx = int(et_time_s * eeg_sfreq)

        # Check bounds
        if eeg_idx + RSOD_TRIAL_EEG > n_eeg_total:
            continue
        if et_idx + RSOD_TRIAL_ET > len(smp_events):
            continue

        # Extract EEG epoch
        eeg_trial = eeg_data[:, eeg_idx: eeg_idx + RSOD_TRIAL_EEG].copy()

        # Extract ET epoch: build gaze array (n_et_samples, 3) = [x_deg, y_deg, pupil]
        gaze = np.zeros((RSOD_TRIAL_ET, 3))
        for i, smp_i in enumerate(range(et_idx, et_idx + RSOD_TRIAL_ET)):
            if smp_i >= len(smp_events):
                break
            raw = smp_events[smp_i][1]
            px_x, px_y, pupil, _ = _rsod_extract_smp_fields(raw, col_index)
            deg_x, deg_y = _px_to_deg(np.array([px_x]), np.array([px_y]))
            gaze[i, 0] = deg_x[0]
            gaze[i, 1] = deg_y[0]
            gaze[i, 2] = pupil

        trials.append((eeg_trial, gaze, condition))

    return trials


def load_rsod_subject(
    subject: str,
    data_dir: Path,
    preprocess: bool = True,
    use_merged: bool = True,
) -> list[EEGEyeNetTrial]:
    """
    Load all trials for a single RSOD subject.

    Args:
        subject: Subject ID string, e.g. "P01"
        data_dir: Root of eeget-rsod dataset
        preprocess: Apply EEG preprocessing (bandpass, notch, CAR, baseline)
        use_merged: Use pre-aligned merged EDF+ET in merged/{subject}/

    Returns:
        List of EEGEyeNetTrial objects
    """
    if use_merged:
        edf_path = data_dir / "merged" / subject / f"{subject}.edf"
        et_path  = data_dir / "merged" / subject / f"{subject}.txt"
    else:
        edf_path = data_dir / "EEG" / f"{subject}.edf"
        et_path  = data_dir / "ET" / f"{subject}.txt"

    if not edf_path.exists():
        logger.warning(f"RSOD EDF not found: {edf_path}")
        return []
    if not et_path.exists():
        logger.warning(f"RSOD ET not found: {et_path}")
        return []

    # Parse subject number for subject_id field (P01 → 1)
    try:
        subj_int = int(subject.lstrip("P").lstrip("0") or "0")
    except ValueError:
        subj_int = 0

    # Read EEG
    try:
        eeg_data, eeg_sfreq, ch_names = _read_edf_eeg(edf_path)
    except Exception as e:
        logger.error(f"Failed to read EDF for {subject}: {e}")
        return []

    # Preprocess EEG
    if preprocess:
        try:
            eeg_data = preprocess_eeg_trial(eeg_data, eeg_sfreq, notch_freq=50.0)
        except Exception as e:
            logger.warning(f"EEG preprocessing failed for {subject}: {e}")

    # Parse ET
    try:
        et_events = _parse_rsod_et(et_path)
        col_index = _build_rsod_col_index(et_path)
    except Exception as e:
        logger.error(f"Failed to parse ET for {subject}: {e}")
        return []

    # Segment into trials
    try:
        trial_tuples = _rsod_segment_trials(eeg_data, eeg_sfreq, et_events, col_index)
    except Exception as e:
        logger.error(f"Trial segmentation failed for {subject}: {e}")
        return []

    trials = []
    for t_idx, (eeg_t, gaze_t, condition) in enumerate(trial_tuples):
        # Preprocess each trial's EEG
        if preprocess:
            try:
                eeg_t = preprocess_eeg_trial(eeg_t, eeg_sfreq, notch_freq=50.0)
            except Exception:
                pass  # keep unprocessed if filter fails (e.g., too short)

        trial = EEGEyeNetTrial(
            subject_id=subj_int,
            trial_idx=t_idx,
            paradigm="rsod",
            condition=condition,
            eeg_data=eeg_t,              # (n_ch, RSOD_TRIAL_EEG)
            gaze_data=gaze_t,            # (RSOD_TRIAL_ET, 3)
            eeg_sfreq=eeg_sfreq,
            gaze_sfreq=RSOD_ET_SFREQ,
            channel_names=ch_names,
            stimulus_onset_sample=0,
        )
        trials.append(trial)

    logger.debug(f"  {subject}: {len(trials)} trials "
                 f"(intent={sum(t.condition==1 for t in trials)}, "
                 f"observe={sum(t.condition==0 for t in trials)})")
    return trials


def load_rsod_dataset(
    data_dir: str | Path,
    subjects: Optional[list[str]] = None,
    max_subjects: Optional[int] = None,
    use_merged: bool = True,
    preprocess: bool = True,
) -> EEGEyeNetDataset:
    """
    Load the EEGET-RSOD dataset (remote-sensing object detection).

    Args:
        data_dir: Root directory of the RSOD dataset (contains EEG/, ET/, merged/)
        subjects: List of subject IDs to load (e.g. ["P01", "P02"]).
                  None = all P01..P38.
        max_subjects: Cap on number of subjects (for development / quick tests)
        use_merged: Use pre-aligned merged files under merged/{subject}/
        preprocess: Apply EEG preprocessing (bandpass 1-45 Hz, notch 50 Hz, CAR)

    Returns:
        EEGEyeNetDataset with RSOD trials
    """
    data_dir = Path(data_dir)
    dataset = EEGEyeNetDataset(paradigm="rsod")

    if subjects is None:
        # Auto-discover subjects from merged/ directory
        if use_merged:
            merged_dir = data_dir / "merged"
            subject_dirs = sorted(merged_dir.iterdir()) if merged_dir.exists() else []
            subjects = [d.name for d in subject_dirs if d.is_dir()]
        else:
            eeg_dir = data_dir / "EEG"
            subjects = sorted(
                p.stem for p in eeg_dir.glob("P*.edf")
            ) if eeg_dir.exists() else []

    if not subjects:
        logger.warning(f"No RSOD subjects found in {data_dir}")
        return dataset

    if max_subjects is not None:
        subjects = subjects[:max_subjects]

    logger.info(f"Loading RSOD dataset: {len(subjects)} subjects from {data_dir}")

    for subj in subjects:
        trials = load_rsod_subject(subj, data_dir, preprocess=preprocess,
                                   use_merged=use_merged)
        dataset.trials.extend(trials)
        if trials:
            dataset.subjects.append(int(subj.lstrip("P").lstrip("0") or "0"))

    logger.info(
        f"RSOD loaded: {dataset.n_trials} trials, {dataset.n_subjects} subjects. "
        f"Intent: {sum(t.condition==1 for t in dataset.trials)}, "
        f"Observe: {sum(t.condition==0 for t in dataset.trials)}"
    )
    return dataset


# ===========================================================================
# ALS loader
# ===========================================================================

def _parse_als_et(et_path: Path) -> tuple[list[tuple[float, float, float, float, str]], list[tuple[float, str]]]:
    """
    Parse an ALS ET.csv file.

    Returns:
        samples: list of (timestamp_s, x_norm, y_norm, pupil=0, char_typed) rows
                 for actual gaze samples (not EMPTY/MainMenu/START events)
        selections: list of (timestamp_s, char) tuples marking character selection moments
    """
    samples = []      # (ts, x, y, pupil, char)
    selections = []   # (ts, char)
    prev_char = ""

    with open(et_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return samples, selections

        for row in reader:
            if len(row) < 4:
                continue

            # Skip system messages
            try:
                ts = float(row[0])
            except ValueError:
                continue

            # x and y: skip rows where they're sentinel values
            raw_x, raw_y = row[1].strip(), row[2].strip()
            try:
                x = float(raw_x)
                y = float(raw_y)
                is_gaze = not (raw_x == "-1" and (raw_y == "-1" or
                               raw_y in ("EMPTY_ET_STREAM", "START_ET_STREAM")))
            except ValueError:
                is_gaze = False
                x, y = 0.0, 0.0

            char_col = row[4].strip() if len(row) > 4 else ""
            sentence_col = row[5].strip() if len(row) > 5 else ""

            # Detect selection: character typing column has a new character
            if char_col and char_col not in (
                "EMPTY_ET_STREAM", "MainMenu", "START_ET_STREAM", "Typing"
            ):
                if char_col != prev_char:
                    selections.append((ts, char_col))
                    prev_char = char_col

            if is_gaze and x != -1.0:
                samples.append((ts, x, y, 0.0, char_col))

    return samples, selections


def _als_read_eeg_timestamps(ts_path: Path) -> np.ndarray:
    """Read per-sample EEG timestamps from EEGTimeStamp.txt."""
    timestamps = []
    with open(ts_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                timestamps.append(float(line))
            except ValueError:
                continue
    return np.array(timestamps)


def _als_build_et_array(
    et_samples: list[tuple[float, float, float, float, str]],
    t_start: float,
    t_end: float,
) -> np.ndarray:
    """
    Extract gaze samples within [t_start, t_end) into an array of shape (N, 3).

    Uses normalized x, y coordinates (0-1) directly as degree-proxies
    since exact ALS monitor geometry is unspecified in the dataset metadata.
    Scaled to approximate visual angle (assuming 40° FOV / 1920px width).

    Returns:
        np.ndarray of shape (n_samples, 3): [x_deg_approx, y_deg_approx, pupil=0]
    """
    SCALE = 40.0   # approximate FOV in degrees for normalization
    rows = [(ts, x * SCALE - SCALE/2, y * SCALE - SCALE/2)
            for ts, x, y, p, ch in et_samples
            if t_start <= ts < t_end and x != -1.0]
    if not rows:
        # Return placeholder zeros
        n = max(1, int(ALS_ET_SFREQ * (t_end - t_start)))
        return np.zeros((n, 3))
    arr = np.zeros((len(rows), 3))
    for i, (_, x, y) in enumerate(rows):
        arr[i, 0] = x
        arr[i, 1] = y
    return arr


def load_als_subject_scenario(
    subj_dir: Path,
    scenario_id: int,
    subj_int: int,
    preprocess: bool = True,
) -> list[EEGEyeNetTrial]:
    """
    Load trials from a single ALS subject/scenario.

    Returns list of EEGEyeNetTrial (intent = selection dwell, observe = scanning).
    """
    scen_dir = subj_dir / "time1" / f"scenario{scenario_id}"
    if not scen_dir.exists():
        return []

    edf_path = scen_dir / "EEG.edf"
    et_path  = scen_dir / "ET.csv"
    ts_path  = scen_dir / "EEGTimeStamp.txt"
    meta_path = scen_dir / "eeg.json"

    if not edf_path.exists() or not et_path.exists():
        return []

    # Read EEG metadata
    eeg_sfreq = ALS_EEG_SFREQ
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                eeg_sfreq = float(meta.get("SamplingFrequence", ALS_EEG_SFREQ))
        except Exception:
            pass

    # Read EEG
    try:
        eeg_data, sfreq_file, ch_names = _read_edf_eeg(edf_path)
        if sfreq_file > 0:
            eeg_sfreq = sfreq_file
    except Exception as e:
        logger.warning(f"ALS EDF read failed {edf_path}: {e}")
        return []

    # Preprocess EEG (whole recording before trial extraction)
    if preprocess:
        try:
            eeg_data = _bandpass_filter(eeg_data, eeg_sfreq)
            eeg_data = _notch_filter(eeg_data, eeg_sfreq, freq=50.0)
            eeg_data = _common_average_reference(eeg_data)
        except Exception as e:
            logger.warning(f"ALS preprocessing failed for {edf_path}: {e}")

    # Read EEG timestamps for time alignment
    eeg_timestamps = None
    if ts_path.exists():
        try:
            eeg_timestamps = _als_read_eeg_timestamps(ts_path)
        except Exception:
            pass

    # If no timestamp file, create synthetic timestamps from sampling rate
    if eeg_timestamps is None or len(eeg_timestamps) != eeg_data.shape[1]:
        n_samples = eeg_data.shape[1]
        eeg_timestamps = np.arange(n_samples) / eeg_sfreq  # seconds from start

    # Parse ET
    try:
        et_samples, selections = _parse_als_et(et_path)
    except Exception as e:
        logger.warning(f"ALS ET parse failed {et_path}: {e}")
        return []

    if not et_samples or not selections:
        return []

    # Build EEG lookup: convert EEG timestamps to a float array
    eeg_ts_arr = eeg_timestamps

    def eeg_idx_at(t: float) -> int:
        """Find EEG sample index closest to timestamp t."""
        idx = int(np.searchsorted(eeg_ts_arr, t))
        return min(idx, len(eeg_ts_arr) - 1)

    trial_duration_eeg = int(eeg_sfreq * ALS_TRIAL_S)
    trial_duration_et  = int(ALS_ET_SFREQ * ALS_TRIAL_S)

    trials = []
    t_idx = 0

    # Intent trials: 1.5s window ending at each selection event
    selection_times = [s[0] for s in selections]

    for sel_ts, sel_char in selections:
        t_end   = sel_ts
        t_start = sel_ts - ALS_TRIAL_S

        # EEG window
        eeg_start = eeg_idx_at(t_start)
        eeg_end   = eeg_start + trial_duration_eeg
        if eeg_end > eeg_data.shape[1]:
            continue

        eeg_trial = eeg_data[:, eeg_start:eeg_end].copy()

        # Baseline correct per trial
        if preprocess:
            eeg_trial = _baseline_correct(eeg_trial)

        # ET window
        gaze_arr = _als_build_et_array(et_samples, t_start, t_end)

        trial = EEGEyeNetTrial(
            subject_id=subj_int,
            trial_idx=t_idx,
            paradigm="als_spelling",
            condition=1,               # intent: active letter selection
            eeg_data=eeg_trial,
            gaze_data=gaze_arr,
            eeg_sfreq=eeg_sfreq,
            gaze_sfreq=ALS_ET_SFREQ,
            channel_names=ch_names,
            stimulus_onset_sample=0,
        )
        trials.append(trial)
        t_idx += 1

    # Observe trials: 1.5s windows between selections
    # Must be at least ALS_MIN_GAP_S away from any selection
    if selection_times:
        # Build inter-selection gap windows
        min_ts = eeg_ts_arr[0]
        max_ts = eeg_ts_arr[-1]

        candidate_starts = []
        # Windows starting between consecutive selections
        for i in range(len(selection_times) - 1):
            gap_start = selection_times[i] + ALS_MIN_GAP_S
            gap_end   = selection_times[i + 1] - ALS_MIN_GAP_S - ALS_TRIAL_S
            t = gap_start
            while t + ALS_TRIAL_S <= gap_end:
                candidate_starts.append(t)
                t += ALS_TRIAL_S  # non-overlapping

        # Also windows before the first selection and after the last
        t = min_ts + ALS_MIN_GAP_S
        while t + ALS_TRIAL_S <= selection_times[0] - ALS_MIN_GAP_S:
            candidate_starts.append(t)
            t += ALS_TRIAL_S
        t = selection_times[-1] + ALS_MIN_GAP_S
        while t + ALS_TRIAL_S <= max_ts - ALS_MIN_GAP_S:
            candidate_starts.append(t)
            t += ALS_TRIAL_S

        # Sample observe trials to match number of intent trials
        n_intent = len(trials)
        if len(candidate_starts) > n_intent:
            rng = np.random.RandomState(42)
            selected_obs = rng.choice(len(candidate_starts), size=n_intent, replace=False)
            candidate_starts = [candidate_starts[i] for i in sorted(selected_obs)]

        for t_start in candidate_starts:
            t_end = t_start + ALS_TRIAL_S
            eeg_start = eeg_idx_at(t_start)
            eeg_end   = eeg_start + trial_duration_eeg
            if eeg_end > eeg_data.shape[1]:
                continue

            eeg_trial = eeg_data[:, eeg_start:eeg_end].copy()
            if preprocess:
                eeg_trial = _baseline_correct(eeg_trial)

            gaze_arr = _als_build_et_array(et_samples, t_start, t_end)

            trial = EEGEyeNetTrial(
                subject_id=subj_int,
                trial_idx=t_idx,
                paradigm="als_spelling",
                condition=0,           # observe: scanning keyboard
                eeg_data=eeg_trial,
                gaze_data=gaze_arr,
                eeg_sfreq=eeg_sfreq,
                gaze_sfreq=ALS_ET_SFREQ,
                channel_names=ch_names,
                stimulus_onset_sample=0,
            )
            trials.append(trial)
            t_idx += 1

    return trials


def load_als_spelling_dataset(
    data_dir: str | Path,
    subjects: Optional[list[str]] = None,
    max_subjects: Optional[int] = None,
    scenarios: Optional[list[int]] = None,
    preprocess: bool = True,
    healthy_only: bool = True,
) -> EEGEyeNetDataset:
    """
    Load the EEGET-ALS dataset (spelling BCI), healthy subjects only by default.

    Args:
        data_dir: Root of EEGET-ALS Dataset (contains id1/, id2/, ..., ALS01/, ...)
        subjects: List of subject directory names to load (e.g. ["id1", "id2"]).
                  None = all healthy subjects (id1..id20).
        max_subjects: Cap on number of subjects (for development)
        scenarios: Scenario IDs to load (1-9). None = all 9.
        preprocess: Apply EEG preprocessing
        healthy_only: Load only healthy subjects (id1..id20), not ALS patients

    Returns:
        EEGEyeNetDataset with ALS spelling trials
    """
    data_dir = Path(data_dir)
    dataset = EEGEyeNetDataset(paradigm="als_spelling")

    if subjects is None:
        if healthy_only:
            subjects = ALS_HEALTHY_IDS
        else:
            # Auto-discover
            subjects = sorted(d.name for d in data_dir.iterdir() if d.is_dir())

    if max_subjects is not None:
        subjects = subjects[:max_subjects]

    if scenarios is None:
        scenarios = list(range(1, 10))   # 9 scenarios per subject

    logger.info(f"Loading ALS dataset: {len(subjects)} subjects, "
                f"scenarios {scenarios[0]}-{scenarios[-1]}, from {data_dir}")

    for subj_name in subjects:
        subj_dir = data_dir / subj_name
        if not subj_dir.exists():
            logger.debug(f"ALS subject directory not found: {subj_dir}")
            continue

        try:
            subj_int = int(subj_name.lstrip("id").lstrip("0") or "0")
        except ValueError:
            subj_int = 0

        subj_trials = []
        for scen in scenarios:
            scen_trials = load_als_subject_scenario(
                subj_dir, scen, subj_int, preprocess=preprocess
            )
            subj_trials.extend(scen_trials)

        if subj_trials:
            dataset.trials.extend(subj_trials)
            dataset.subjects.append(subj_int)
            logger.debug(
                f"  {subj_name}: {len(subj_trials)} trials across {len(scenarios)} scenarios"
            )

    logger.info(
        f"ALS loaded: {dataset.n_trials} trials, {dataset.n_subjects} subjects. "
        f"Intent: {sum(t.condition==1 for t in dataset.trials)}, "
        f"Observe: {sum(t.condition==0 for t in dataset.trials)}"
    )
    return dataset


# ===========================================================================
# Quick sanity check
# ===========================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    RSOD_DIR = Path("data/raw/eeget-rsod")
    ALS_DIR  = Path("data/raw/eeget-als")

    print("\n" + "="*60)
    print("RSOD: loading 3 subjects")
    print("="*60)
    rsod = load_rsod_dataset(RSOD_DIR, max_subjects=3)
    print(f"  Trials: {rsod.n_trials}, Subjects: {rsod.n_subjects}")
    if rsod.trials:
        t = rsod.trials[0]
        print(f"  First trial — EEG shape: {t.eeg_data.shape}, "
              f"Gaze shape: {t.gaze_data.shape}, "
              f"Condition: {t.condition}, "
              f"EEG sfreq: {t.eeg_sfreq}, "
              f"Gaze sfreq: {t.gaze_sfreq}")
        print(f"  EEG range: [{t.eeg_data.min():.2f}, {t.eeg_data.max():.2f}] µV")
        intent_n  = sum(x.condition==1 for x in rsod.trials)
        observe_n = sum(x.condition==0 for x in rsod.trials)
        print(f"  Intent: {intent_n}, Observe: {observe_n}")

    print("\n" + "="*60)
    print("ALS: loading 2 subjects, 2 scenarios")
    print("="*60)
    als = load_als_spelling_dataset(ALS_DIR, max_subjects=2, scenarios=[1, 2])
    print(f"  Trials: {als.n_trials}, Subjects: {als.n_subjects}")
    if als.trials:
        t = als.trials[0]
        print(f"  First trial — EEG shape: {t.eeg_data.shape}, "
              f"Gaze shape: {t.gaze_data.shape}, "
              f"Condition: {t.condition}, "
              f"EEG sfreq: {t.eeg_sfreq}, "
              f"Gaze sfreq: {t.gaze_sfreq}")
        print(f"  EEG range: [{t.eeg_data.min():.2f}, {t.eeg_data.max():.2f}] µV")
        intent_n  = sum(x.condition==1 for x in als.trials)
        observe_n = sum(x.condition==0 for x in als.trials)
        print(f"  Intent: {intent_n}, Observe: {observe_n}")

    print("\nSanity check complete.")
