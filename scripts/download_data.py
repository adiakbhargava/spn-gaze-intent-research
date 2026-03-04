#!/usr/bin/env python3
"""
Download EEGEyeNet Dataset

Downloads the EEGEyeNet dataset from the official OSF repository (Kastrati et al., 2021).
Data files are stored in the Dropbox storage provider on the OSF node, under `prepared/`.

Two variants are available for each paradigm:
  - *_max.npz: Full version (more trials, float64 precision) — ~11 GB for prosaccade
  - *_min.npz: Minimal version (fewer trials, smaller size) — ~1.5 GB for prosaccade

IMPORTANT: Both variants contain only EEG data + target positions (labels). They do NOT
include raw eye-tracking time series. Gaze trajectories are derived from target positions
by our preprocessing script (scripts/preprocess_real_data.py), which creates plausible
saccade profiles from the known fixation targets.

For genuine eye-tracking recordings, you would need the raw per-subject .mat files
from the `dots_data/synchronised_max/` folder on OSF (not currently automated).

Supports programmatic download via the OSF REST API with progress reporting,
plus hardcoded GUID-based fallback URLs for reliability.

Usage:
    # Download the prosaccade paradigm (recommended)
    python scripts/download_data.py --paradigm prosaccade

    # Generate synthetic data for development (no download needed)
    python scripts/download_data.py --synthetic --n-subjects 20

    # Re-download even if files already exist
    python scripts/download_data.py --paradigm prosaccade --force

    # Inspect what's available on OSF without downloading
    python scripts/download_data.py --list-remote
"""

import argparse
import json
import logging
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.eegeyenet_loader import generate_synthetic_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# EEGEyeNet OSF repository — Kastrati et al. (2021)
EEGEYENET_OSF_URL = "https://osf.io/ktv7m/"
EEGEYENET_OSF_NODE = "ktv7m"
OSF_API_BASE = "https://api.osf.io/v2"

# The EEGEyeNet data files live in the Dropbox storage provider (NOT osfstorage)
# under the `prepared/` folder on the root node.
OSF_STORAGE_PROVIDER = "dropbox"
OSF_DATA_FOLDER = "prepared"

# Known file GUIDs on OSF — allows direct download without API enumeration.
# These are stable identifiers that resolve to https://osf.io/download/<guid>/
KNOWN_FILE_GUIDS = {
    "prosaccade": {
        "Position_task_with_dots_synchronised_max.npz": "9vbq2",         # Full (more trials, float64)
        "Position_task_with_dots_synchronised_max_hilbert.npz": "ksw6x",  # Full + Hilbert transform
        "Position_task_with_dots_synchronised_min.npz": "ge87t",          # Minimal (fewer trials)
        "Position_task_with_dots_synchronised_min_hilbert.npz": "bmrn9",  # Minimal + Hilbert
    },
    "antisaccade": {
        "LR_task_with_antisaccade_synchronised_max.npz": "v8u7h",
        "LR_task_with_antisaccade_synchronised_max_hilbert.npz": "4vh5c",
        "LR_task_with_antisaccade_synchronised_min.npz": "a897e",
        "LR_task_with_antisaccade_synchronised_min_hilbert.npz": "jkrzh",
    },
}

# File patterns for each paradigm on OSF
# The EEGEyeNet dataset uses "Position_task" for prosaccade data
PARADIGM_FILE_PATTERNS = {
    "prosaccade": [
        "Position_task_with_dots_synchronised_max",     # Full version (more trials, float64)
        "Position_task_with_dots_synchronised_min",      # Minimal version (fewer trials)
    ],
    "antisaccade": [
        "LR_task_with_antisaccade_synchronised_max",
        "LR_task_with_antisaccade_synchronised_min",
    ],
    "visual_symbol_search": [
        "Direction_task_with_dots_synchronised_max",
        "Direction_task_with_dots_synchronised_min",
    ],
}

EEGEYENET_PARADIGM_INFO = {
    "prosaccade": {
        "description": "Directed saccades to targets — closest to intent selection (RECOMMENDED)",
        "size_gb": "~2-4",
        "n_subjects": 356,
        "n_trials_per_subject": "~80-120",
        "why": (
            "Best fit for this pipeline: subjects look TOWARD a target (analogous to "
            "intent selection in gaze interfaces). Produces the clearest SPN signal "
            "(the anticipatory negativity from occipitoparietal electrodes that "
            "Reddy et al. CHI 2024 used to confirm intent)."
        ),
    },
    "antisaccade": {
        "description": "Saccades away from targets — tests inhibitory control",
        "size_gb": "~2-4",
        "n_subjects": 356,
        "n_trials_per_subject": "~80-120",
        "why": (
            "Less relevant for intent decoding. Subjects look AWAY from the target, "
            "which is the opposite of the selection behavior we want to classify."
        ),
    },
    "visual_symbol_search": {
        "description": "Free viewing / visual search — complex gaze patterns",
        "size_gb": "~25+",
        "n_subjects": 356,
        "n_trials_per_subject": "varies",
        "why": (
            "Too large and less targeted. Free-viewing doesn't map cleanly to the "
            "intent vs. observe binary classification this pipeline uses."
        ),
    },
}

DATA_DIR = Path("data")


# ---------------------------------------------------------------------------
# OSF API helpers
# ---------------------------------------------------------------------------

def _osf_api_get(url: str, timeout: int = 30) -> dict:
    """Make a GET request to the OSF API and return parsed JSON."""
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def list_osf_files(node_id: str = EEGEYENET_OSF_NODE) -> list[dict]:
    """List all files in the OSF node's Dropbox/prepared folder, following pagination.

    The EEGEyeNet dataset stores its prepared NPZ files in a Dropbox storage
    provider (not the default osfstorage).  We query the ``prepared/`` sub-folder
    directly so we only enumerate the relevant files.
    """
    # Try the Dropbox prepared folder first (where the NPZ files actually are)
    urls_to_try = [
        f"{OSF_API_BASE}/nodes/{node_id}/files/{OSF_STORAGE_PROVIDER}/{OSF_DATA_FOLDER}/",
        f"{OSF_API_BASE}/nodes/{node_id}/files/{OSF_STORAGE_PROVIDER}/",
        f"{OSF_API_BASE}/nodes/{node_id}/files/osfstorage/",
    ]

    for url in urls_to_try:
        try:
            all_files = _list_osf_directory(url)
            if all_files:
                return all_files
        except (urllib.error.HTTPError, urllib.error.URLError):
            continue

    return []


def _list_osf_directory(url: str) -> list[dict]:
    """List files at an OSF directory URL, following pagination and recursing into folders."""
    all_files = []
    while url:
        data = _osf_api_get(url)
        for item in data.get("data", []):
            attrs = item.get("attributes", {})
            links = item.get("links", {})
            all_files.append({
                "name": attrs.get("name", ""),
                "kind": attrs.get("kind", ""),
                "size": attrs.get("size") or 0,
                "download": links.get("download"),
                "path": attrs.get("materialized_path", ""),
            })
            # Recurse into sub-folders
            if attrs.get("kind") == "folder":
                folder_url = item.get("relationships", {}).get(
                    "files", {}
                ).get("links", {}).get("related", {}).get("href")
                if folder_url:
                    all_files.extend(_list_osf_directory(folder_url))

        # Follow pagination
        url = data.get("links", {}).get("next")

    return all_files


def find_paradigm_files(
    osf_files: list[dict],
    paradigm: str,
) -> list[dict]:
    """Find files matching a paradigm's file patterns, preferring full over _min."""
    patterns = PARADIGM_FILE_PATTERNS.get(paradigm, [])
    matches = []
    for f in osf_files:
        if f["kind"] != "file":
            continue
        for pattern in patterns:
            if pattern.lower() in f["name"].lower():
                is_min = "_min" in f["name"].lower()
                matches.append({**f, "is_min": is_min, "pattern": pattern})
    # Sort: full versions first, then _min
    matches.sort(key=lambda x: (x["is_min"], x["name"]))
    return matches


def download_file(url: str, dest: Path, expected_size: int = 0) -> bool:
    """Download a file from URL with progress reporting."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")

    logger.info(f"Downloading: {dest.name}")
    if expected_size > 0:
        logger.info(f"  Size: {expected_size / 1e9:.2f} GB")

    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=60) as resp:
            total = int(resp.headers.get("Content-Length", expected_size) or 0)
            downloaded = 0
            last_report = 0

            with open(tmp, "wb") as f:
                while True:
                    chunk = resp.read(1024 * 1024)  # 1 MB chunks
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    # Report progress every 10%
                    if total > 0:
                        pct = downloaded / total * 100
                        if pct - last_report >= 10:
                            logger.info(f"  Progress: {pct:.0f}% ({downloaded / 1e6:.0f} MB / {total / 1e6:.0f} MB)")
                            last_report = pct

        tmp.rename(dest)
        logger.info(f"  Saved: {dest} ({downloaded / 1e6:.1f} MB)")
        return True

    except Exception as e:
        logger.warning(f"  Download failed: {e}")
        if tmp.exists():
            tmp.unlink()
        return False


# ---------------------------------------------------------------------------
# Main download logic
# ---------------------------------------------------------------------------

def download_eegeyenet(
    paradigm: str = "prosaccade",
    output_dir: Path = DATA_DIR / "raw" / "eegeyenet",
    max_subjects: int | None = None,
    force: bool = False,
):
    """Download EEGEyeNet data from OSF, falling back to manual instructions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paradigm_dir = output_dir / paradigm
    paradigm_dir.mkdir(parents=True, exist_ok=True)
    preprocessed_dir = paradigm_dir / "preprocessed"
    preprocessed_dir.mkdir(parents=True, exist_ok=True)

    info = EEGEYENET_PARADIGM_INFO.get(paradigm, {})

    logger.info(f"\n{'=' * 60}")
    logger.info(f"EEGEyeNet Dataset — {paradigm} paradigm")
    logger.info(f"{'=' * 60}")
    logger.info(f"Description: {info.get('description', paradigm)}")
    logger.info(f"Size: {info.get('size_gb', 'unknown')}")
    logger.info(f"Why: {info.get('why', '')}")

    # Check if data already exists
    existing_npz = list(output_dir.glob("*synchronised*.npz"))
    existing_preprocessed = (preprocessed_dir / "eeg_data.npy").exists()

    if existing_preprocessed and not force:
        eeg = np.load(preprocessed_dir / "eeg_data.npy", mmap_mode="r")
        logger.info(f"\nPreprocessed data already found! EEG shape: {eeg.shape}")
        logger.info("Ready to train:")
        logger.info(f"  python scripts/train.py --data-dir {output_dir} --paradigm {paradigm}")
        logger.info("\nTo re-download, use --force")
        return

    if existing_npz and not force:
        logger.info(f"\nRaw NPZ already found: {[f.name for f in existing_npz]}")
        logger.info("Run preprocessing:")
        logger.info(f"  python scripts/preprocess_real_data.py --npz {existing_npz[0]}")
        logger.info("\nTo re-download, use --force")
        return

    # Try programmatic download via OSF API
    logger.info("\nQuerying OSF API for available files...")
    downloaded = False
    try:
        osf_files = list_osf_files()
        matches = find_paradigm_files(osf_files, paradigm)

        if matches:
            logger.info(f"Found {len(matches)} matching file(s) on OSF:")
            for m in matches:
                tag = " (full — preferred)" if not m["is_min"] else " (minimal)"
                size_str = f"{m['size'] / 1e9:.2f} GB" if m['size'] > 0 else "unknown size"
                logger.info(f"  {m['name']} — {size_str}{tag}")

            # Download the best match (full version preferred over _min)
            target = matches[0]
            dest = output_dir / target["name"]

            if dest.exists() and not force:
                logger.info(f"\nFile already exists: {dest}")
                downloaded = True
            elif target["download"]:
                downloaded = download_file(target["download"], dest, target["size"])
            else:
                logger.warning(f"  No download link for {target['name']}")
        else:
            logger.info("No matching files found via API enumeration — trying known GUIDs...")

    except (urllib.error.URLError, TimeoutError, OSError) as e:
        logger.info(f"Could not reach OSF API ({e.__class__.__name__}) — trying known GUIDs...")

    # Fallback: use known stable GUID-based download URLs
    if not downloaded:
        downloaded = _try_guid_download(paradigm, output_dir, force)

    if downloaded:
        npz_path = list(output_dir.glob("*synchronised*.npz"))
        if npz_path:
            logger.info(f"\nDownload complete! Next step — preprocess the data:")
            logger.info(f"  python scripts/preprocess_real_data.py --npz {npz_path[0]}")
            logger.info(f"\nThen train:")
            logger.info(f"  python scripts/train.py --data-dir {output_dir} --paradigm {paradigm}")
        return

    # Fall back to manual instructions
    _print_manual_instructions(paradigm, output_dir, paradigm_dir, preprocessed_dir, max_subjects)


def _try_guid_download(paradigm: str, output_dir: Path, force: bool = False) -> bool:
    """Try downloading via known stable GUID-based URLs (bypasses API enumeration)."""
    guids = KNOWN_FILE_GUIDS.get(paradigm, {})
    if not guids:
        return False

    # Prefer the full (_max) version over _min; skip hilbert variants
    preferred_order = [
        name for name in guids
        if "_hilbert" not in name and "_max" in name
    ] + [
        name for name in guids
        if "_hilbert" not in name and "_min" in name
    ]

    for filename in preferred_order:
        guid = guids[filename]
        dest = output_dir / filename
        if dest.exists() and not force:
            logger.info(f"File already exists: {dest}")
            return True

        url = f"https://osf.io/download/{guid}/"
        tag = "(full — more trials, float64)" if "_max" in filename else "(minimal — fewer trials)"
        logger.info(f"Downloading via GUID: {filename} {tag}")
        if download_file(url, dest):
            return True
        else:
            logger.info(f"  GUID download failed for {filename}, trying next...")

    return False


def _print_manual_instructions(paradigm, output_dir, paradigm_dir, preprocessed_dir, max_subjects):
    """Print step-by-step manual download instructions."""
    logger.info(f"\n{'=' * 60}")
    logger.info("MANUAL DOWNLOAD INSTRUCTIONS")
    logger.info(f"{'=' * 60}")
    logger.info(
        f"\n1. Go to the EEGEyeNet OSF repository:\n"
        f"   {EEGEYENET_OSF_URL}\n"
        f"\n2. Navigate to: Files > and look for the prosaccade / Position task data\n"
        f"\n3. You have two options:\n"
        f"\n   Option A — Download the NPZ file (RECOMMENDED):\n"
        f"     Look for 'Position_task_with_dots_synchronised*.npz'\n"
        f"     (Prefer the version WITHOUT '_min' if available — it includes\n"
        f"     actual eye-tracking time series alongside EEG data.)\n"
        f"     Place it in: {output_dir}/\n"
        f"     Then preprocess:\n"
        f"       python scripts/preprocess_real_data.py --npz <path_to_npz>\n"
        f"\n   Option B — Raw MATLAB .mat files (larger, per-subject):\n"
        f"     Download EP*.mat files and place them in:\n"
        f"     {paradigm_dir}/\n"
        f"     Requires h5py or scipy for loading.\n"
        f"\n4. Then train with real data:\n"
        f"   python scripts/train.py --data-dir {output_dir} --paradigm {paradigm}"
    )

    if max_subjects:
        logger.info(f"\n   (with --max-subjects {max_subjects} for quick testing)")


def list_remote_files():
    """List all files available on the EEGEyeNet OSF repository."""
    logger.info(f"Querying OSF repository: {EEGEYENET_OSF_URL}")
    logger.info(f"  Storage provider: {OSF_STORAGE_PROVIDER}/{OSF_DATA_FOLDER}/")

    try:
        files = list_osf_files()
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        logger.error(f"Could not reach OSF API: {e}")
        logger.info("\nKnown files (from hardcoded GUIDs):")
        for paradigm, guids in KNOWN_FILE_GUIDS.items():
            logger.info(f"\n  {paradigm}:")
            for name, guid in guids.items():
                logger.info(f"    {name}  (https://osf.io/download/{guid}/)")
        return

    if not files:
        logger.info("No files found via API — showing known GUIDs instead")
        for paradigm, guids in KNOWN_FILE_GUIDS.items():
            logger.info(f"\n  {paradigm}:")
            for name, guid in guids.items():
                logger.info(f"    {name}  (https://osf.io/download/{guid}/)")
        return

    logger.info(f"\nFound {len(files)} items on OSF:\n")
    for f in sorted(files, key=lambda x: x["path"]):
        if f["kind"] == "file":
            size_str = f"{f['size'] / 1e6:.1f} MB" if f.get('size') else "? MB"
            logger.info(f"  {f['path']}  ({size_str})")
        else:
            logger.info(f"  {f['path']}/")

    # Highlight paradigm matches
    for paradigm in PARADIGM_FILE_PATTERNS:
        matches = find_paradigm_files(files, paradigm)
        if matches:
            logger.info(f"\n  {paradigm} paradigm matches:")
            for m in matches:
                tag = " [FULL]" if not m["is_min"] else " [minimal]"
                logger.info(f"    {m['name']}{tag}")


def generate_synthetic(
    n_subjects: int = 20,
    trials_per_subject: int = 80,
    output_dir: Path = DATA_DIR / "synthetic",
):
    """Generate synthetic EEGEyeNet-like dataset for development."""
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating synthetic dataset: {n_subjects} subjects, {trials_per_subject} trials each")
    dataset = generate_synthetic_dataset(
        n_subjects=n_subjects,
        trials_per_subject=trials_per_subject,
    )

    # Save as NumPy arrays
    eeg_all = np.array([t.eeg_data for t in dataset.trials])
    gaze_all = np.array([t.gaze_data for t in dataset.trials])
    labels = dataset.get_labels()
    subject_ids = dataset.get_subject_ids()
    channel_names = np.array(dataset.trials[0].channel_names if dataset.trials else [])

    np.save(output_dir / "eeg_data.npy", eeg_all)
    np.save(output_dir / "gaze_data.npy", gaze_all)
    np.save(output_dir / "labels.npy", labels)
    np.save(output_dir / "subject_ids.npy", subject_ids)
    np.save(output_dir / "channel_names.npy", channel_names)

    logger.info(
        f"Synthetic dataset saved to {output_dir}:\n"
        f"  eeg_data.npy:     {eeg_all.shape} ({eeg_all.nbytes / 1e6:.1f} MB)\n"
        f"  gaze_data.npy:    {gaze_all.shape} ({gaze_all.nbytes / 1e6:.1f} MB)\n"
        f"  labels.npy:       {labels.shape}\n"
        f"  subject_ids.npy:  {subject_ids.shape}\n"
        f"  Trials: {len(dataset.trials)} ({np.sum(labels == 1)} intent, {np.sum(labels == 0)} observe)"
    )


def main():
    parser = argparse.ArgumentParser(description="Download or generate EEGEyeNet data")
    parser.add_argument("--paradigm", default="prosaccade",
                       choices=["prosaccade", "antisaccade", "visual_symbol_search"])
    parser.add_argument("--synthetic", action="store_true",
                       help="Generate synthetic dataset instead of downloading")
    parser.add_argument("--n-subjects", type=int, default=20,
                       help="Number of synthetic subjects")
    parser.add_argument("--trials-per-subject", type=int, default=80,
                       help="Trials per synthetic subject")
    parser.add_argument("--max-subjects", type=int, default=None,
                       help="Limit number of subjects to download/load (for quick testing)")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Custom output directory")
    parser.add_argument("--force", action="store_true",
                       help="Re-download even if files already exist")
    parser.add_argument("--list-paradigms", action="store_true",
                       help="Show info about each paradigm and exit")
    parser.add_argument("--list-remote", action="store_true",
                       help="List files available on the OSF repository")
    args = parser.parse_args()

    if args.list_paradigms:
        print("\nEEGEyeNet Paradigm Options:\n")
        for name, info in EEGEYENET_PARADIGM_INFO.items():
            rec = " *** RECOMMENDED ***" if name == "prosaccade" else ""
            print(f"  {name}{rec}")
            print(f"    {info['description']}")
            print(f"    Size: {info['size_gb']}, Subjects: {info['n_subjects']}")
            print(f"    Rationale: {info['why']}")
            print()
        return

    if args.list_remote:
        list_remote_files()
        return

    if args.synthetic:
        output = Path(args.output_dir) if args.output_dir else DATA_DIR / "synthetic"
        generate_synthetic(args.n_subjects, args.trials_per_subject, output)
    else:
        output = Path(args.output_dir) if args.output_dir else DATA_DIR / "raw" / "eegeyenet"
        download_eegeyenet(args.paradigm, output, max_subjects=args.max_subjects, force=args.force)


if __name__ == "__main__":
    main()
