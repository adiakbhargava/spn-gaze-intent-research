"""
BCI2000 .dat File Parser

Parses the binary BCI2000 .dat format used in clinical BCI trials, including
the Prabhakaran lab's stroke rehabilitation protocol (NCT02098265).

BCI2000 .dat file structure:
    1. Text header (ends with \\r\\n\\r\\n) containing:
       - HeaderLen, SourceCh, StatevectorLen, DataFormat
       - Parameter definitions (nested sections)
       - State definitions (bit positions in state vector)
    2. Binary data blocks, each containing:
       - Signal data: (n_channels × n_samples_per_block) in DataFormat
       - State vector: (StatevectorLen bytes × n_samples_per_block)

Key states for latency analysis:
    - Roundtrip: end-to-end processing latency per block (ms equivalent)
    - SourceTime: timestamp when source module acquired the block
    - StimulusCode: current stimulus condition
    - ResultCode: trial outcome (hit/miss)
    - Feedback: whether feedback is being presented

Reference:
    Schalk et al. "BCI2000: A General-Purpose Brain-Computer Interface (BCI)
    System" IEEE TBME, 2004.
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# BCI2000 data format types
DATA_FORMATS = {
    "int16": ("h", 2),
    "int32": ("i", 4),
    "float24": (None, 3),  # Special case
    "float32": ("f", 4),
}


@dataclass
class BCI2000State:
    """A state variable definition from BCI2000 header."""
    name: str
    length: int    # bits
    start_bit: int
    byte_offset: int
    default_value: int = 0

    def extract(self, state_vector: bytes) -> int:
        """Extract this state's value from a state vector byte array."""
        byte_idx = self.byte_offset
        bit_offset = self.start_bit

        # Read enough bytes to cover all bits
        n_bytes = (bit_offset + self.length + 7) // 8
        value = 0
        for i in range(min(n_bytes, len(state_vector) - byte_idx)):
            value |= state_vector[byte_idx + i] << (8 * i)

        # Extract the specific bits
        value >>= bit_offset
        value &= (1 << self.length) - 1
        return value


@dataclass
class BCI2000Header:
    """Parsed BCI2000 .dat file header."""
    header_len: int = 0
    source_ch: int = 0
    statevector_len: int = 0
    data_format: str = "int16"
    sampling_rate: float = 256.0
    sample_block_size: int = 32
    states: dict[str, BCI2000State] = field(default_factory=dict)
    parameters: dict[str, str] = field(default_factory=dict)

    @property
    def bytes_per_sample(self) -> int:
        _, size = DATA_FORMATS.get(self.data_format, ("h", 2))
        return size

    @property
    def block_data_bytes(self) -> int:
        return self.source_ch * self.sample_block_size * self.bytes_per_sample

    @property
    def block_state_bytes(self) -> int:
        return self.statevector_len * self.sample_block_size

    @property
    def block_total_bytes(self) -> int:
        return self.block_data_bytes + self.block_state_bytes


@dataclass
class BCI2000Run:
    """A parsed BCI2000 recording run."""
    filepath: Path
    header: BCI2000Header
    signal: np.ndarray = field(repr=False)  # (n_channels, n_total_samples)
    states_per_sample: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    n_blocks: int = 0

    @property
    def duration_sec(self) -> float:
        return self.signal.shape[1] / self.header.sampling_rate

    @property
    def n_samples(self) -> int:
        return self.signal.shape[1]

    def get_state(self, name: str) -> Optional[np.ndarray]:
        return self.states_per_sample.get(name)

    def get_roundtrip_ms(self) -> Optional[np.ndarray]:
        """Get Roundtrip latency in milliseconds per block.

        Note: In BCI2000 v1.1, the Roundtrip state stores a raw 16-bit
        timestamp (ms modulo 65536), NOT an actual latency delta.  The
        values returned here are the raw state values; use
        :meth:`get_block_interval_ms` for meaningful timing.
        """
        rt = self.get_state("Roundtrip")
        if rt is None:
            return None
        # Roundtrip is stored per-sample but constant within a block,
        # so take one value per block
        block_size = self.header.sample_block_size
        n_blocks = len(rt) // block_size
        return rt[::block_size][:n_blocks].astype(np.float64)

    def get_source_time_ms(self) -> Optional[np.ndarray]:
        """Get SourceTime timestamps in ms per block."""
        st = self.get_state("SourceTime")
        if st is None:
            return None
        block_size = self.header.sample_block_size
        n_blocks = len(st) // block_size
        return st[::block_size][:n_blocks].astype(np.float64)

    def get_block_interval_ms(self) -> Optional[np.ndarray]:
        """Get inter-block intervals in ms, handling 16-bit wraparound.

        SourceTime is a 16-bit ms counter (wraps at 65536).  This method
        computes successive deltas with correct wrap handling, giving the
        actual time between consecutive data blocks — the fundamental
        BCI2000 processing cadence.

        Returns:
            Array of inter-block intervals in ms, or None if SourceTime
            state is not available.
        """
        st = self.get_source_time_ms()
        if st is None or len(st) < 2:
            return None
        deltas = np.diff(st)
        # Handle 16-bit unsigned wraparound (modulo 65536)
        deltas = np.where(deltas < 0, deltas + 65536, deltas)
        # Filter out implausible values (> 10 seconds)
        deltas = np.where(deltas > 10000, np.nan, deltas)
        return deltas


def _extract_kv(text: str, key: str) -> str | None:
    """Extract value for *key*= from a space-delimited key=value string.

    Handles both ``key=value`` and ``key= value`` forms used by BCI2000.
    Returns the first whitespace-delimited token after the '=' or None.
    """
    idx = text.find(key + "=")
    if idx == -1:
        return None
    after_eq = text[idx + len(key) + 1:].lstrip()
    return after_eq.split()[0] if after_eq else None


def parse_header(raw_header: str) -> BCI2000Header:
    """Parse the text header of a BCI2000 .dat file."""
    header = BCI2000Header()
    # Normalize line endings: strip \r from each line
    lines = [line.strip("\r").strip() for line in raw_header.split("\n")]

    in_state_section = False

    # -----------------------------------------------------------
    # The first line of a real BCI2000 file packs ALL core fields
    # on a single line, e.g.:
    #   BCI2000V= 1.1 HeaderLen= 20886 SourceCh= 16 ...
    # Parse them with a key-value scanner before entering the
    # per-line loop.
    # -----------------------------------------------------------
    if lines:
        first_line = lines[0]
        val = _extract_kv(first_line, "HeaderLen")
        if val is not None:
            header.header_len = int(val)
        val = _extract_kv(first_line, "SourceCh")
        if val is not None:
            header.source_ch = int(val)
        val = _extract_kv(first_line, "StatevectorLen")
        if val is not None:
            header.statevector_len = int(val)
        val = _extract_kv(first_line, "DataFormat")
        if val is not None:
            header.data_format = val.lower()

    for line in lines:
        if not line:
            continue

        # Parse BCI2000 parameter lines
        # Format: Section type Name= Value DefaultValue LowRange HighRange // comment
        if "SamplingRate=" in line:
            try:
                # Extract value after "SamplingRate="
                after_eq = line.split("SamplingRate=")[1].strip()
                val = after_eq.split()[0].replace("Hz", "")
                header.sampling_rate = float(val)
            except (ValueError, IndexError):
                pass

        elif "SampleBlockSize=" in line:
            try:
                after_eq = line.split("SampleBlockSize=")[1].strip()
                val = after_eq.split()[0]
                header.sample_block_size = int(val)
            except (ValueError, IndexError):
                pass

        # State section detection
        elif "State Vector Definition" in line:
            in_state_section = True
            continue

        # Another section starts — end state parsing
        elif line.startswith("["):
            in_state_section = False

        # Parse state definitions
        # Format: <name> <length> <default> <byteOffset> <bitOffset>
        elif in_state_section:
            parts = line.split()
            if len(parts) >= 5:
                try:
                    state = BCI2000State(
                        name=parts[0],
                        length=int(parts[1]),
                        default_value=int(parts[2]),
                        byte_offset=int(parts[3]),
                        start_bit=int(parts[4]),
                    )
                    header.states[state.name] = state
                except (ValueError, IndexError):
                    continue

    return header


def parse_dat_file(filepath: str | Path) -> BCI2000Run:
    """
    Parse a BCI2000 .dat file into signal data and state vectors.

    Args:
        filepath: Path to the .dat file

    Returns:
        BCI2000Run with parsed signal and state data
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"BCI2000 .dat file not found: {filepath}")

    with open(filepath, "rb") as f:
        raw = f.read()

    # Find header boundary (ends with \r\n\r\n)
    header_end = raw.find(b"\r\n\r\n")
    if header_end == -1:
        # Try Unix line endings
        header_end = raw.find(b"\n\n")
        header_sep_len = 2
    else:
        header_sep_len = 4

    if header_end == -1:
        raise ValueError("Could not find header boundary in .dat file")

    raw_header = raw[:header_end].decode("ascii", errors="replace")
    header = parse_header(raw_header)

    # Override header_len if parsed from header text
    data_start = header.header_len if header.header_len > 0 else header_end + header_sep_len

    # Parse binary data
    data_bytes = raw[data_start:]
    fmt_char, sample_size = DATA_FORMATS.get(header.data_format, ("h", 2))

    if fmt_char is None:
        raise ValueError(f"Unsupported data format: {header.data_format}")

    # ------------------------------------------------------------------
    # BCI2000 per-sample data layout (interleaved):
    #   For each sample within a block the file stores:
    #       n_channels * sample_size  bytes of signal
    #       statevector_len           bytes of state vector
    # ------------------------------------------------------------------
    sample_signal_bytes = header.source_ch * sample_size
    sample_total_bytes = sample_signal_bytes + header.statevector_len
    block_total = sample_total_bytes * header.sample_block_size

    if block_total == 0:
        raise ValueError("Computed block size is zero — invalid header parameters")

    n_blocks = len(data_bytes) // block_total
    if n_blocks == 0:
        logger.warning("No complete data blocks found in file")
        return BCI2000Run(
            filepath=filepath,
            header=header,
            signal=np.zeros((header.source_ch, 0)),
            n_blocks=0,
        )

    total_samples = n_blocks * header.sample_block_size
    signal = np.zeros((header.source_ch, total_samples))
    state_data = {name: np.zeros(total_samples, dtype=np.int64) for name in header.states}

    sample_fmt = f"<{header.source_ch}{fmt_char}"

    for block_idx in range(n_blocks):
        block_offset = block_idx * block_total
        sample_start = block_idx * header.sample_block_size

        for s_idx in range(header.sample_block_size):
            offset = block_offset + s_idx * sample_total_bytes
            sample_idx = sample_start + s_idx

            # Read signal data for this sample (all channels)
            sig_bytes = data_bytes[offset:offset + sample_signal_bytes]
            try:
                values = struct.unpack(sample_fmt, sig_bytes)
            except struct.error as e:
                logger.warning(f"Block {block_idx} sample {s_idx}: signal unpack error: {e}")
                continue
            signal[:, sample_idx] = values

            # Read state vector for this sample
            sv_bytes = data_bytes[offset + sample_signal_bytes:offset + sample_total_bytes]
            for state_name, state_def in header.states.items():
                state_data[state_name][sample_idx] = state_def.extract(sv_bytes)

    run = BCI2000Run(
        filepath=filepath,
        header=header,
        signal=signal,
        states_per_sample=state_data,
        n_blocks=n_blocks,
    )

    logger.info(
        f"Parsed {filepath.name}: {n_blocks} blocks, "
        f"{total_samples} samples, {run.duration_sec:.1f}s, "
        f"{header.source_ch} channels @ {header.sampling_rate} Hz"
    )
    return run


def compute_latency_stats(run: BCI2000Run) -> dict:
    """
    Compute latency statistics from a BCI2000 run.

    Primary metric: inter-block interval from SourceTime (with 16-bit
    wrap handling).  This represents BCI2000's real-time processing
    cadence — the time between consecutive data block acquisitions.

    Returns:
        Dictionary with latency statistics
    """
    stats = {"filepath": str(run.filepath), "n_blocks": run.n_blocks}

    expected_ms = run.header.sample_block_size / run.header.sampling_rate * 1000
    stats["expected_block_interval_ms"] = float(expected_ms)

    # Primary metric: inter-block intervals (16-bit wrap-safe)
    intervals = run.get_block_interval_ms()
    if intervals is not None:
        valid = intervals[~np.isnan(intervals)]
        if len(valid) > 0:
            stats["block_interval"] = {
                "mean_ms": float(np.mean(valid)),
                "median_ms": float(np.median(valid)),
                "std_ms": float(np.std(valid)),
                "min_ms": float(np.min(valid)),
                "max_ms": float(np.max(valid)),
                "p95_ms": float(np.percentile(valid, 95)),
                "p99_ms": float(np.percentile(valid, 99)),
                "expected_ms": float(expected_ms),
                "n_intervals": int(len(valid)),
                "jitter_ms": float(np.std(valid)),
            }

    # Raw roundtrip values (16-bit timestamps — kept for reference)
    roundtrip = run.get_roundtrip_ms()
    if roundtrip is not None and len(roundtrip) > 0:
        rt_valid = roundtrip[roundtrip > 0]
        if len(rt_valid) > 0:
            stats["roundtrip_raw"] = {
                "note": "Raw 16-bit state values (timestamps, not latencies)",
                "mean": float(np.mean(rt_valid)),
                "median": float(np.median(rt_valid)),
                "n_valid_blocks": int(len(rt_valid)),
            }

    return stats


def generate_synthetic_bci2000_dat(
    filepath: str | Path,
    n_channels: int = 16,
    n_blocks: int = 500,
    sample_block_size: int = 32,
    sampling_rate: float = 256.0,
) -> Path:
    """
    Generate a synthetic BCI2000 .dat file for testing.

    Creates a valid .dat file with realistic header structure and
    state variables (Roundtrip, SourceTime, StimulusCode, ResultCode).
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Build header
    states = {
        "Roundtrip": (16, 0, 0, 0),     # 16 bits, byte 0, bit 0
        "SourceTime": (16, 0, 2, 0),    # 16 bits, byte 2, bit 0
        "StimulusCode": (8, 0, 4, 0),   # 8 bits, byte 4, bit 0
        "Feedback": (1, 0, 5, 0),       # 1 bit, byte 5, bit 0
        "ResultCode": (8, 0, 5, 1),     # 8 bits, byte 5, bit 1
    }
    statevector_len = 7  # bytes

    # BCI2000 real files pack core fields on the first line
    header_lines = [
        f"BCI2000V= 1.1 HeaderLen= {{HEADER_LEN}} SourceCh= {n_channels} "
        f"StatevectorLen= {statevector_len} DataFormat= int16",
        f"[ State Vector Definition ]",
    ]

    for name, (length, default, byte_off, bit_off) in states.items():
        header_lines.append(f"{name} {length} {default} {byte_off} {bit_off}")

    header_lines += [
        f"[ Parameter Definition ]",
        f"Source:Signal%20Properties int SamplingRate= {int(sampling_rate)} "
        f"{int(sampling_rate)} 1 % // sampling rate in Hz",
        f"Source:Signal%20Properties int SampleBlockSize= {sample_block_size} "
        f"{sample_block_size} 1 % // samples per block",
    ]

    # BCI2000 header ends with \r\n\r\n — no empty lines allowed within the header
    # as \r\n\r\n is the header terminator
    header_text = "\r\n".join(header_lines) + "\r\n\r\n"
    # Calculate actual header length: substitute with a guess, then adjust
    for guess in range(380, 520):
        candidate = header_text.replace("{HEADER_LEN}", str(guess))
        if len(candidate.encode("ascii")) == guess:
            header_text = candidate
            break
    else:
        # Fallback: just use the length of the template with placeholder replaced
        candidate = header_text.replace("{HEADER_LEN}", "999")
        header_text = header_text.replace("{HEADER_LEN}", str(len(candidate.encode("ascii"))))
    header_bytes = header_text.encode("ascii")

    # Generate synthetic data — per-sample interleaved format:
    # For each sample: (n_channels * sample_size) signal bytes + statevector_len state bytes
    rng = np.random.RandomState(42)
    data_chunks = []

    current_time = 0
    for block_idx in range(n_blocks):
        # Signal data: sinusoidal + noise (int16)
        t = np.arange(sample_block_size) / sampling_rate + block_idx * sample_block_size / sampling_rate
        signal_block = np.zeros((n_channels, sample_block_size), dtype=np.int16)
        for ch in range(n_channels):
            sig = 100 * np.sin(2 * np.pi * (10 + ch * 0.5) * t)
            sig += rng.randn(sample_block_size) * 20
            signal_block[ch] = sig.astype(np.int16)

        # State vector values (constant within a block)
        roundtrip_val = int(rng.uniform(15, 45))  # 15-45 ms typical
        current_time += int(sample_block_size / sampling_rate * 1000)
        stim_code = rng.choice([0, 1, 2])
        feedback = 1 if stim_code > 0 else 0
        result_code = rng.choice([0, 1]) if stim_code > 0 else 0

        for s_idx in range(sample_block_size):
            # Write signal for this sample (all channels)
            data_chunks.append(signal_block[:, s_idx].tobytes())

            # Write state vector for this sample
            sv = bytearray(statevector_len)
            # Roundtrip (16 bits at byte 0)
            sv[0] = roundtrip_val & 0xFF
            sv[1] = (roundtrip_val >> 8) & 0xFF
            # SourceTime (16 bits at byte 2)
            sv[2] = current_time & 0xFF
            sv[3] = (current_time >> 8) & 0xFF
            # StimulusCode (8 bits at byte 4)
            sv[4] = stim_code & 0xFF
            # Feedback (1 bit at byte 5, bit 0) + ResultCode (8 bits at byte 5, bit 1)
            sv[5] = (feedback & 0x01) | ((result_code & 0xFF) << 1)
            data_chunks.append(bytes(sv))

    with open(filepath, "wb") as f:
        f.write(header_bytes)
        for chunk in data_chunks:
            f.write(chunk)

    logger.info(f"Generated synthetic BCI2000 .dat: {filepath} ({n_blocks} blocks)")
    return filepath
