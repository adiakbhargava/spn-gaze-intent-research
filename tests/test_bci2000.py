"""Tests for BCI2000 .dat parser."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.data.bci2000_parser import (
    BCI2000Header,
    BCI2000State,
    generate_synthetic_bci2000_dat,
    parse_dat_file,
    compute_latency_stats,
)


class TestBCI2000State:
    def test_extract_simple(self):
        state = BCI2000State(name="Test", length=8, start_bit=0, byte_offset=0)
        sv = bytes([0x42, 0x00, 0x00])
        assert state.extract(sv) == 0x42

    def test_extract_offset(self):
        state = BCI2000State(name="Test", length=8, start_bit=0, byte_offset=1)
        sv = bytes([0x00, 0xAB, 0x00])
        assert state.extract(sv) == 0xAB

    def test_extract_16bit(self):
        state = BCI2000State(name="Test", length=16, start_bit=0, byte_offset=0)
        sv = bytes([0x34, 0x12, 0x00])  # Little-endian: 0x1234
        assert state.extract(sv) == 0x1234


class TestSyntheticDat:
    def test_generate_and_parse(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.dat"
            generate_synthetic_bci2000_dat(
                filepath,
                n_channels=4,
                n_blocks=50,
                sample_block_size=16,
                sampling_rate=256.0,
            )
            assert filepath.exists()

            run = parse_dat_file(filepath)
            assert run.n_blocks == 50
            assert run.signal.shape[0] == 4  # n_channels
            assert run.signal.shape[1] == 50 * 16  # n_blocks * block_size
            assert run.header.sampling_rate == 256.0

    def test_roundtrip_extraction(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.dat"
            generate_synthetic_bci2000_dat(filepath, n_blocks=100)
            run = parse_dat_file(filepath)
            rt = run.get_roundtrip_ms()
            assert rt is not None
            assert len(rt) == 100
            # Synthetic roundtrip should be 15-45 ms
            valid = rt[rt > 0]
            assert len(valid) > 0
            assert np.min(valid) >= 15
            assert np.max(valid) <= 45

    def test_latency_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.dat"
            generate_synthetic_bci2000_dat(filepath, n_blocks=200)
            run = parse_dat_file(filepath)
            stats = compute_latency_stats(run)
            # Primary metric is block_interval (from SourceTime deltas)
            assert "block_interval" in stats
            assert stats["block_interval"]["mean_ms"] > 0
            assert stats["block_interval"]["p95_ms"] >= stats["block_interval"]["median_ms"]
            # Expected block interval should match sampling config
            assert "expected_block_interval_ms" in stats
