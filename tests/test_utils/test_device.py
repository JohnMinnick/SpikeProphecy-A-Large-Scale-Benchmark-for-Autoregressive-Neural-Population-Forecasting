"""
Tests for src/utils/device.py

Tests device resolution, GPU info logging, and helper functions.
Uses known-answer tests with deterministic inputs.
"""

import logging

import pytest
import torch

from src.utils.device import (
    get_default_num_workers,
    get_num_gpus,
    log_device_info,
    resolve_device,
)


class TestResolveDevice:
    """Tests for resolve_device()."""

    def test_auto_returns_valid_device(self):
        """Auto should resolve to either 'cuda' or 'cpu'."""
        device = resolve_device("auto")
        assert device.type in ("cuda", "cpu")

    def test_cpu_returns_cpu(self):
        """Explicit 'cpu' should always return cpu device."""
        device = resolve_device("cpu")
        assert device.type == "cpu"

    def test_cpu_case_insensitive(self):
        """Device strings should be case-insensitive."""
        device = resolve_device("CPU")
        assert device.type == "cpu"

    def test_cpu_with_whitespace(self):
        """Leading/trailing whitespace should be stripped."""
        device = resolve_device("  cpu  ")
        assert device.type == "cpu"

    def test_invalid_device_raises_valueerror(self):
        """Unknown device string should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown device config"):
            resolve_device("tpu")

    def test_cuda_without_gpu_raises_runtimeerror(self):
        """Requesting CUDA when unavailable should raise RuntimeError."""
        if torch.cuda.is_available():
            pytest.skip("CUDA is available, can't test unavailability")
        with pytest.raises(RuntimeError, match="CUDA is not available"):
            resolve_device("cuda")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA GPU")
    def test_cuda_returns_cuda_device(self):
        """When CUDA is available, 'cuda' should resolve to cuda device."""
        device = resolve_device("cuda")
        assert device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA GPU")
    def test_auto_prefers_cuda(self):
        """When CUDA is available, 'auto' should resolve to cuda."""
        device = resolve_device("auto")
        assert device.type == "cuda"


class TestLogDeviceInfo:
    """Tests for log_device_info()."""

    def test_logs_cpu_info(self, caplog):
        """CPU device should log 'Running on CPU'."""
        device = torch.device("cpu")
        with caplog.at_level(logging.INFO):
            log_device_info(device)
        assert "Running on CPU" in caplog.text

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA GPU")
    def test_logs_gpu_info(self, caplog):
        """CUDA device should log GPU name and VRAM."""
        device = torch.device("cuda")
        with caplog.at_level(logging.INFO):
            log_device_info(device)
        assert "GPU 0:" in caplog.text
        assert "GB VRAM" in caplog.text
        assert "CUDA version:" in caplog.text


class TestGetNumGpus:
    """Tests for get_num_gpus()."""

    def test_returns_non_negative_int(self):
        """GPU count should be a non-negative integer."""
        count = get_num_gpus()
        assert isinstance(count, int)
        assert count >= 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA GPU")
    def test_at_least_one_when_cuda_available(self):
        """When CUDA is available, should report at least 1 GPU."""
        assert get_num_gpus() >= 1


class TestGetDefaultNumWorkers:
    """Tests for get_default_num_workers()."""

    def test_override_returns_override(self):
        """When override is provided, it should be returned directly."""
        device = torch.device("cpu")
        assert get_default_num_workers(device, override=8) == 8

    def test_cpu_default_is_zero(self):
        """CPU should default to 0 workers."""
        device = torch.device("cpu")
        assert get_default_num_workers(device) == 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA GPU")
    def test_cuda_default_is_four(self):
        """CUDA should default to 4 workers."""
        device = torch.device("cuda")
        assert get_default_num_workers(device) == 4
