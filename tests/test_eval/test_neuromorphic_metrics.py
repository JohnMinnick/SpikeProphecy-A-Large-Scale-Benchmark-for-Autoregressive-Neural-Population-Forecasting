"""
Tests for neuromorphic evaluation metrics.

Validates SynOps computation, spike sparsity, energy estimates,
and ANN FLOPs comparison.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.eval.neuromorphic_metrics import (
    compute_synops,
    spike_sparsity,
    ann_flops_estimate,
    compute_neuromorphic_metrics,
    NeuromorphicMetrics,
    ENERGY_PER_SYNOP,
)


class TestSpikeSparsity:
    """Tests for spike_sparsity()."""

    def test_all_silent(self):
        """All-zero spikes → sparsity = 1.0."""
        spikes = torch.zeros(4, 50, 128)
        assert spike_sparsity(spikes) == pytest.approx(1.0)

    def test_all_active(self):
        """All-ones spikes → sparsity = 0.0."""
        spikes = torch.ones(4, 50, 128)
        assert spike_sparsity(spikes) == pytest.approx(0.0)

    def test_half_active(self):
        """50% active → sparsity ≈ 0.5."""
        spikes = torch.zeros(4, 50, 128)
        spikes[:, :, :64] = 1.0  # Half the neurons active
        assert spike_sparsity(spikes) == pytest.approx(0.5, abs=0.01)

    def test_sparse_snn_typical(self):
        """Typical SNN sparsity is > 0.9."""
        # 5% firing probability
        spikes = (torch.rand(4, 100, 256) < 0.05).float()
        sparsity = spike_sparsity(spikes)
        assert sparsity > 0.9


class TestComputeSynops:
    """Tests for compute_synops()."""

    def test_zero_spikes_zero_synops(self):
        """No spikes → zero SynOps."""
        spikes = torch.zeros(4, 50, 128)
        synops = compute_synops(spikes, [(128, 256)])
        assert synops == pytest.approx(0.0)

    def test_synops_scales_with_activity(self):
        """More activity → more SynOps."""
        low_spikes = (torch.rand(4, 50, 128) < 0.01).float()
        high_spikes = (torch.rand(4, 50, 128) < 0.1).float()

        synops_low = compute_synops(low_spikes, [(128, 256)])
        synops_high = compute_synops(high_spikes, [(128, 256)])
        assert synops_high > synops_low

    def test_synops_scales_with_fan_out(self):
        """Larger fan-out → more SynOps."""
        spikes = (torch.rand(4, 50, 128) < 0.05).float()
        synops_small = compute_synops(spikes, [(128, 64)])
        synops_large = compute_synops(spikes, [(128, 512)])
        assert synops_large > synops_small


class TestAnnFlopsEstimate:
    """Tests for ann_flops_estimate()."""

    def test_lstm_flops_positive(self):
        """LSTM FLOPs estimate > 0."""
        flops = ann_flops_estimate(1240, 256, 2, 1240, "lstm")
        assert flops > 0

    def test_mamba_flops_positive(self):
        """Mamba FLOPs estimate > 0."""
        flops = ann_flops_estimate(1240, 256, 2, 1240, "mamba")
        assert flops > 0

    def test_lru_flops_positive(self):
        """LRU FLOPs estimate > 0."""
        flops = ann_flops_estimate(1240, 256, 2, 1240, "lru")
        assert flops > 0

    def test_lstm_more_flops_than_lru(self):
        """LSTM has more FLOPs than LRU (4 gates vs diagonal)."""
        lstm_flops = ann_flops_estimate(1240, 256, 2, 1240, "lstm")
        lru_flops = ann_flops_estimate(1240, 256, 2, 1240, "lru")
        assert lstm_flops > lru_flops


class TestComputeNeuromorphicMetrics:
    """Tests for full neuromorphic metrics computation."""

    def _make_simple_model(self):
        """Create a simple model with weight matrices."""
        model = nn.Sequential(
            nn.Linear(128, 256),
            nn.Linear(256, 128),
        )
        return model

    def test_returns_dataclass(self):
        """Returns NeuromorphicMetrics dataclass."""
        model = self._make_simple_model()
        spikes = (torch.rand(4, 50, 128) < 0.05).float()
        metrics = compute_neuromorphic_metrics(spikes, model)
        assert isinstance(metrics, NeuromorphicMetrics)

    def test_sparsity_in_range(self):
        """Sparsity is between 0 and 1."""
        model = self._make_simple_model()
        spikes = (torch.rand(4, 50, 128) < 0.05).float()
        metrics = compute_neuromorphic_metrics(spikes, model)
        assert 0 <= metrics.spike_sparsity <= 1

    def test_energy_positive(self):
        """Energy is positive for non-zero spikes."""
        model = self._make_simple_model()
        spikes = (torch.rand(4, 50, 128) < 0.05).float()
        metrics = compute_neuromorphic_metrics(spikes, model)
        assert metrics.energy_per_timestep_pj >= 0
        assert metrics.energy_per_inference_pj >= 0

    def test_compute_ratio_positive(self):
        """Compute ratio is positive."""
        model = self._make_simple_model()
        spikes = (torch.rand(4, 50, 128) < 0.05).float()
        metrics = compute_neuromorphic_metrics(spikes, model)
        assert metrics.compute_ratio > 0

    def test_to_dict(self):
        """to_dict() returns a serializable dict."""
        model = self._make_simple_model()
        spikes = (torch.rand(4, 50, 128) < 0.05).float()
        metrics = compute_neuromorphic_metrics(spikes, model)
        d = metrics.to_dict()
        assert isinstance(d, dict)
        assert "synops_per_timestep" in d
        assert "spike_sparsity" in d

    def test_summary_string(self):
        """summary() returns a formatted string."""
        model = self._make_simple_model()
        spikes = (torch.rand(4, 50, 128) < 0.05).float()
        metrics = compute_neuromorphic_metrics(spikes, model)
        s = metrics.summary()
        assert "Spike sparsity" in s
        assert "SynOps" in s

    def test_hardware_parameter(self):
        """Hardware parameter is stored."""
        model = self._make_simple_model()
        spikes = (torch.rand(4, 50, 128) < 0.05).float()
        metrics = compute_neuromorphic_metrics(
            spikes, model, hardware="truenorth",
        )
        assert metrics.hardware == "truenorth"
