"""
Tests for Mixture-of-Experts output layer.
"""

import numpy as np
import pytest
import torch

from src.models.moe_output import MoEOutputLayer


class TestMoEOutputLayer:
    """Tests for MoE output with Fano-conditioned routing."""

    def test_construction_no_fano(self):
        """MoE initializes without Fano factors."""
        moe = MoEOutputLayer(hidden_size=64, output_size=100)
        assert len(moe.experts) == 3
        assert moe.fano_prior.shape == (100, 3)

    def test_construction_with_fano(self):
        """MoE initializes with Fano factors."""
        fano = np.array([0.5] * 30 + [1.2] * 40 + [2.5] * 30)
        moe = MoEOutputLayer(hidden_size=64, output_size=100,
                              fano_factors=fano)
        # Sub-Poisson neurons should have high weight on expert 0
        assert moe.fano_prior[0, 0] == pytest.approx(0.8)
        # Super-Poisson on expert 2
        assert moe.fano_prior[99, 2] == pytest.approx(0.8)

    def test_output_shape(self):
        """Output shape is (batch, M)."""
        moe = MoEOutputLayer(hidden_size=64, output_size=100)
        h = torch.randn(4, 64)
        output = moe(h)
        assert output.shape == (4, 100)

    def test_output_positive(self):
        """Output rates are positive (softplus applied)."""
        moe = MoEOutputLayer(hidden_size=64, output_size=100)
        h = torch.randn(4, 64)
        output = moe(h)
        assert torch.all(output > 0)

    def test_gradient_flows(self):
        """Gradients flow through all experts and router."""
        moe = MoEOutputLayer(hidden_size=64, output_size=100)
        h = torch.randn(4, 64, requires_grad=True)
        output = moe(h)
        loss = output.sum()
        loss.backward()
        assert h.grad is not None

        # All expert weights should have gradients
        for expert in moe.experts:
            assert expert.weight.grad is not None

    def test_static_routing(self):
        """Static routing uses Fano prior directly."""
        fano = np.array([0.5] * 50 + [2.0] * 50)
        moe = MoEOutputLayer(
            hidden_size=64, output_size=100,
            fano_factors=fano, router_type="static",
        )
        h = torch.randn(4, 64)
        output = moe(h)
        assert output.shape == (4, 100)

    def test_routing_stats(self):
        """get_routing_stats returns valid statistics."""
        fano = np.array([0.5] * 30 + [1.2] * 40 + [2.5] * 30)
        moe = MoEOutputLayer(
            hidden_size=64, output_size=100,
            fano_factors=fano,
        )
        h = torch.randn(4, 64)
        stats = moe.get_routing_stats(h)

        assert "routing_entropy" in stats
        assert "routing_weight_sub_poisson" in stats
        assert "routing_weight_near_poisson" in stats
        assert "routing_weight_super_poisson" in stats

        # Weights should sum to ~1
        total = sum(
            stats[f"routing_weight_{name}"]
            for name in moe.expert_names
        )
        assert total == pytest.approx(1.0, abs=0.01)

    def test_different_n_experts(self):
        """Works with different number of experts."""
        moe = MoEOutputLayer(hidden_size=64, output_size=100, n_experts=2)
        assert len(moe.experts) == 2
        h = torch.randn(4, 64)
        output = moe(h)
        assert output.shape == (4, 100)

    def test_fano_prior_matches_populations(self):
        """Fano prior correctly routes each population."""
        fano = np.array([0.3] * 20 + [1.1] * 60 + [3.0] * 20)
        moe = MoEOutputLayer(
            hidden_size=64, output_size=100,
            fano_factors=fano,
        )
        # First 20 neurons (sub-Poisson): expert 0 dominant
        assert moe.fano_prior[:20, 0].mean() > 0.7
        # Middle 60 (near): expert 1 dominant
        assert moe.fano_prior[20:80, 1].mean() > 0.7
        # Last 20 (super): expert 2 dominant
        assert moe.fano_prior[80:100, 2].mean() > 0.7
