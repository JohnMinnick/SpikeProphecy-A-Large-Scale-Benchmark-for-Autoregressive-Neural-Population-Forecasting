"""
Tests for Conway-Maxwell-Poisson (CMP) loss function.

Tests cover:
- CMP(nu=1) equivalence to full Poisson NLL
- LearnableDispersion initialization at nu=1
- Gradient flow through rates and dispersion
- Impact of dispersion parameter on loss
- Per-element output shapes
"""

import torch
import pytest
import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.train.cmp_loss import (
    cmp_nll,
    cmp_nll_per_element,
    LearnableDispersion,
    _log_z_cmp,
)


class TestLogZCMP:
    """Tests for the normalizing constant Z(lambda, nu)."""

    def test_z_poisson_matches(self):
        """At nu=1, Z = exp(lambda) so log Z = lambda = exp(log_lambda)."""
        # Use actual lambda values, convert to log space for the function
        lambdas = torch.tensor([1.0, 2.0, 0.5])
        log_lambda = torch.log(lambdas)
        nu = torch.ones_like(log_lambda)
        log_z = _log_z_cmp(log_lambda, nu)

        # Z(lambda, nu=1) = exp(lambda), so log Z = lambda
        torch.testing.assert_close(
            log_z, lambdas, atol=0.01, rtol=0.01,
        )

    def test_z_positive(self):
        """Z should always be positive (log Z finite)."""
        log_lambda = torch.randn(10)
        nu = torch.ones(10).abs() + 0.1  # nu > 0
        log_z = _log_z_cmp(log_lambda, nu)
        assert torch.all(torch.isfinite(log_z)), "log Z should be finite"

    def test_z_shape(self):
        """Output shape should match input shape."""
        log_lambda = torch.randn(3, 5)
        nu = torch.ones(3, 5)
        log_z = _log_z_cmp(log_lambda, nu)
        assert log_z.shape == (3, 5)


class TestCMPNLL:
    """Tests for CMP negative log-likelihood."""

    def test_matches_poisson_at_nu_1(self):
        """CMP NLL at nu=1 should equal full Poisson NLL (with lgamma)."""
        rates = torch.tensor([[2.0, 3.0, 0.5]])
        counts = torch.tensor([[1.0, 4.0, 0.0]])
        nu = torch.ones(3)

        cmp_loss = cmp_nll(rates, counts, nu)

        # Full Poisson NLL: lambda - y*log(lambda) + log(y!)
        eps = 1e-8
        full_poisson = (
            rates - counts * torch.log(rates + eps) + torch.lgamma(counts + 1)
        ).mean()

        torch.testing.assert_close(
            cmp_loss, full_poisson, atol=0.01, rtol=0.01,
        )

    def test_gradient_flows(self):
        """Gradients should flow through both rates and nu."""
        rates = torch.tensor([[2.0, 3.0]], requires_grad=True)
        counts = torch.tensor([[1.0, 4.0]])
        nu = torch.tensor([0.8, 1.5], requires_grad=True)

        loss = cmp_nll(rates, counts, nu)
        loss.backward()

        assert rates.grad is not None, "Rate gradients required"
        assert nu.grad is not None, "Nu gradients required"
        assert torch.all(torch.isfinite(rates.grad)), "Finite rate grads"
        assert torch.all(torch.isfinite(nu.grad)), "Finite nu grads"

    def test_dispersion_modulates_loss(self):
        """Different nu values should give different loss."""
        rates = torch.tensor([[5.0]])
        counts = torch.tensor([[5.0]])

        loss_poisson = cmp_nll(rates, counts, torch.ones(1))
        loss_subp = cmp_nll(rates, counts, torch.tensor([2.0]))

        # Sub-Poisson nu should give different loss
        assert abs(loss_poisson.item() - loss_subp.item()) > 0.01

    def test_per_element_shape(self):
        """Per-element output should match input shape."""
        batch_rates = torch.randn(32, 64).exp()
        batch_counts = torch.randint(0, 10, (32, 64)).float()
        nu = torch.ones(64)

        per_elem = cmp_nll_per_element(batch_rates, batch_counts, nu)
        assert per_elem.shape == (32, 64)


class TestLearnableDispersion:
    """Tests for the learnable dispersion parameter module."""

    def test_initial_value(self):
        """Should initialize near nu=1 (Poisson)."""
        disp = LearnableDispersion(100)
        nu = disp()
        assert abs(nu.mean().item() - 1.0) < 0.05

    def test_positive(self):
        """Nu should always be positive."""
        disp = LearnableDispersion(50)
        nu = disp()
        assert torch.all(nu > 0)

    def test_gradient_flows(self):
        """Gradients should flow through the dispersion parameter."""
        disp = LearnableDispersion(10)
        nu = disp()
        loss = nu.sum()
        loss.backward()
        assert disp.raw_nu.grad is not None

    def test_stats(self):
        """get_stats should return expected keys."""
        disp = LearnableDispersion(20)
        stats = disp.get_stats()
        assert "nu_mean" in stats
        assert "nu_median" in stats
        assert "nu_min" in stats
        assert "nu_max" in stats
        assert "nu_frac_sub_poisson" in stats
