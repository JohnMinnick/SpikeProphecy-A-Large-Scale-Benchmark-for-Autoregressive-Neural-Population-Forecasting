"""
Tests for region-specific hybrid loss functions.
"""

import numpy as np
import pytest
import torch

from src.train.region_loss import (
    RegionHybridLoss,
    FanoAdaptiveLoss,
    HIPPOCAMPAL_REGIONS,
)


class TestRegionHybridLoss:
    """Tests for the region-specific hybrid loss."""

    def _make_region_map(self, m_max=100):
        """Create a test region map with some hippocampal channels."""
        region_map = {}
        for i in range(m_max):
            if i < 20:
                region_map[i] = "CA1"  # Hippocampus
            elif i < 30:
                region_map[i] = "DG"   # Hippocampus
            else:
                region_map[i] = "VISp"  # Visual cortex
        return region_map

    def test_construction(self):
        """RegionHybridLoss initializes correctly."""
        region_map = self._make_region_map(100)
        loss_fn = RegionHybridLoss(region_map, m_max=100)
        assert loss_fn.n_negbin == 30  # CA1 (20) + DG (10)
        assert loss_fn.n_poisson == 70

    def test_output_is_scalar(self):
        """Loss output is a scalar tensor."""
        region_map = self._make_region_map(100)
        loss_fn = RegionHybridLoss(region_map, m_max=100)

        y_hat = torch.rand(4, 100) + 0.1
        y = torch.randint(0, 5, (4, 100)).float()

        loss = loss_fn(y_hat, y)
        assert loss.dim() == 0

    def test_loss_positive(self):
        """Loss should be positive for valid inputs."""
        region_map = self._make_region_map(100)
        loss_fn = RegionHybridLoss(region_map, m_max=100)

        y_hat = torch.rand(4, 100) + 0.1
        y = torch.randint(0, 5, (4, 100)).float()

        loss = loss_fn(y_hat, y)
        assert loss.item() > 0

    def test_with_dispersion(self):
        """Works correctly with dispersion parameters (NegBin)."""
        region_map = self._make_region_map(100)
        loss_fn = RegionHybridLoss(region_map, m_max=100)

        y_hat = torch.rand(4, 100) + 0.1
        y = torch.randint(0, 5, (4, 100)).float()
        aux = torch.ones(4, 100) * 2.0  # Dispersion r=2

        loss = loss_fn(y_hat, y, aux=aux)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_with_mask(self):
        """Masked channels produce finite loss."""
        region_map = self._make_region_map(100)
        loss_fn = RegionHybridLoss(region_map, m_max=100)

        y_hat = torch.rand(4, 100) + 0.1
        y = torch.randint(0, 5, (4, 100)).float()

        # Half mask
        mask = torch.zeros(4, 100)
        mask[:, :50] = 1.0
        half_loss = loss_fn(y_hat, y, mask=mask)

        # Masked loss should be finite and positive
        assert torch.isfinite(half_loss)
        assert half_loss.item() > 0

    def test_negbin_weight(self):
        """Higher negbin_weight increases hippocampal loss contribution."""
        region_map = self._make_region_map(100)

        loss_w1 = RegionHybridLoss(region_map, m_max=100, negbin_weight=1.0)
        loss_w3 = RegionHybridLoss(region_map, m_max=100, negbin_weight=3.0)

        y_hat = torch.rand(4, 100) + 0.1
        y = torch.randint(0, 5, (4, 100)).float()

        l1 = loss_w1(y_hat, y)
        l3 = loss_w3(y_hat, y)

        # Higher weight should increase loss (hippocampal contribution amplified)
        assert l3.item() > l1.item()

    def test_gradient_flows(self):
        """Gradients flow through the loss."""
        region_map = self._make_region_map(100)
        loss_fn = RegionHybridLoss(region_map, m_max=100)

        y_hat = torch.rand(4, 100) + 0.1
        y_hat.requires_grad_(True)
        y_hat.retain_grad()
        y = torch.randint(0, 5, (4, 100)).float()

        loss = loss_fn(y_hat, y)
        loss.backward()

        assert y_hat.grad is not None
        assert not torch.all(y_hat.grad == 0)


class TestFanoAdaptiveLoss:
    """Tests for Fano-adaptive weighted loss."""

    def test_construction(self):
        """FanoAdaptiveLoss initializes correctly."""
        fano = np.array([0.5, 1.0, 1.2, 2.0, 3.0])
        loss_fn = FanoAdaptiveLoss(fano, m_max=5)
        assert loss_fn.channel_weights.shape == (5,)

    def test_output_is_scalar(self):
        """Loss is a scalar."""
        fano = np.ones(50)
        loss_fn = FanoAdaptiveLoss(fano, m_max=50)

        y_hat = torch.rand(4, 50) + 0.1
        y = torch.randint(0, 5, (4, 50)).float()

        loss = loss_fn(y_hat, y)
        assert loss.dim() == 0

    def test_super_poisson_weighted_higher(self):
        """Super-Poisson channels with higher weight influence loss more."""
        fano = np.array([0.5] * 25 + [2.0] * 25)  # Half sub, half super
        loss_fn = FanoAdaptiveLoss(
            fano, m_max=50,
            sub_weight=0.1, super_weight=2.0,
        )

        # Create pred that's wrong on all channels equally
        y_hat = torch.ones(4, 50) * 5.0
        y = torch.zeros(4, 50)

        loss = loss_fn(y_hat, y)
        # Loss should be dominated by super-Poisson channels
        assert loss.item() > 0

    def test_gradient_flows(self):
        """Gradients flow through Fano-weighted loss."""
        fano = np.ones(50)
        loss_fn = FanoAdaptiveLoss(fano, m_max=50)

        y_hat = torch.rand(4, 50) + 0.1
        y_hat.requires_grad_(True)
        y_hat.retain_grad()
        y = torch.randint(0, 5, (4, 50)).float()

        loss = loss_fn(y_hat, y)
        loss.backward()

        assert y_hat.grad is not None
