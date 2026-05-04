"""
Tests for GAC-SNN (Gated-Aligned-Coupled Spiking Neural Network).

Tests cover:
- ShortTermPlasticity: shape, range, gradient flow
- DendriticGate: shape, gate initialization, gating behavior
- GacStudentSNN: forward pass, alignment signals, gradient flow
- MechanismAlignmentLoss: all 6 loss components, backward pass
- Ablation: STP-only, dendrite-only, select modes
"""

import torch
import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.gac_snn import (
    ShortTermPlasticity,
    DendriticGate,
    GacStudentSNN,
)
from src.distill.mechanism_loss import (
    MechanismAlignmentLoss,
    SignalProjector,
)


# Fixtures
B, T, M, H = 4, 10, 32, 64


class TestShortTermPlasticity:
    """Tests for STP module."""

    def test_output_shape(self):
        """Output should match input shape."""
        stp = ShortTermPlasticity(H)
        x = torch.randn(B, H)
        u = torch.ones(B, H) * 0.5
        x_res = torch.ones(B, H)
        out, u_new, x_new = stp(x, u, x_res)
        assert out.shape == (B, H)
        assert u_new.shape == (B, H)
        assert x_new.shape == (B, H)

    def test_facilitation_range(self):
        """Facilitation u should be in (0, 1)."""
        stp = ShortTermPlasticity(H)
        x = torch.randn(B, H)
        u = torch.ones(B, H) * 0.5
        x_res = torch.ones(B, H)
        _, u_new, _ = stp(x, u, x_res)
        assert torch.all(u_new >= 0) and torch.all(u_new <= 1)

    def test_gradient_flows(self):
        """Gradients should flow through STP."""
        stp = ShortTermPlasticity(H)
        x = torch.randn(B, H, requires_grad=True)
        u = torch.ones(B, H) * 0.5
        x_res = torch.ones(B, H)
        out, _, _ = stp(x, u, x_res)
        out.sum().backward()
        assert x.grad is not None


class TestDendriticGate:
    """Tests for dendritic branch gating."""

    def test_output_shape(self):
        """Output should match input shape."""
        dend = DendriticGate(H, num_branches=4)
        activity = torch.randn(B, H)
        gated = dend(activity)
        assert gated.shape == (B, H)

    def test_initial_gate_values(self):
        """Gates should initialize near 1.0 (pass-through)."""
        dend = DendriticGate(H, num_branches=4)
        activity = torch.randn(B, H)
        _ = dend(activity)
        # sigmoid(2.0) ≈ 0.88, so gates should be near that
        assert dend._last_gates.mean() > 0.7

    def test_gradient_flows(self):
        """Gradients should flow through dendrite."""
        dend = DendriticGate(H, num_branches=4)
        activity = torch.randn(B, H, requires_grad=True)
        gated = dend(activity)
        gated.sum().backward()
        assert activity.grad is not None


class TestGacStudentSNN:
    """Tests for the full GAC-SNN model."""

    def test_forward_shape(self):
        """Output shapes should match."""
        model = GacStudentSNN(
            input_size=M, hidden_size=H, output_size=M,
        )
        x = torch.randn(B, T, M)
        rates, spikes = model(x)
        assert rates.shape == (B, M)
        assert spikes.shape == (B, T, H)

    def test_alignment_signals_present(self):
        """All alignment signals should be populated after forward."""
        model = GacStudentSNN(
            input_size=M, hidden_size=H, output_size=M,
            enable_stp=True, enable_dendrite=True,
        )
        x = torch.randn(B, T, M)
        model(x)
        signals = model.get_alignment_signals()
        assert signals["betas"] is not None
        assert signals["stp_gains"] is not None
        assert signals["dendrite_gates"] is not None

    def test_all_params_get_gradients(self):
        """All parameters should receive gradients."""
        model = GacStudentSNN(
            input_size=M, hidden_size=H, output_size=M,
        )
        x = torch.randn(B, T, M)
        rates, _ = model(x)
        rates.sum().backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_stp_only(self):
        """STP-only ablation should work."""
        model = GacStudentSNN(
            input_size=M, hidden_size=H, output_size=M,
            enable_stp=True, enable_dendrite=False,
        )
        x = torch.randn(B, T, M)
        rates, _ = model(x)
        signals = model.get_alignment_signals()
        assert signals["stp_gains"] is not None
        assert signals["dendrite_gates"] is None
        assert rates.shape == (B, M)

    def test_dendrite_only(self):
        """Dendrite-only ablation should work."""
        model = GacStudentSNN(
            input_size=M, hidden_size=H, output_size=M,
            enable_stp=False, enable_dendrite=True,
        )
        x = torch.randn(B, T, M)
        rates, _ = model(x)
        signals = model.get_alignment_signals()
        assert signals["stp_gains"] is None
        assert signals["dendrite_gates"] is not None

    def test_from_config(self):
        """Config-based construction should work."""
        config = {
            "model": {
                "hidden_size": 64,
                "num_layers": 2,
                "enable_stp": True,
                "enable_dendrite": True,
            }
        }
        model = GacStudentSNN.from_config(config, input_size=M)
        x = torch.randn(B, T, M)
        rates, spikes = model(x)
        assert rates.shape == (B, M)


class TestMechanismAlignmentLoss:
    """Tests for the mechanism alignment loss."""

    def test_all_components_computed(self):
        """All 6 loss components should be in output."""
        loss_fn = MechanismAlignmentLoss(
            d_delta=512, d_state=16, snn_hidden_size=H,
        )
        student_rates = torch.randn(B, M).exp()
        spikes = torch.zeros(B, T, H)
        gt = torch.randint(0, 5, (B, M)).float()
        teacher_rates = torch.randn(B, M).exp()

        result = loss_fn(
            student_rates, spikes, gt, teacher_rates,
        )
        expected_keys = {
            "loss", "poisson", "distill",
            "tau_align", "stp_align", "dend_align",
            "reg", "distill_weight",
        }
        assert set(result.keys()) == expected_keys

    def test_backward_with_alignment(self):
        """Full backward pass should work with alignment signals."""
        model = GacStudentSNN(
            input_size=M, hidden_size=H, output_size=M,
        )
        loss_fn = MechanismAlignmentLoss(
            d_delta=512, d_state=16, snn_hidden_size=H,
        )
        x = torch.randn(B, T, M)
        rates, spikes = model(x)
        gt = torch.randint(0, 5, (B, M)).float()
        teacher_rates = torch.randn(B, M).exp()

        # Simulate Mamba signals
        mamba_signals = {
            "delta": torch.randn(B * T, 512),
            "B": torch.randn(B * T, 16),
            "C": torch.randn(B * T, 16),
        }
        snn_signals = model.get_alignment_signals()

        result = loss_fn(
            rates, spikes, gt, teacher_rates,
            mamba_signals=mamba_signals,
            snn_signals=snn_signals,
        )
        result["loss"].backward()
        # No error means backward pass works

    def test_zero_alignment_weights(self):
        """Disabling alignment should give zero alignment losses."""
        loss_fn = MechanismAlignmentLoss(
            d_delta=512, d_state=16, snn_hidden_size=H,
            gamma_tau=0.0, gamma_stp=0.0, gamma_dend=0.0,
        )
        result = loss_fn(
            torch.randn(B, M).exp(),
            torch.zeros(B, T, H),
            torch.randint(0, 5, (B, M)).float(),
            torch.randn(B, M).exp(),
        )
        assert result["tau_align"].item() == 0.0
        assert result["stp_align"].item() == 0.0
        assert result["dend_align"].item() == 0.0


class TestSignalProjector:
    """Tests for the signal projector network."""

    def test_output_range(self):
        """Output should be in [0, 1] (sigmoid)."""
        proj = SignalProjector(32, 64)
        x = torch.randn(B, 32)
        out = proj(x)
        assert torch.all(out >= 0) and torch.all(out <= 1)

    def test_output_shape(self):
        """Should map input_dim → output_dim."""
        proj = SignalProjector(16, 128)
        x = torch.randn(B, 16)
        assert proj(x).shape == (B, 128)
