"""
Mamba classifier for SHD (Spiking Heidelberg Digits) benchmark.

Reuses the MambaBlock stack from mamba_baseline.py but replaces the
forecasting output head with a classification head. Supports the same
instrumentation hooks (enable_instrumentation / get_internal_signals)
for mechanism-aligned distillation.

This is the "teacher" model for SHD experiments.
"""

import logging
from typing import Dict, Any, Optional

import torch
import torch.nn as nn

from src.models.mamba_baseline import (
    MambaBlock, InstrumentedMambaBlock, _MAMBA_AVAILABLE,
)

logger = logging.getLogger(__name__)


class MambaClassifier(nn.Module):
    """
    Mamba-based sequence classifier for SHD and similar tasks.

    Architecture:
        1. Input projection: Linear(input_size, d_model)
        2. N stacked MambaBlocks (SSM layers with residual connections)
        3. Classification head: LayerNorm → mean-pool → Linear(d_model, n_classes)

    Supports instrumentation for mechanism-aligned distillation via
    enable_instrumentation() and get_internal_signals().

    Args:
        input_size: Number of input channels (700 for SHD).
        d_model: Model embedding dimension.
        n_layers: Number of stacked Mamba blocks.
        n_classes: Number of output classes (20 for SHD).
        d_state: SSM state dimension.
        d_conv: Local convolution width.
        expand: Expansion factor for inner dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_size: int = 700,
        d_model: int = 128,
        n_layers: int = 2,
        n_classes: int = 20,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_classes = n_classes

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model),
        )

        # Stacked Mamba blocks (uses project's MambaBlock from mamba_baseline.py)
        if not _MAMBA_AVAILABLE:
            raise RuntimeError(
                "mamba-ssm is required for MambaClassifier. "
                "Install with: pip install mamba-ssm"
            )
        self.mamba_blocks = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Classification head: LayerNorm → mean pool → Linear
        self.head_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, n_classes)

        logger.info(
            "MambaClassifier: input=%d, d_model=%d, layers=%d, "
            "classes=%d, params=%d",
            input_size, d_model, n_layers, n_classes,
            sum(p.numel() for p in self.parameters()),
        )

    def forward(
        self, x: torch.Tensor, return_features: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input spike counts (batch, T, input_size).
            return_features: If True, also return hidden features.

        Returns:
            logits: (batch, n_classes) classification logits.
        """
        # Input projection: (batch, T, input_size) → (batch, T, d_model)
        h = self.input_proj(x)

        # Mamba blocks (pre-norm residual is built into each block)
        for block in self.mamba_blocks:
            h = block(h)

        h = self.dropout(h)

        # Mean-pool over time dimension: (batch, T, d_model) → (batch, d_model)
        h_pooled = h.mean(dim=1)

        # Classification head
        h_pooled = self.head_norm(h_pooled)
        logits = self.classifier(h_pooled)

        if return_features:
            return logits, h_pooled
        return logits

    def enable_instrumentation(self) -> None:
        """
        Enable SSM signal instrumentation for mechanism alignment.

        Wraps each MambaBlock with InstrumentedMambaBlock to capture
        delta, B, C signals from x_proj.
        """
        instrumented = nn.ModuleList()
        for block in self.mamba_blocks:
            if isinstance(block, InstrumentedMambaBlock):
                instrumented.append(block)
            else:
                instrumented.append(InstrumentedMambaBlock(block))
        self.mamba_blocks = instrumented
        logger.info(
            "Mamba instrumentation enabled: %d blocks wrapped",
            len(instrumented),
        )

    def disable_instrumentation(self) -> None:
        """Remove instrumentation hooks and unwrap blocks."""
        unwrapped = nn.ModuleList()
        for block in self.mamba_blocks:
            if isinstance(block, InstrumentedMambaBlock):
                block.remove_hooks()
                unwrapped.append(block.block)
            else:
                unwrapped.append(block)
        self.mamba_blocks = unwrapped

    def get_internal_signals(self) -> Dict[str, torch.Tensor]:
        """
        Get captured SSM signals from the last forward pass.

        Returns signals from the LAST instrumented block.

        Returns:
            Dict with 'delta', 'B', 'C' (or None if not instrumented).
        """
        result = {"delta": None, "B": None, "C": None}

        last_block = None
        for block in reversed(list(self.mamba_blocks)):
            if isinstance(block, InstrumentedMambaBlock):
                last_block = block
                break

        if last_block is None:
            return result

        signals = last_block.get_signals()
        bc = signals.get("bc")
        if bc is not None:
            mamba_layer = last_block.block.mamba
            dt_rank = getattr(mamba_layer, 'dt_rank', 0)
            d_state = getattr(mamba_layer, 'd_state', 16)

            if bc.shape[-1] >= dt_rank + 2 * d_state:
                result["delta"] = bc[..., :dt_rank]
                result["B"] = bc[..., dt_rank:dt_rank + d_state]
                result["C"] = bc[..., dt_rank + d_state:]

        return result

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MambaClassifier":
        """Create from config dict."""
        model_cfg = config.get("model", {})
        return cls(
            input_size=model_cfg.get("input_size", 700),
            d_model=model_cfg.get("d_model", 128),
            n_layers=model_cfg.get("n_layers", 2),
            n_classes=model_cfg.get("n_classes", 20),
            d_state=model_cfg.get("d_state", 16),
            d_conv=model_cfg.get("d_conv", 4),
            expand=model_cfg.get("expand", 2),
            dropout=model_cfg.get("dropout", 0.2),
        )
