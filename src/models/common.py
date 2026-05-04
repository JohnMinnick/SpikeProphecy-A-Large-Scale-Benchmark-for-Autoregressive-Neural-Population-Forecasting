"""
Shared model components used across all teacher architectures.

Contains architecture-agnostic building blocks and the model factory
function. Extracted from teacher.py to break the coupling where Mamba
(the active teacher) was importing from the LSTM module.

Components:
    - VALID_DISTRIBUTIONS: Tuple of supported output distribution types.
    - PopulationCouplingLayer: Cross-neuron bottleneck MLP with residual.
    - create_teacher_model(): Factory dispatching on config["model"]["architecture"].
"""

import logging
from typing import Any, Dict, Optional

import torch.nn as nn

logger = logging.getLogger(__name__)

# Valid output distribution types for all teacher architectures.
# "poisson" — single rate λ (default)
# "negbin"  — rate λ + dispersion r (overdispersed counts)
# "zip"     — rate λ + zero-inflation gate π (excess zeros)
VALID_DISTRIBUTIONS = ("poisson", "negbin", "zip")


class PopulationCouplingLayer(nn.Module):
    """
    Bottleneck MLP with residual connection for cross-neuron coupling.

    Mixes information across the M output channels so the model can
    capture correlated neural ensemble activity (e.g., co-firing patterns).
    Operates on raw logits (pre-Softplus) so the residual path can learn
    both positive and negative corrections without fighting the non-linearity.

    Architecture:  x → Linear(M, h) → ReLU → Linear(h, M) → + x

    Args:
        num_channels: Number of output channels (M).
        hidden_size: Bottleneck hidden dimension (default 32).
    """

    def __init__(self, num_channels: int, hidden_size: int = 32):
        super().__init__()
        self.num_channels = num_channels
        self.hidden_size = hidden_size

        # Bottleneck MLP: M → hidden → M
        self.mlp = nn.Sequential(
            nn.Linear(num_channels, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_channels),
        )

    def forward(self, x):
        """
        Apply cross-channel coupling with residual connection.

        Args:
            x: Raw logits of shape (batch, M).

        Returns:
            Coupled logits of shape (batch, M).
        """
        return x + self.mlp(x)


def create_teacher_model(
    config: Dict[str, Any],
    input_size: int,
    session_dims: Optional[Dict[str, int]] = None,
) -> nn.Module:
    """
    Factory function to create a teacher model based on config.

    Dispatches on ``config["model"]["architecture"]`` to select
    the backbone architecture. Defaults to "lstm" for backward
    compatibility with configs that don't specify architecture.

    Supported architectures:
        - "lstm": TeacherLSTM (legacy baseline)
        - "lru": TeacherLRU (legacy, ring eigenvalue init)
        - "transformer": TeacherTransformer (legacy baseline)
        - "mamba": TeacherMamba (active primary teacher)

    Args:
        config: Full teacher config dict with 'model' section.
        input_size: Number of input features (M + history features).
        session_dims: Optional dict mapping session_id → neuron count.
            When provided, creates per-session input/output projections.

    Returns:
        Configured teacher model instance.

    Raises:
        ValueError: If architecture is not recognized.
    """
    arch = config.get("model", {}).get("architecture", "lstm")

    if arch == "lstm":
        # Lazy import to avoid circular dependency
        from src.models.teacher import TeacherLSTM
        logger.info("Creating TeacherLSTM (architecture='lstm')")
        return TeacherLSTM.from_config(config, input_size, session_dims)
    elif arch == "lru":
        from src.models.lru import TeacherLRU
        logger.info("Creating TeacherLRU (architecture='lru')")
        return TeacherLRU.from_config(config, input_size, session_dims)
    elif arch == "transformer":
        from src.models.transformer_baseline import TeacherTransformer
        logger.info("Creating TeacherTransformer (architecture='transformer')")
        return TeacherTransformer.from_config(config, input_size, session_dims)
    elif arch == "mamba":
        # Lazy import — mamba-ssm requires Linux + CUDA
        from src.models.mamba_baseline import TeacherMamba
        logger.info("Creating TeacherMamba (architecture='mamba')")
        return TeacherMamba.from_config(config, input_size, session_dims)
    elif arch == "hgrn2":
        # Lazy import — flash-linear-attention requires Linux + CUDA
        from src.models.hgrn2_baseline import TeacherHGRN2
        logger.info("Creating TeacherHGRN2 (architecture='hgrn2')")
        return TeacherHGRN2.from_config(config, input_size, session_dims)
    elif arch == "gated_delta":
        # Vendored non-diagonal SSM (delta rule); no FLA dep.
        from src.models.gated_delta_baseline import TeacherGatedDelta
        logger.info("Creating TeacherGatedDelta (architecture='gated_delta')")
        return TeacherGatedDelta.from_config(config, input_size, session_dims)
    else:
        raise ValueError(
            f"Unknown architecture: '{arch}'. Must be 'lstm', 'lru', "
            f"'transformer', 'mamba', 'hgrn2', or 'gated_delta'."
        )
