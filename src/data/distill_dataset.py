"""
Distillation dataset wrapper for online teacher inference.

Wraps existing multi-session DataLoaders to produce (x, y, teacher_rates)
triplets by running a frozen teacher model on each batch.  This avoids
pre-extracting and storing large .pt files on S3.

The wrapper operates at the collate/batch level: it takes a standard
(x, y, mask) batch from MaskedSpikeCountDataset and appends the teacher's
predicted rates.

Usage:
    from src.data.distill_dataset import DistillCollator

    collator = DistillCollator(teacher_model, device)
    loader = DataLoader(dataset, collate_fn=collator, batch_size=512)
    for x, y, teacher_rates in loader:
        ...
"""

import logging
from typing import List, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class DistillCollator:
    """
    Collate function that runs a frozen teacher on each batch.

    Wraps the default collation for MaskedSpikeCountDataset (which yields
    (x, y, mask) or (x, y, mask, cov) tuples) and appends teacher_rates.

    The teacher model is kept in eval mode with torch.no_grad() for
    efficiency.  Teacher inference adds ~50% overhead per batch compared
    to pure data loading, but avoids >10GB pre-extracted target files.

    Args:
        teacher: Pretrained teacher model (LSTM or LRU).  Must be frozen
            (requires_grad=False) and in eval mode.
        device: Device to run teacher inference on.
        output_channels: Number of output channels (M_max) for the teacher.
            If None, uses the full output dimension.
    """

    def __init__(
        self,
        teacher: nn.Module,
        device: torch.device,
        output_channels: int | None = None,
    ):
        self.teacher = teacher
        self.device = device
        self.output_channels = output_channels

        # Ensure teacher is frozen and in eval mode
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad_(False)

        logger.info(
            "DistillCollator: teacher frozen, device=%s, output_channels=%s",
            device, output_channels,
        )

    @torch.no_grad()
    def __call__(
        self,
        batch: List[Tuple[torch.Tensor, ...]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Collate a batch and compute teacher predictions.

        Args:
            batch: List of samples from MaskedSpikeCountDataset.
                Each sample is (x, y, mask) or (x, y, mask, cov).

        Returns:
            Tuple of (x, y, teacher_rates):
                - x: (batch, T, input_size) student input
                - y: (batch, output_channels) ground-truth targets
                - teacher_rates: (batch, output_channels) teacher predictions
        """
        # Standard collation: stack individual samples into batch tensors
        # Handle both 3-tuple (x, y, mask) and 4-tuple (x, y, mask, cov)
        has_covariates = len(batch[0]) == 4

        xs = torch.stack([s[0] for s in batch])
        ys = torch.stack([s[1] for s in batch])
        masks = torch.stack([s[2] for s in batch])

        if has_covariates:
            covs = torch.stack([s[3] for s in batch])
        else:
            covs = None

        # Run teacher inference on GPU
        xs_device = xs.to(self.device)

        if has_covariates and covs is not None:
            teacher_out = self.teacher(xs_device, covariates=covs.to(self.device))
        else:
            teacher_out = self.teacher(xs_device)

        # Teacher output shape depends on model — could be just rates
        # or a dict with 'rates' key.  Handle both cases.
        if isinstance(teacher_out, dict):
            teacher_rates = teacher_out["rates"]
        else:
            teacher_rates = teacher_out

        # Slice to output_channels if needed (session-specific heads
        # may produce variable sizes, but student always uses M_max)
        if self.output_channels is not None:
            teacher_rates = teacher_rates[:, :self.output_channels]
            ys = ys[:, :self.output_channels]

        # Move everything to CPU for the trainer to handle device placement
        return xs.cpu(), ys.cpu(), teacher_rates.cpu()
