"""
Tests for DistillTrainer.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.distill.distill_trainer import DistillTrainer
from src.distill.loss import DistillationLoss
from src.models.student import StudentSNN


@pytest.fixture
def distill_data():
    """Create synthetic data for distillation (x, y, y_teacher)."""
    B, T, M = 8, 10, 5
    x = torch.randn(B, T, M).abs()
    y = torch.randint(0, 5, (B, M)).float()
    y_teacher = torch.rand(B, M).abs()
    
    dataset = TensorDataset(x, y, y_teacher)
    loader = DataLoader(dataset, batch_size=4)
    return loader


@pytest.fixture
def model():
    return StudentSNN(input_size=5, hidden_size=10, output_size=5)


@pytest.fixture
def criterion():
    return DistillationLoss(distill_weight=0.5, reg_weight=0.1)


@pytest.fixture
def config():
    return {
        "training": {
            "epochs": 2,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "warmup_epochs": 0,
            "patience": 5,
            "grad_clip_norm": 1.0,
        },
        "loss": {"log_input": False},
    }


def test_distill_trainer_steps(model, distill_data, criterion, config, tmp_path):
    """DistillTrainer should run training and validation steps."""
    device = torch.device("cpu")
    trainer = DistillTrainer(
        model, distill_data, distill_data, config, device, criterion, exp_dir=tmp_path
    )
    
    # Run training
    history = trainer.train()
    
    assert "train_loss" in history
    assert len(history["train_loss"]) == 2
    assert "val_distill_loss" in history
    assert "val_poisson_loss" in history
    
    # Check if files saved
    assert (tmp_path / "final_model.pt").exists()


def test_distill_trainer_loss_backprop(model, distill_data, criterion, config):
    """Gradients should be updated."""
    device = torch.device("cpu")
    trainer = DistillTrainer(
        model, distill_data, distill_data, config, device, criterion
    )
    
    # Store initial weights
    initial_weight = model.input_proj.weight.clone()
    
    # Run one epoch
    trainer._train_one_epoch()
    
    # Weights should change
    assert not torch.allclose(model.input_proj.weight, initial_weight)
