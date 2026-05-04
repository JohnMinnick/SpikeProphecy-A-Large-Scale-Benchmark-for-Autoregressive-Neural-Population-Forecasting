"""
Shared pytest configuration for the test suite.

Disables Weights & Biases during tests to prevent test runs
from polluting the W&B project dashboard.
"""
import os

# Disable W&B for all tests — prevents test runs from creating
# real W&B entries that clutter the production dashboard
os.environ["WANDB_MODE"] = "disabled"
