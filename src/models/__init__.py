# Model architectures (teacher ANN and student SNN).
#
# Active modules:
#   - common:              Shared components (PopulationCouplingLayer, factory)
#   - mamba_baseline:      TeacherMamba — primary teacher (selective SSM)
#   - student:             StudentSNN — primary student (multi neuron-type)
#   - selective_rsynaptic: Input-dependent β neuron (Mamba↔SNN bridge)
#   - gac_snn:             GacStudentSNN — mechanism-aligned student (exploratory)
#
# Legacy modules (functional, not for new work):
#   - teacher:             TeacherLSTM — superseded by Mamba (ADR-0017)
#   - lru:                 TeacherLRU — superseded by Mamba
#   - transformer_baseline: TeacherTransformer — benchmark only
#   - moe_output:          MoE output layer — never deployed
#   - ti_lif:              Ternary-Integer LIF — available via student neuron_type
#   - mamba_classifier:    Mamba for SHD classification — benchmark only

from src.models.common import (
    VALID_DISTRIBUTIONS,
    PopulationCouplingLayer,
    create_teacher_model,
)

__all__ = [
    "VALID_DISTRIBUTIONS",
    "PopulationCouplingLayer",
    "create_teacher_model",
]
