"""Experiments package for running systematic evaluations."""

from .runner import run_experiment_grid, ExperimentConfig
from .batch import run_batch_episodes
from .transferability import (
    run_cross_model_experiment,
    compute_transfer_matrix,
    evaluate_defense_transfer,
    analyze_attack_type_transfer,
    TransferabilityConfig
)

__all__ = [
    "run_experiment_grid",
    "ExperimentConfig",
    "run_batch_episodes",
    "run_cross_model_experiment",
    "compute_transfer_matrix",
    "evaluate_defense_transfer",
    "analyze_attack_type_transfer",
    "TransferabilityConfig",
]
