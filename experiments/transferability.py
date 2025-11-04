"""
Transferability experiments for prompt injection attacks.

Tests whether attacks generated for one model transfer to other models,
and whether defenses trained on one model generalize.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

from experiments.runner import ExperimentConfig, run_experiment_grid
from analysis.metrics import calculate_transfer_score


@dataclass
class TransferabilityConfig:
    """Configuration for transferability experiments."""
    
    # Source/target model pairs
    source_models: List[str] = None
    target_models: List[str] = None
    
    # Attack parameters
    attack_strengths: List[float] = None
    attack_types: List[str] = None
    
    # Experiment parameters
    n_episodes: int = 20
    seed: int = 42
    output_dir: str = "data/transferability"
    
    def __post_init__(self):
        """Set defaults."""
        if self.source_models is None:
            self.source_models = ['mock']
        if self.target_models is None:
            self.target_models = ['mock']
        if self.attack_strengths is None:
            self.attack_strengths = [0.0, 0.3, 0.6, 0.9]
        if self.attack_types is None:
            self.attack_types = ['direct']


def run_cross_model_experiment(
    source_model: str,
    target_model: str,
    attack_strengths: List[float],
    attack_types: List[str] = ['direct'],
    n_episodes: int = 20,
    seed: int = 42,
    output_dir: str = "data/transferability",
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run attack transferability experiment between two models.
    
    Tests if attacks generated for source_model also work on target_model.
    
    Args:
        source_model: Model to generate attacks for
        target_model: Model to test attacks on
        attack_strengths: List of attack strengths to test
        attack_types: List of attack types to test
        n_episodes: Episodes per configuration
        seed: Random seed
        output_dir: Output directory
        verbose: Print progress
    
    Returns:
        Tuple of (source_results, target_results) DataFrames
    """
    if verbose:
        print("=" * 70)
        print(f"Cross-Model Transferability: {source_model} → {target_model}")
        print("=" * 70)
        print()
    
    # Run on source model
    if verbose:
        print(f"[1/2] Running experiments on SOURCE model: {source_model}")
        print("-" * 70)
    
    source_config = ExperimentConfig(
        models=[source_model],
        attack_strengths=attack_strengths,
        attack_types=attack_types,
        n_episodes=n_episodes,
        seed=seed,
        output_dir=f"{output_dir}/source_{source_model}"
    )
    
    source_results = run_experiment_grid(source_config, verbose=False)
    
    if verbose:
        print(f"Source ASR: {source_results['asr'].mean():.2%}")
        print()
    
    # Run on target model with SAME attack documents
    if verbose:
        print(f"[2/2] Running experiments on TARGET model: {target_model}")
        print("-" * 70)
    
    target_config = ExperimentConfig(
        models=[target_model],
        attack_strengths=attack_strengths,
        attack_types=attack_types,
        n_episodes=n_episodes,
        seed=seed,  # Same seed = same attack documents
        output_dir=f"{output_dir}/target_{target_model}"
    )
    
    target_results = run_experiment_grid(target_config, verbose=False)
    
    if verbose:
        print(f"Target ASR: {target_results['asr'].mean():.2%}")
        print()
    
    # Calculate transfer score
    transfer_score = calculate_transfer_score(
        source_results, 
        target_results, 
        metric='asr'
    )
    
    if verbose:
        print("=" * 70)
        print(f"Transfer Score: {transfer_score:.3f}")
        print(f"  (1.0 = perfect transfer, 0.5 = no correlation, 0.0 = negative)")
        print("=" * 70)
        print()
    
    return source_results, target_results


def compute_transfer_matrix(
    models: List[str],
    attack_strengths: List[float] = [0.3, 0.6, 0.9],
    attack_types: List[str] = ['direct'],
    n_episodes: int = 10,
    seed: int = 42,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compute full transfer matrix between all model pairs.
    
    For each (source, target) pair, measures attack transferability.
    
    Args:
        models: List of model names
        attack_strengths: Attack strengths to test
        attack_types: Attack types to test
        n_episodes: Episodes per configuration
        seed: Random seed
        verbose: Print progress
    
    Returns:
        DataFrame with transfer scores (rows=source, cols=target)
    """
    n_models = len(models)
    transfer_matrix = np.zeros((n_models, n_models))
    
    if verbose:
        print("=" * 70)
        print("Computing Transfer Matrix")
        print("=" * 70)
        print(f"Models: {models}")
        print(f"Total pairs: {n_models * n_models}")
        print("=" * 70)
        print()
    
    for i, source in enumerate(models):
        for j, target in enumerate(models):
            if verbose:
                print(f"[{i*n_models + j + 1}/{n_models*n_models}] "
                      f"{source} → {target}", end=" ")
            
            if source == target:
                # Perfect transfer to self
                transfer_matrix[i, j] = 1.0
                if verbose:
                    print("(self) ✓")
            else:
                # Run cross-model experiment
                source_results, target_results = run_cross_model_experiment(
                    source_model=source,
                    target_model=target,
                    attack_strengths=attack_strengths,
                    attack_types=attack_types,
                    n_episodes=n_episodes,
                    seed=seed,
                    verbose=False
                )
                
                # Calculate transfer score
                score = calculate_transfer_score(
                    source_results,
                    target_results,
                    metric='asr'
                )
                
                transfer_matrix[i, j] = score
                
                if verbose:
                    print(f"score={score:.3f}")
    
    # Convert to DataFrame
    df = pd.DataFrame(
        transfer_matrix,
        index=models,
        columns=models
    )
    
    if verbose:
        print()
        print("=" * 70)
        print("Transfer Matrix:")
        print("=" * 70)
        print(df.to_string(float_format=lambda x: f"{x:.3f}"))
        print("=" * 70)
    
    return df


def evaluate_defense_transfer(
    defense_type: str,
    models: List[str],
    attack_strength: float = 0.7,
    n_episodes: int = 10,
    seed: int = 42,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Evaluate if defenses transfer across models.
    
    Evaluates defense effectiveness on each model.
    
    Args:
        defense_type: 'sanitizer', 'detector', or 'verifier'
        models: List of models to test
        attack_strength: Attack strength to use
        n_episodes: Episodes per configuration
        seed: Random seed
        verbose: Print progress
    
    Returns:
        DataFrame with columns: model, baseline_asr, defended_asr, effectiveness
    """
    if verbose:
        print("=" * 70)
        print(f"Defense Transfer Test: {defense_type}")
        print("=" * 70)
        print()
    
    results = []
    
    for model in models:
        if verbose:
            print(f"Testing {defense_type} on {model}...")
        
        # Baseline (no defense)
        baseline_config = ExperimentConfig(
            models=[model],
            attack_strengths=[attack_strength],
            use_sanitizer=[False],
            use_detector=[False],
            use_verifier=[False],
            n_episodes=n_episodes,
            seed=seed
        )
        
        baseline_df = run_experiment_grid(baseline_config, verbose=False)
        baseline_asr = baseline_df['asr'].mean()
        
        # With defense
        defended_config = ExperimentConfig(
            models=[model],
            attack_strengths=[attack_strength],
            use_sanitizer=[defense_type == 'sanitizer'],
            use_detector=[defense_type == 'detector'],
            use_verifier=[defense_type == 'verifier'],
            n_episodes=n_episodes,
            seed=seed
        )
        
        defended_df = run_experiment_grid(defended_config, verbose=False)
        defended_asr = defended_df['asr'].mean()
        
        # Calculate effectiveness
        if baseline_asr > 0:
            effectiveness = (baseline_asr - defended_asr) / baseline_asr
        else:
            effectiveness = 0.0
        
        results.append({
            'model': model,
            'defense': defense_type,
            'baseline_asr': baseline_asr,
            'defended_asr': defended_asr,
            'effectiveness': effectiveness
        })
        
        if verbose:
            print(f"  Baseline ASR: {baseline_asr:.2%}")
            print(f"  Defended ASR: {defended_asr:.2%}")
            print(f"  Effectiveness: {effectiveness:.2%}")
            print()
    
    df = pd.DataFrame(results)
    
    if verbose:
        print("=" * 70)
        print("Defense Transfer Summary:")
        print("=" * 70)
        print(df.to_string(index=False))
        print("=" * 70)
    
    return df


def analyze_attack_type_transfer(
    models: List[str],
    attack_types: List[str] = ['direct', 'hidden', 'polite'],
    attack_strength: float = 0.7,
    n_episodes: int = 10,
    seed: int = 42,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Analyze which attack types transfer best across models.
    
    Args:
        models: List of models to test
        attack_types: Attack types to compare
        attack_strength: Fixed attack strength
        n_episodes: Episodes per configuration
        seed: Random seed
        verbose: Print progress
    
    Returns:
        DataFrame with ASR for each (model, attack_type) pair
    """
    if verbose:
        print("=" * 70)
        print("Attack Type Transferability Analysis")
        print("=" * 70)
        print()
    
    results = []
    
    for model in models:
        if verbose:
            print(f"Model: {model}")
            print("-" * 70)
        
        for attack_type in attack_types:
            config = ExperimentConfig(
                models=[model],
                attack_strengths=[attack_strength],
                attack_types=[attack_type],
                n_episodes=n_episodes,
                seed=seed
            )
            
            df = run_experiment_grid(config, verbose=False)
            asr = df['asr'].mean()
            
            results.append({
                'model': model,
                'attack_type': attack_type,
                'asr': asr
            })
            
            if verbose:
                print(f"  {attack_type:10s}: ASR = {asr:.2%}")
        
        if verbose:
            print()
    
    df = pd.DataFrame(results)
    
    # Pivot for easier comparison
    pivot = df.pivot(index='model', columns='attack_type', values='asr')
    
    if verbose:
        print("=" * 70)
        print("Attack Type Comparison (ASR):")
        print("=" * 70)
        print(pivot.to_string(float_format=lambda x: f"{x:.2%}"))
        print("=" * 70)
    
    return df


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Transferability Experiments Demo")
    print("=" * 70)
    print()
    
    # Demo 1: Cross-model transfer (using different thresholds as proxy for different models)
    print("Demo 1: Cross-Model Attack Transfer")
    print("Testing if attacks that work on threshold=0.3 also work on threshold=0.7")
    print()
    
    # Simulate "different models" using different injection thresholds
    # Lower threshold = easier to attack = "weaker model"
    # Higher threshold = harder to attack = "stronger model"
    
    config_weak = ExperimentConfig(
        models=['mock'],
        injection_thresholds=[0.3],  # Easier to attack
        attack_strengths=[0.5, 0.7],
        n_episodes=5,
        seed=42,
        output_dir='data/transferability/demo'
    )
    
    config_strong = ExperimentConfig(
        models=['mock'],
        injection_thresholds=[0.7],  # Harder to attack
        attack_strengths=[0.5, 0.7],
        n_episodes=5,
        seed=42,  # Same seed = same attacks
        output_dir='data/transferability/demo'
    )
    
    print("Running on 'weak' model (threshold=0.3)...")
    results_weak = run_experiment_grid(config_weak, verbose=False)
    
    print("Running on 'strong' model (threshold=0.7)...")
    results_strong = run_experiment_grid(config_strong, verbose=False)
    
    # Compare
    print()
    print("Results:")
    print(f"  Weak model ASR:   {results_weak['asr'].mean():.2%}")
    print(f"  Strong model ASR: {results_strong['asr'].mean():.2%}")
    
    transfer_score = calculate_transfer_score(results_weak, results_strong, metric='asr')
    print(f"  Transfer score:   {transfer_score:.3f}")
    
    print()
    print("=" * 70)
    print("Demo complete!")
    print()
    print("For full transferability experiments, use:")
    print("  - compute_transfer_matrix() for all model pairs")
    print("  - test_defense_transfer() for defense generalization")
    print("  - analyze_attack_type_transfer() for attack type comparison")
    print("=" * 70)
