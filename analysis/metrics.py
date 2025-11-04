"""
Metrics calculation for experiment analysis.

Provides functions to compute ASR, utility, and other robustness metrics.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional


def calculate_asr(
    episodes: List[Dict[str, Any]],
    attack_threshold: float = 0.1
) -> float:
    """
    Calculate Attack Success Rate (ASR).
    
    ASR = fraction of timesteps where agent was compromised.
    
    Args:
        episodes: List of episode summary dicts
        attack_threshold: Minimum attack strength to consider
    
    Returns:
        ASR value in [0, 1]
    """
    total_steps = sum(ep['steps'] for ep in episodes)
    compromised_steps = sum(ep['compromised_steps'] for ep in episodes)
    
    if total_steps == 0:
        return 0.0
    
    return compromised_steps / total_steps


def calculate_utility(
    episodes: List[Dict[str, Any]]
) -> float:
    """
    Calculate utility (goal-reaching rate).
    
    Utility = fraction of episodes where goal was reached.
    
    Args:
        episodes: List of episode summary dicts
    
    Returns:
        Utility value in [0, 1]
    """
    if not episodes:
        return 0.0
    
    goal_reached = sum(1 for ep in episodes if ep['goal_reached'])
    return goal_reached / len(episodes)


def calculate_steps_to_compromise(
    episodes: List[Dict[str, Any]]
) -> Optional[float]:
    """
    Calculate average steps until first compromise.
    
    Args:
        episodes: List of episode summary dicts
    
    Returns:
        Average steps to compromise, or None if never compromised
    """
    first_compromise_steps = []
    
    for ep in episodes:
        if ep['compromised_steps'] > 0:
            # Assume compromise happened at midpoint of episode
            # (more precise tracking would require per-step logs)
            first_compromise_steps.append(ep['steps'] / 2)
    
    if not first_compromise_steps:
        return None
    
    return np.mean(first_compromise_steps)


def calculate_defense_effectiveness(
    baseline_asr: float,
    defended_asr: float
) -> float:
    """
    Calculate defense effectiveness.
    
    Effectiveness = reduction in ASR relative to baseline.
    
    Args:
        baseline_asr: ASR without defenses
        defended_asr: ASR with defenses
    
    Returns:
        Effectiveness in [0, 1], where 1 = complete defense
    """
    if baseline_asr == 0:
        return 1.0 if defended_asr == 0 else 0.0
    
    reduction = (baseline_asr - defended_asr) / baseline_asr
    return max(0.0, min(1.0, reduction))


def aggregate_metrics(
    results_df: pd.DataFrame,
    group_by: List[str] = None
) -> pd.DataFrame:
    """
    Aggregate metrics across parameter combinations.
    
    Args:
        results_df: DataFrame from experiment runner
        group_by: Columns to group by (default: ['attack_strength'])
    
    Returns:
        DataFrame with aggregated metrics (mean, std, median)
    """
    if group_by is None:
        group_by = ['attack_strength']
    
    metrics = ['asr', 'utility', 'avg_steps']
    
    agg_funcs = {
        metric: ['mean', 'std', 'median', 'min', 'max']
        for metric in metrics
    }
    
    aggregated = results_df.groupby(group_by).agg(agg_funcs)
    
    # Flatten column names
    aggregated.columns = ['_'.join(col).strip() for col in aggregated.columns.values]
    
    return aggregated.reset_index()


def compute_pareto_frontier(
    results_df: pd.DataFrame,
    objective1: str = 'asr',
    objective2: str = 'utility',
    minimize_obj1: bool = True,
    minimize_obj2: bool = False
) -> pd.DataFrame:
    """
    Compute Pareto frontier for multi-objective optimization.
    
    Args:
        results_df: DataFrame with experiment results
        objective1: First objective column name
        objective2: Second objective column name
        minimize_obj1: Whether to minimize objective1 (default: True for ASR)
        minimize_obj2: Whether to minimize objective2 (default: False for utility)
    
    Returns:
        DataFrame with only Pareto-optimal configurations
    """
    df = results_df.copy()
    
    # Adjust for minimization/maximization
    obj1_vals = df[objective1].values * (1 if minimize_obj1 else -1)
    obj2_vals = df[objective2].values * (1 if minimize_obj2 else -1)
    
    # Find Pareto frontier
    is_pareto = np.ones(len(df), dtype=bool)
    
    for i in range(len(df)):
        for j in range(len(df)):
            if i != j:
                # j dominates i if j is better in both objectives
                if (obj1_vals[j] <= obj1_vals[i] and obj2_vals[j] <= obj2_vals[i] and
                    (obj1_vals[j] < obj1_vals[i] or obj2_vals[j] < obj2_vals[i])):
                    is_pareto[i] = False
                    break
    
    return df[is_pareto].copy()


def calculate_transfer_score(
    source_results: pd.DataFrame,
    target_results: pd.DataFrame,
    metric: str = 'asr'
) -> float:
    """
    Calculate attack transferability score.
    
    Measures how well attacks transfer between models/defenses.
    
    Args:
        source_results: Results on source model/defense
        target_results: Results on target model/defense
        metric: Metric to compare (default: 'asr')
    
    Returns:
        Transfer score in [0, 1], where 1 = perfect transfer
    """
    if len(source_results) != len(target_results):
        raise ValueError("Source and target must have same length")
    
    source_vals = source_results[metric].values
    target_vals = target_results[metric].values
    
    # Correlation as transfer score
    if len(source_vals) < 2:
        return 0.0
    
    correlation = np.corrcoef(source_vals, target_vals)[0, 1]
    
    # Convert to [0, 1] range
    return (correlation + 1) / 2


def print_metrics_summary(results_df: pd.DataFrame):
    """
    Print formatted summary of experiment metrics.
    
    Args:
        results_df: DataFrame from experiment runner
    """
    print("=" * 70)
    print("Metrics Summary")
    print("=" * 70)
    print()
    
    # Overall statistics
    print("Overall Statistics:")
    print(f"  Total experiments: {len(results_df)}")
    print(f"  Total episodes: {results_df['total_episodes'].sum()}")
    print()
    
    # ASR by attack strength
    print("ASR by Attack Strength:")
    asr_by_attack = results_df.groupby('attack_strength')['asr'].agg(['mean', 'std', 'count'])
    for idx, row in asr_by_attack.iterrows():
        print(f"  {idx:.1f}: {row['mean']:.2%} ± {row['std']:.2%} (n={int(row['count'])})")
    print()
    
    # Utility by attack strength
    print("Utility by Attack Strength:")
    util_by_attack = results_df.groupby('attack_strength')['utility'].agg(['mean', 'std'])
    for idx, row in util_by_attack.iterrows():
        print(f"  {idx:.1f}: {row['mean']:.2%} ± {row['std']:.2%}")
    print()
    
    # Defense effectiveness
    if 'use_sanitizer' in results_df.columns:
        print("Defense Effectiveness:")
        baseline = results_df[~results_df['use_sanitizer'] & ~results_df['use_detector']]
        defended = results_df[results_df['use_sanitizer'] | results_df['use_detector']]
        
        if len(baseline) > 0 and len(defended) > 0:
            baseline_asr = baseline['asr'].mean()
            defended_asr = defended['asr'].mean()
            effectiveness = calculate_defense_effectiveness(baseline_asr, defended_asr)
            print(f"  Baseline ASR: {baseline_asr:.2%}")
            print(f"  Defended ASR: {defended_asr:.2%}")
            print(f"  Effectiveness: {effectiveness:.2%}")
        print()
    
    print("=" * 70)


# Example usage
if __name__ == "__main__":
    print("Metrics Calculation Demo")
    print()
    
    # Example episode data
    episodes = [
        {'steps': 20, 'compromised_steps': 5, 'goal_reached': True},
        {'steps': 30, 'compromised_steps': 10, 'goal_reached': False},
        {'steps': 25, 'compromised_steps': 0, 'goal_reached': True},
    ]
    
    # Calculate metrics
    asr = calculate_asr(episodes)
    utility = calculate_utility(episodes)
    steps_to_comp = calculate_steps_to_compromise(episodes)
    
    print(f"ASR: {asr:.2%}")
    print(f"Utility: {utility:.2%}")
    print(f"Steps to compromise: {steps_to_comp:.1f}")
    print()
    
    # Example DataFrame
    data = {
        'attack_strength': [0.0, 0.3, 0.6, 0.9],
        'asr': [0.0, 0.15, 0.45, 0.75],
        'utility': [0.95, 0.85, 0.65, 0.40],
        'avg_steps': [18.5, 22.3, 27.8, 31.2],
        'total_episodes': [10, 10, 10, 10]
    }
    df = pd.DataFrame(data)
    
    print("Example Results:")
    print(df.to_string(index=False))
    print()
    
    # Aggregate
    agg = aggregate_metrics(df)
    print("Aggregated Metrics:")
    print(agg.to_string(index=False))
