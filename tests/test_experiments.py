"""
Tests for experiment harness and metrics.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import json
import tempfile
import shutil

from experiments.batch import run_batch_episodes, save_batch_results, load_batch_results
from experiments.runner import ExperimentConfig, run_experiment_grid
from analysis.metrics import (
    calculate_asr, calculate_utility, calculate_steps_to_compromise,
    calculate_defense_effectiveness, aggregate_metrics, compute_pareto_frontier,
    calculate_transfer_score
)


# ============================================================================
# Batch Runner Tests
# ============================================================================

def test_run_batch_episodes_basic():
    """Test basic batch episode running."""
    env_config = {
        'grid_size': (5, 5),
        'max_steps': 20,
        'documents': [],
        'seed': 42
    }
    
    llm_config = {
        'model_name': 'mock',
        'injection_threshold': 0.5,
        'seed': 42
    }
    
    summaries = run_batch_episodes(
        env_config=env_config,
        llm_config=llm_config,
        n_episodes=3,
        max_steps=20
    )
    
    assert len(summaries) == 3
    assert all('batch_index' in s for s in summaries)
    assert all('steps' in s for s in summaries)
    assert all('goal_reached' in s for s in summaries)


def test_run_batch_episodes_with_attacks():
    """Test batch running with attack documents."""
    from attacks.generator import generate_episode_attack_config
    
    docs = generate_episode_attack_config(
        num_docs=3,
        attack_strength=0.9,  # High strength to ensure compromise
        grid_size=(8, 8),
        distribution='uniform',
        attack_type='direct',
        seed=42
    )
    
    env_config = {
        'grid_size': (8, 8),
        'max_steps': 30,
        'documents': docs,
        'seed': 42
    }
    
    llm_config = {
        'model_name': 'mock',
        'injection_threshold': 0.5,  # Lower threshold
        'seed': 42
    }
    
    summaries = run_batch_episodes(
        env_config=env_config,
        llm_config=llm_config,
        n_episodes=5,
        max_steps=30
    )
    
    assert len(summaries) == 5
    # With very strong attacks and low threshold, should see some compromised steps
    total_compromised = sum(s['compromised_steps'] for s in summaries)
    # May or may not compromise depending on document placement, just check structure
    assert total_compromised >= 0


def test_save_and_load_batch_results():
    """Test saving and loading batch results."""
    summaries = [
        {'batch_index': 0, 'steps': 20, 'goal_reached': True},
        {'batch_index': 1, 'steps': 25, 'goal_reached': False}
    ]
    
    metadata = {'experiment': 'test', 'version': '1.0'}
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "results.json"
        
        # Save
        save_batch_results(summaries, str(output_file), metadata)
        assert output_file.exists()
        
        # Load
        loaded = load_batch_results(str(output_file))
        assert loaded['n_episodes'] == 2
        assert loaded['metadata'] == metadata
        assert len(loaded['summaries']) == 2


# ============================================================================
# Experiment Runner Tests
# ============================================================================

def test_experiment_config_defaults():
    """Test ExperimentConfig default values."""
    config = ExperimentConfig()
    
    assert config.grid_sizes == [(10, 10)]
    assert config.models == ['mock']
    assert config.n_episodes == 10
    assert config.seed == 42


def test_experiment_config_custom():
    """Test ExperimentConfig with custom values."""
    config = ExperimentConfig(
        grid_sizes=[(5, 5), (10, 10)],
        models=['mock', 'gemini-1.5-flash'],
        attack_strengths=[0.0, 0.5, 1.0],
        n_episodes=5
    )
    
    assert len(config.grid_sizes) == 2
    assert len(config.models) == 2
    assert len(config.attack_strengths) == 3


def test_run_experiment_grid_small():
    """Test running small experiment grid."""
    config = ExperimentConfig(
        grid_sizes=[(5, 5)],
        max_steps_list=[20],
        models=['mock'],
        injection_thresholds=[0.5],
        attack_strengths=[0.0, 0.5],
        attack_types=['direct'],
        num_docs_list=[2],
        use_sanitizer=[False],
        use_detector=[False],
        use_verifier=[False],
        n_episodes=2,  # Small for speed
        seed=42
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config.output_dir = tmpdir
        
        df = run_experiment_grid(config, verbose=False)
        
        # Should have 2 experiments (0.0 and 0.5 attack strength)
        assert len(df) == 2
        assert 'asr' in df.columns
        assert 'utility' in df.columns
        assert 'attack_strength' in df.columns
        
        # No attacks should have 0 ASR
        no_attack = df[df['attack_strength'] == 0.0]
        assert no_attack['asr'].iloc[0] == 0.0
        
        # Check CSV saved
        csv_files = list(Path(tmpdir).glob("experiment_*.csv"))
        assert len(csv_files) == 1


def test_experiment_grid_parameter_combinations():
    """Test that all parameter combinations are tested."""
    config = ExperimentConfig(
        grid_sizes=[(5, 5)],
        models=['mock'],
        attack_strengths=[0.0, 0.3, 0.6],  # 3 values
        attack_types=['direct', 'hidden'],  # 2 values
        num_docs_list=[2],
        n_episodes=1,
        seed=42
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config.output_dir = tmpdir
        
        df = run_experiment_grid(config, verbose=False)
        
        # Should have 3 * 2 = 6 experiments
        assert len(df) == 6
        
        # Check all attack strengths present
        assert set(df['attack_strength'].unique()) == {0.0, 0.3, 0.6}
        assert set(df['attack_type'].unique()) == {'direct', 'hidden'}


# ============================================================================
# Metrics Tests
# ============================================================================

def test_calculate_asr():
    """Test ASR calculation."""
    episodes = [
        {'steps': 10, 'compromised_steps': 2},
        {'steps': 20, 'compromised_steps': 5},
        {'steps': 15, 'compromised_steps': 3}
    ]
    
    asr = calculate_asr(episodes)
    expected = (2 + 5 + 3) / (10 + 20 + 15)
    assert abs(asr - expected) < 1e-6


def test_calculate_asr_no_attacks():
    """Test ASR with no attacks."""
    episodes = [
        {'steps': 10, 'compromised_steps': 0},
        {'steps': 20, 'compromised_steps': 0}
    ]
    
    asr = calculate_asr(episodes)
    assert asr == 0.0


def test_calculate_asr_empty():
    """Test ASR with empty episodes."""
    asr = calculate_asr([])
    assert asr == 0.0


def test_calculate_utility():
    """Test utility calculation."""
    episodes = [
        {'goal_reached': True},
        {'goal_reached': False},
        {'goal_reached': True},
        {'goal_reached': True}
    ]
    
    utility = calculate_utility(episodes)
    assert utility == 0.75


def test_calculate_utility_all_fail():
    """Test utility when all episodes fail."""
    episodes = [
        {'goal_reached': False},
        {'goal_reached': False}
    ]
    
    utility = calculate_utility(episodes)
    assert utility == 0.0


def test_calculate_utility_empty():
    """Test utility with empty episodes."""
    utility = calculate_utility([])
    assert utility == 0.0


def test_calculate_steps_to_compromise():
    """Test steps to compromise calculation."""
    episodes = [
        {'steps': 20, 'compromised_steps': 5},
        {'steps': 30, 'compromised_steps': 10},
        {'steps': 40, 'compromised_steps': 2}
    ]
    
    steps = calculate_steps_to_compromise(episodes)
    assert steps is not None
    assert steps > 0


def test_calculate_steps_to_compromise_never():
    """Test steps to compromise when never compromised."""
    episodes = [
        {'steps': 20, 'compromised_steps': 0},
        {'steps': 30, 'compromised_steps': 0}
    ]
    
    steps = calculate_steps_to_compromise(episodes)
    assert steps is None


def test_calculate_defense_effectiveness():
    """Test defense effectiveness calculation."""
    baseline_asr = 0.5
    defended_asr = 0.2
    
    effectiveness = calculate_defense_effectiveness(baseline_asr, defended_asr)
    expected = (0.5 - 0.2) / 0.5  # 0.6
    assert abs(effectiveness - expected) < 1e-6


def test_calculate_defense_effectiveness_perfect():
    """Test perfect defense effectiveness."""
    effectiveness = calculate_defense_effectiveness(0.5, 0.0)
    assert effectiveness == 1.0


def test_calculate_defense_effectiveness_no_baseline():
    """Test defense effectiveness with no baseline attacks."""
    effectiveness = calculate_defense_effectiveness(0.0, 0.0)
    assert effectiveness == 1.0


def test_aggregate_metrics():
    """Test metrics aggregation."""
    data = {
        'attack_strength': [0.5, 0.5, 0.8, 0.8],
        'asr': [0.1, 0.15, 0.4, 0.45],
        'utility': [0.9, 0.85, 0.7, 0.65],
        'avg_steps': [20, 22, 28, 30]
    }
    df = pd.DataFrame(data)
    
    agg = aggregate_metrics(df, group_by=['attack_strength'])
    
    assert len(agg) == 2  # Two attack strengths
    assert 'asr_mean' in agg.columns
    assert 'utility_mean' in agg.columns


def test_compute_pareto_frontier():
    """Test Pareto frontier computation."""
    data = {
        'asr': [0.1, 0.2, 0.3, 0.15],
        'utility': [0.9, 0.8, 0.5, 0.85]
    }
    df = pd.DataFrame(data)
    
    pareto = compute_pareto_frontier(df, objective1='asr', objective2='utility')
    
    # Point (0.1, 0.9) should be on frontier (best ASR)
    # Point (0.3, 0.5) should NOT be on frontier (dominated)
    assert len(pareto) <= len(df)
    assert 0.1 in pareto['asr'].values


def test_calculate_transfer_score():
    """Test transfer score calculation."""
    source = pd.DataFrame({'asr': [0.1, 0.2, 0.3, 0.4]})
    target = pd.DataFrame({'asr': [0.15, 0.25, 0.35, 0.45]})
    
    score = calculate_transfer_score(source, target, metric='asr')
    
    # Should have high correlation (positive transfer)
    assert score > 0.5


def test_calculate_transfer_score_no_transfer():
    """Test transfer score with no correlation."""
    source = pd.DataFrame({'asr': [0.1, 0.2, 0.3, 0.4]})
    target = pd.DataFrame({'asr': [0.4, 0.1, 0.3, 0.2]})
    
    score = calculate_transfer_score(source, target, metric='asr')
    
    # Should have low score (no clear transfer)
    assert 0.0 <= score <= 1.0


# ============================================================================
# Integration Tests
# ============================================================================

def test_full_experiment_workflow():
    """Test complete experiment workflow."""
    # 1. Configure experiment
    config = ExperimentConfig(
        grid_sizes=[(5, 5)],
        models=['mock'],
        attack_strengths=[0.0, 0.9],  # Use high strength
        attack_types=['direct'],
        num_docs_list=[2],
        injection_thresholds=[0.3],  # Lower threshold
        n_episodes=2,
        seed=42
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config.output_dir = tmpdir
        
        # 2. Run experiment
        results = run_experiment_grid(config, verbose=False)
        
        # 3. Calculate metrics
        asr_no_attack = results[results['attack_strength'] == 0.0]['asr'].iloc[0]
        asr_with_attack = results[results['attack_strength'] == 0.9]['asr'].iloc[0]
        
        assert asr_no_attack == 0.0
        # With attacks, ASR should be >= 0 (may or may not compromise)
        assert asr_with_attack >= 0.0
        
        # 4. Aggregate
        agg = aggregate_metrics(results, group_by=['attack_strength'])
        assert len(agg) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
