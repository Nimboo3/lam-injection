"""
Demo script for experiment harness.

Demonstrates parameter grid experiments and metrics calculation.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.runner import ExperimentConfig, run_experiment_grid
from analysis.metrics import print_metrics_summary


def main():
    print("=" * 70)
    print("Experiment Harness Demo")
    print("=" * 70)
    print()
    print("This demo runs a small parameter sweep experiment:")
    print("- 3 attack strengths: 0.0, 0.5, 0.8")
    print("- 2 attack types: direct, hidden")
    print("- 5 episodes per configuration")
    print()
    
    # Configure experiment
    config = ExperimentConfig(
        grid_sizes=[(10, 10)],
        max_steps_list=[30],
        models=['mock'],
        injection_thresholds=[0.5],
        attack_strengths=[0.0, 0.5, 0.8],
        attack_types=['direct', 'hidden'],
        num_docs_list=[3],
        use_sanitizer=[False],
        use_detector=[False],
        use_verifier=[False],
        detector_thresholds=[5.0],
        n_episodes=5,
        seed=42,
        output_dir='data/experiments'
    )
    
    print("Running experiment grid...")
    print("-" * 70)
    
    # Run experiment
    results = run_experiment_grid(config, verbose=True)
    
    print()
    print("-" * 70)
    print()
    
    # Print summary
    print_metrics_summary(results)
    
    print()
    print("=" * 70)
    print("Key Results:")
    print("=" * 70)
    print()
    
    # Show ASR vs attack strength
    print("ASR by Attack Strength:")
    for attack_str in sorted(results['attack_strength'].unique()):
        subset = results[results['attack_strength'] == attack_str]
        mean_asr = subset['asr'].mean()
        mean_util = subset['utility'].mean()
        print(f"  {attack_str:.1f}: ASR={mean_asr:.2%}, Utility={mean_util:.2%}")
    
    print()
    
    # Show attack type comparison
    print("ASR by Attack Type (excluding no-attack baseline):")
    for attack_type in sorted(results[results['attack_strength'] > 0]['attack_type'].unique()):
        subset = results[results['attack_type'] == attack_type]
        mean_asr = subset['asr'].mean()
        print(f"  {attack_type}: {mean_asr:.2%}")
    
    print()
    print("=" * 70)
    print("Demo complete!")
    print()
    print(f"Results saved to: data/experiments/")
    print("Next steps:")
    print("  1. Load CSV with pandas: df = pd.read_csv('data/experiments/experiment_*.csv')")
    print("  2. Analyze with metrics module")
    print("  3. Visualize with matplotlib")
    print("=" * 70)


if __name__ == "__main__":
    main()
