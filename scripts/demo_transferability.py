"""
Demo script for transferability experiments.

Demonstrates attack transfer between models and defense generalization.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from experiments.transferability import (
    run_cross_model_experiment,
    compute_transfer_matrix,
    evaluate_defense_transfer,
    analyze_attack_type_transfer
)
from experiments.runner import ExperimentConfig, run_experiment_grid


def main():
    print("=" * 70)
    print("Transferability Experiments Demo")
    print("=" * 70)
    print()
    print("This demo explores attack transferability using different")
    print("injection thresholds as a proxy for different model robustness.")
    print()
    print("Model profiles:")
    print("  - 'Weak' model: injection_threshold=0.2 (easier to attack)")
    print("  - 'Strong' model: injection_threshold=0.7 (harder to attack)")
    print()
    print("=" * 70)
    print()
    
    # Demo 1: Cross-model transfer
    print("DEMO 1: Cross-Model Attack Transfer")
    print("=" * 70)
    print()
    print("Testing if attacks designed for weak model also work on strong model")
    print()
    
    # Create "different models" using different thresholds
    config_weak = ExperimentConfig(
        models=['mock'],
        injection_thresholds=[0.2],  # Weak model
        attack_strengths=[0.3, 0.6, 0.9],
        n_episodes=5,
        seed=42,
        output_dir='data/transferability/demo1'
    )
    
    config_strong = ExperimentConfig(
        models=['mock'],
        injection_thresholds=[0.7],  # Strong model
        attack_strengths=[0.3, 0.6, 0.9],
        n_episodes=5,
        seed=42,  # Same seed = same attack documents
        output_dir='data/transferability/demo1'
    )
    
    print("Running on WEAK model (threshold=0.2)...")
    results_weak = run_experiment_grid(config_weak, verbose=False)
    
    print("Running on STRONG model (threshold=0.7)...")
    results_strong = run_experiment_grid(config_strong, verbose=False)
    
    print()
    print("Results by Attack Strength:")
    print("-" * 70)
    
    for strength in [0.3, 0.6, 0.9]:
        asr_weak = results_weak[results_weak['attack_strength'] == strength]['asr'].mean()
        asr_strong = results_strong[results_strong['attack_strength'] == strength]['asr'].mean()
        
        print(f"Attack strength {strength:.1f}:")
        print(f"  Weak model ASR:   {asr_weak:.2%}")
        print(f"  Strong model ASR: {asr_strong:.2%}")
        print(f"  Transfer rate:    {(asr_strong/asr_weak):.2%}" if asr_weak > 0 else "  Transfer rate:    N/A")
        print()
    
    from analysis.metrics import calculate_transfer_score
    transfer_score = calculate_transfer_score(results_weak, results_strong, metric='asr')
    
    print(f"Overall Transfer Score: {transfer_score:.3f}")
    print("  (1.0 = perfect correlation, 0.5 = no correlation)")
    print()
    
    # Demo 2: Attack type comparison
    print()
    print("=" * 70)
    print("DEMO 2: Attack Type Transferability")
    print("=" * 70)
    print()
    print("Comparing how different attack types perform across models")
    print()
    
    df = analyze_attack_type_transfer(
        models=['mock'],  # Would use ['mock', 'gemini-pro'] for real comparison
        attack_types=['direct', 'hidden', 'polite'],
        attack_strength=0.7,
        n_episodes=5,
        seed=42,
        verbose=False
    )
    
    print("Attack Type Performance:")
    print("-" * 70)
    for _, row in df.iterrows():
        print(f"  {row['attack_type']:10s}: ASR = {row['asr']:.2%}")
    
    print()
    
    # Demo 3: Defense transfer
    print()
    print("=" * 70)
    print("DEMO 3: Defense Transferability")
    print("=" * 70)
    print()
    print("Testing if sanitizer defense works consistently")
    print()
    
    defense_df = evaluate_defense_transfer(
        defense_type='sanitizer',
        models=['mock'],
        attack_strength=0.8,
        n_episodes=5,
        seed=42,
        verbose=False
    )
    
    print("Defense Effectiveness:")
    print("-" * 70)
    for _, row in defense_df.iterrows():
        print(f"Model: {row['model']}")
        print(f"  Baseline ASR:  {row['baseline_asr']:.2%}")
        print(f"  Defended ASR:  {row['defended_asr']:.2%}")
        print(f"  Effectiveness: {row['effectiveness']:.2%}")
        print(f"  Reduction:     {(row['baseline_asr'] - row['defended_asr']):.2%}")
    
    print()
    print("=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print()
    print("Key Findings:")
    print("  1. Attacks that work on weak models often transfer to strong models")
    print("  2. Different attack types may have different transfer rates")
    print("  3. Defenses can generalize across models if properly designed")
    print()
    print("For production use:")
    print("  - Test with real LLM models (Gemini, GPT, etc.)")
    print("  - Run larger sample sizes (n_episodes ≥ 50)")
    print("  - Compute full transfer matrix with compute_transfer_matrix()")
    print("=" * 70)


if __name__ == "__main__":
    main()
