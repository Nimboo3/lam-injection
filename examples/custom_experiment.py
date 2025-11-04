"""
Custom Experiment Example

Demonstrates how to create a custom experiment with:
- Multiple attack types
- Defense mechanisms
- Custom analysis

This example shows the full flexibility of the benchmark.
"""

from experiments.experiment_runner import ExperimentRunner
from llm.mock_llm import MockLLM
from defenses.sanitizer import PromptSanitizer
from defenses.detector import InjectionDetector
from analysis.visualize import (
    plot_attack_type_comparison,
    plot_defense_effectiveness,
    create_summary_dashboard
)
import pandas as pd


def run_baseline_experiment():
    """Run experiment without defenses."""
    print("Running baseline experiment (no defenses)...")
    
    llm = MockLLM(injection_threshold=0.5, seed=42)
    runner = ExperimentRunner(
        llm=llm,
        grid_size=5,
        max_steps=30,
        defense=None,  # No defense
        seed=42
    )
    
    results = runner.run_experiment(
        attack_strengths=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        num_episodes=10,
        attack_types=['direct', 'hidden', 'polite'],
        output_dir="data/custom_experiment/baseline"
    )
    
    results['defense_type'] = 'none'
    print(f"  ✓ Baseline complete: {len(results)} results")
    return results


def run_sanitizer_experiment():
    """Run experiment with sanitizer defense."""
    print("Running experiment with sanitizer defense...")
    
    llm = MockLLM(injection_threshold=0.5, seed=42)
    runner = ExperimentRunner(
        llm=llm,
        grid_size=5,
        max_steps=30,
        defense=PromptSanitizer(aggressive=True),
        seed=42
    )
    
    results = runner.run_experiment(
        attack_strengths=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        num_episodes=10,
        attack_types=['direct', 'hidden', 'polite'],
        output_dir="data/custom_experiment/sanitizer"
    )
    
    results['defense_type'] = 'sanitizer'
    print(f"  ✓ Sanitizer complete: {len(results)} results")
    return results


def run_detector_experiment():
    """Run experiment with detector defense."""
    print("Running experiment with detector defense...")
    
    llm = MockLLM(injection_threshold=0.5, seed=42)
    runner = ExperimentRunner(
        llm=llm,
        grid_size=5,
        max_steps=30,
        defense=InjectionDetector(threshold=5.0),
        seed=42
    )
    
    results = runner.run_experiment(
        attack_strengths=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        num_episodes=10,
        attack_types=['direct', 'hidden', 'polite'],
        output_dir="data/custom_experiment/detector"
    )
    
    results['defense_type'] = 'detector'
    print(f"  ✓ Detector complete: {len(results)} results")
    return results


def analyze_results(baseline_df, sanitizer_df, detector_df):
    """Analyze and compare results."""
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print()
    
    # Combine results
    all_results = pd.concat([baseline_df, sanitizer_df, detector_df], ignore_index=True)
    
    # 1. Overall ASR by defense type
    print("1. Attack Success Rate by Defense Type")
    print("-" * 70)
    defense_asr = all_results.groupby('defense_type')['asr'].mean().sort_values(ascending=False)
    for defense, asr in defense_asr.items():
        print(f"  {defense:15s}: {asr:6.1%}")
    print()
    
    # 2. ASR by attack type
    print("2. Attack Success Rate by Attack Type")
    print("-" * 70)
    attack_type_asr = all_results.groupby('attack_type')['asr'].mean().sort_values(ascending=False)
    for attack_type, asr in attack_type_asr.items():
        print(f"  {attack_type:15s}: {asr:6.1%}")
    print()
    
    # 3. Defense effectiveness
    print("3. Defense Effectiveness (ASR Reduction)")
    print("-" * 70)
    baseline_asr = baseline_df['asr'].mean()
    for defense_type in ['sanitizer', 'detector']:
        defense_df = all_results[all_results['defense_type'] == defense_type]
        defense_asr = defense_df['asr'].mean()
        reduction = (baseline_asr - defense_asr) / baseline_asr if baseline_asr > 0 else 0
        print(f"  {defense_type:15s}: {reduction:6.1%} reduction")
    print()
    
    # 4. Utility preservation
    print("4. Utility Preservation")
    print("-" * 70)
    for defense_type in ['none', 'sanitizer', 'detector']:
        defense_df = all_results[all_results['defense_type'] == defense_type]
        utility = defense_df['utility'].mean()
        print(f"  {defense_type:15s}: {utility:6.1%} goal-reaching rate")
    print()
    
    # 5. Attack type vs defense interaction
    print("5. Attack Type × Defense Interaction")
    print("-" * 70)
    pivot = all_results.pivot_table(
        values='asr',
        index='attack_type',
        columns='defense_type',
        aggfunc='mean'
    )
    print(pivot.round(3))
    print()
    
    return all_results


def create_visualizations(all_results):
    """Generate visualizations."""
    print("Generating visualizations...")
    
    try:
        # Attack type comparison
        fig1 = plot_attack_type_comparison(
            all_results[all_results['defense_type'] == 'none'],
            save_path="data/custom_experiment/attack_types.png"
        )
        print("  ✓ Attack type comparison saved")
        
        # Defense effectiveness (sanitizer)
        fig2 = plot_defense_effectiveness(
            all_results[all_results['defense_type'].isin(['none', 'sanitizer'])],
            defense_column='defense_type',
            save_path="data/custom_experiment/sanitizer_effectiveness.png"
        )
        print("  ✓ Sanitizer effectiveness saved")
        
        # Summary dashboard
        fig3 = create_summary_dashboard(
            all_results,
            save_path="data/custom_experiment/dashboard.png",
            figsize=(18, 12)
        )
        print("  ✓ Summary dashboard saved")
        
    except Exception as e:
        print(f"  ⚠ Visualization error: {e}")
    
    print()


def main():
    print("=" * 70)
    print("CUSTOM EXPERIMENT: Attack Types × Defense Mechanisms")
    print("=" * 70)
    print()
    print("This experiment compares:")
    print("  • 3 attack types: direct, hidden, polite")
    print("  • 3 defense conditions: none, sanitizer, detector")
    print("  • 6 attack strengths: 0.0 to 1.0")
    print("  • 10 episodes per configuration")
    print()
    print("Total episodes: 3 × 3 × 6 × 10 = 540")
    print("Estimated time: ~5 minutes")
    print()
    input("Press Enter to continue...")
    print()
    
    # Run experiments
    baseline_df = run_baseline_experiment()
    sanitizer_df = run_sanitizer_experiment()
    detector_df = run_detector_experiment()
    
    # Analyze
    all_results = analyze_results(baseline_df, sanitizer_df, detector_df)
    
    # Save combined results
    all_results.to_csv("data/custom_experiment/all_results.csv", index=False)
    print(f"Combined results saved: data/custom_experiment/all_results.csv")
    print()
    
    # Visualize
    create_visualizations(all_results)
    
    print("=" * 70)
    print("EXPERIMENT COMPLETE!")
    print("=" * 70)
    print()
    print("View results:")
    print("  • All results: data/custom_experiment/all_results.csv")
    print("  • Plots: data/custom_experiment/*.png")
    print("  • Logs: data/custom_experiment/{baseline,sanitizer,detector}/logs/")
    print()
    print("Key findings:")
    print("  • Check which attack type is most effective")
    print("  • Compare defense mechanisms")
    print("  • Analyze security-utility trade-offs")
    print()


if __name__ == "__main__":
    main()
