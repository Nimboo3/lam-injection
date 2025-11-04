"""
Quick Benchmark Example

A minimal example showing how to run a basic benchmark experiment.
This is the fastest way to get started with the benchmark.
"""

from experiments.experiment_runner import ExperimentRunner
from llm.mock_llm import MockLLM
from analysis.visualize import plot_asr_vs_strength, plot_asr_and_utility
import pandas as pd


def main():
    print("=" * 70)
    print("QUICK BENCHMARK EXAMPLE")
    print("=" * 70)
    print()
    
    # Step 1: Create LLM (mock mode - no API key needed)
    print("Step 1: Creating mock LLM...")
    llm = MockLLM(injection_threshold=0.5)  # 50% threshold for attacks
    print(f"  ✓ Mock LLM created (threshold={llm.injection_threshold})")
    print()
    
    # Step 2: Create experiment runner
    print("Step 2: Setting up experiment runner...")
    runner = ExperimentRunner(
        llm=llm,
        grid_size=5,           # 5x5 grid
        max_steps=30,          # 30 steps per episode
        num_documents=8        # 8 documents per episode
    )
    print(f"  ✓ Runner configured (grid={runner.grid_size}x{runner.grid_size}, "
          f"max_steps={runner.max_steps})")
    print()
    
    # Step 3: Run experiment
    print("Step 3: Running experiment...")
    print("  Testing attack strengths: 0.0, 0.3, 0.6, 0.9")
    print("  Episodes per strength: 5")
    print("  (This will take ~30 seconds)")
    print()
    
    results = runner.run_experiment(
        attack_strengths=[0.0, 0.3, 0.6, 0.9],
        num_episodes=5,
        output_dir="data/quick_benchmark"
    )
    
    print(f"  ✓ Experiment complete! Generated {len(results)} results")
    print()
    
    # Step 4: View results
    print("Step 4: Results Summary")
    print("-" * 70)
    
    # Group by attack strength
    grouped = results.groupby('attack_strength').agg({
        'asr': 'mean',
        'utility': 'mean',
        'avg_steps': 'mean',
        'compromised_count': 'sum'
    }).round(3)
    
    print(grouped)
    print()
    
    # Step 5: Key insights
    print("Step 5: Key Insights")
    print("-" * 70)
    
    asr_0 = results[results['attack_strength'] == 0.0]['asr'].mean()
    asr_9 = results[results['attack_strength'] == 0.9]['asr'].mean()
    util_0 = results[results['attack_strength'] == 0.0]['utility'].mean()
    util_9 = results[results['attack_strength'] == 0.9]['utility'].mean()
    
    print(f"  • ASR at strength 0.0 (no attack): {asr_0:.1%}")
    print(f"  • ASR at strength 0.9 (strong attack): {asr_9:.1%}")
    print(f"  • ASR increase: {(asr_9 - asr_0):.1%}")
    print()
    print(f"  • Utility at strength 0.0: {util_0:.1%}")
    print(f"  • Utility at strength 0.9: {util_9:.1%}")
    print(f"  • Utility degradation: {(util_0 - util_9):.1%}")
    print()
    
    # Step 6: Visualize
    print("Step 6: Generating visualizations...")
    
    try:
        # ASR curve
        fig1 = plot_asr_vs_strength(
            results,
            save_path="data/quick_benchmark/asr_curve.png",
            title="Quick Benchmark: ASR vs Attack Strength"
        )
        print("  ✓ Saved: data/quick_benchmark/asr_curve.png")
        
        # ASR and utility
        fig2 = plot_asr_and_utility(
            results,
            save_path="data/quick_benchmark/asr_and_utility.png"
        )
        print("  ✓ Saved: data/quick_benchmark/asr_and_utility.png")
        
    except Exception as e:
        print(f"  ⚠ Visualization error: {e}")
    
    print()
    print("=" * 70)
    print("BENCHMARK COMPLETE!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. View results: data/quick_benchmark/results.csv")
    print("  2. View plots: data/quick_benchmark/*.png")
    print("  3. View logs: data/quick_benchmark/logs/*.jsonl")
    print()
    print("To customize:")
    print("  • Adjust attack_strengths for finer granularity")
    print("  • Increase num_episodes for statistical significance")
    print("  • Add defenses: runner = ExperimentRunner(defense=PromptSanitizer())")
    print("  • Try different attack types: attack_types=['direct', 'hidden', 'polite']")
    print()


if __name__ == "__main__":
    main()
