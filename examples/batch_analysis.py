"""
Batch Analysis Example

Demonstrates how to load and analyze existing experiment results.
This is useful when you have CSV files from previous experiments.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from analysis.visualize import (
    plot_asr_vs_strength,
    plot_utility_vs_strength,
    plot_asr_and_utility,
    plot_pareto_frontier,
    create_summary_dashboard
)
from experiments.metrics import compute_pareto_frontier


def load_results(results_dir="data"):
    """Load all CSV result files from a directory."""
    print(f"Loading results from {results_dir}...")
    
    results_path = Path(results_dir)
    csv_files = list(results_path.glob("**/results.csv"))
    
    if not csv_files:
        print(f"  ⚠ No results.csv files found in {results_dir}")
        print("  Creating sample data for demonstration...")
        return create_sample_data()
    
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df['source_file'] = str(csv_file)
        dfs.append(df)
        print(f"  ✓ Loaded {len(df)} results from {csv_file.parent.name}")
    
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal results: {len(combined_df)}")
    return combined_df


def create_sample_data():
    """Create sample data for demonstration."""
    np.random.seed(42)
    
    strengths = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    attack_types = ['direct', 'hidden', 'polite']
    n_runs = 20
    
    data = []
    for strength in strengths:
        for attack_type in attack_types:
            for run in range(n_runs):
                # ASR increases with strength
                asr = min(strength * 0.85 + np.random.uniform(-0.1, 0.1), 1.0)
                asr = max(0, asr)
                
                # Utility decreases with strength
                utility = max(1.0 - strength * 0.4 + np.random.uniform(-0.1, 0.1), 0.0)
                utility = min(1.0, utility)
                
                # Average steps
                avg_steps = 15 + np.random.randint(-5, 5)
                
                data.append({
                    'run_id': run,
                    'attack_strength': strength,
                    'attack_type': attack_type,
                    'asr': asr,
                    'utility': utility,
                    'avg_steps': avg_steps,
                    'compromised_count': int(asr > 0.5),
                    'goal_reached_count': int(utility > 0.5)
                })
    
    df = pd.DataFrame(data)
    print(f"  ✓ Created {len(df)} sample results")
    return df


def basic_statistics(df):
    """Compute basic statistics."""
    print("\n" + "=" * 70)
    print("BASIC STATISTICS")
    print("=" * 70)
    print()
    
    print("Overall Metrics:")
    print(f"  • Mean ASR: {df['asr'].mean():.1%}")
    print(f"  • Mean Utility: {df['utility'].mean():.1%}")
    print(f"  • Mean Steps: {df['avg_steps'].mean():.1f}")
    print()
    
    print("By Attack Strength:")
    grouped = df.groupby('attack_strength').agg({
        'asr': ['mean', 'std'],
        'utility': ['mean', 'std']
    }).round(3)
    print(grouped)
    print()
    
    if 'attack_type' in df.columns:
        print("By Attack Type:")
        type_grouped = df.groupby('attack_type').agg({
            'asr': 'mean',
            'utility': 'mean'
        }).round(3)
        print(type_grouped)
        print()


def correlation_analysis(df):
    """Analyze correlations between variables."""
    print("=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)
    print()
    
    # Correlation matrix
    numeric_cols = ['attack_strength', 'asr', 'utility', 'avg_steps']
    available_cols = [col for col in numeric_cols if col in df.columns]
    
    corr = df[available_cols].corr()
    print("Correlation Matrix:")
    print(corr.round(3))
    print()
    
    # Key correlations
    print("Key Correlations:")
    print(f"  • Attack Strength ↔ ASR: {corr.loc['attack_strength', 'asr']:+.3f}")
    print(f"  • Attack Strength ↔ Utility: {corr.loc['attack_strength', 'utility']:+.3f}")
    print(f"  • ASR ↔ Utility: {corr.loc['asr', 'utility']:+.3f}")
    print()


def pareto_analysis(df):
    """Compute and analyze Pareto frontier."""
    print("=" * 70)
    print("PARETO FRONTIER ANALYSIS")
    print("=" * 70)
    print()
    
    # Group by configuration
    grouped_df = df.groupby('attack_strength')[['asr', 'utility']].mean().reset_index()
    
    # Compute Pareto frontier (minimize ASR, maximize utility)
    costs = grouped_df[['asr', 'utility']].values
    costs[:, 1] = -costs[:, 1]  # Negate utility to minimize
    
    pareto_indices = compute_pareto_frontier(costs)
    pareto_df = grouped_df.iloc[pareto_indices]
    
    print(f"Found {len(pareto_df)} Pareto-optimal configurations:")
    print()
    print(pareto_df.to_string(index=False))
    print()
    
    # Identify best trade-off
    if len(pareto_df) > 0:
        # Best = highest utility among Pareto points
        best_idx = pareto_df['utility'].idxmax()
        best_config = pareto_df.loc[best_idx]
        
        print("Recommended Configuration (best trade-off):")
        print(f"  • Attack Strength: {best_config['attack_strength']:.2f}")
        print(f"  • ASR: {best_config['asr']:.1%}")
        print(f"  • Utility: {best_config['utility']:.1%}")
        print()
    
    return grouped_df, pareto_df


def generate_all_plots(df, grouped_df=None, pareto_df=None):
    """Generate comprehensive visualizations."""
    print("=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    print()
    
    output_dir = Path("data/batch_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. ASR vs strength
        fig1 = plot_asr_vs_strength(df, save_path=str(output_dir / "asr_curve.png"))
        print("  ✓ ASR curve saved")
        
        # 2. Utility vs strength
        fig2 = plot_utility_vs_strength(df, save_path=str(output_dir / "utility_curve.png"))
        print("  ✓ Utility curve saved")
        
        # 3. Combined ASR and utility
        fig3 = plot_asr_and_utility(df, save_path=str(output_dir / "asr_and_utility.png"))
        print("  ✓ Combined plot saved")
        
        # 4. Pareto frontier
        if grouped_df is not None and pareto_df is not None:
            fig4 = plot_pareto_frontier(
                grouped_df,
                pareto_df=pareto_df,
                save_path=str(output_dir / "pareto_frontier.png")
            )
            print("  ✓ Pareto frontier saved")
        
        # 5. Summary dashboard
        fig5 = create_summary_dashboard(
            df,
            save_path=str(output_dir / "dashboard.png"),
            figsize=(16, 10)
        )
        print("  ✓ Summary dashboard saved")
        
        print()
        print(f"All plots saved to: {output_dir}/")
        
    except Exception as e:
        print(f"  ⚠ Error generating plots: {e}")
    
    print()


def export_summary_report(df, output_path="data/batch_analysis/summary_report.txt"):
    """Export text summary report."""
    print(f"Exporting summary report to {output_path}...")
    
    with open(output_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("BATCH ANALYSIS SUMMARY REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Total Results: {len(df)}\n")
        f.write(f"Attack Strengths: {sorted(df['attack_strength'].unique())}\n")
        if 'attack_type' in df.columns:
            f.write(f"Attack Types: {list(df['attack_type'].unique())}\n")
        f.write("\n")
        
        f.write("Overall Metrics:\n")
        f.write(f"  • Mean ASR: {df['asr'].mean():.1%}\n")
        f.write(f"  • Mean Utility: {df['utility'].mean():.1%}\n")
        f.write(f"  • Mean Steps: {df['avg_steps'].mean():.1f}\n")
        f.write("\n")
        
        f.write("By Attack Strength:\n")
        grouped = df.groupby('attack_strength').agg({
            'asr': 'mean',
            'utility': 'mean'
        }).round(3)
        f.write(grouped.to_string())
        f.write("\n\n")
        
        if 'attack_type' in df.columns:
            f.write("By Attack Type:\n")
            type_grouped = df.groupby('attack_type').agg({
                'asr': 'mean',
                'utility': 'mean'
            }).round(3)
            f.write(type_grouped.to_string())
            f.write("\n\n")
    
    print(f"  ✓ Report saved")
    print()


def main():
    print("=" * 70)
    print("BATCH ANALYSIS EXAMPLE")
    print("=" * 70)
    print()
    
    # Load results
    df = load_results("data")
    
    # Basic statistics
    basic_statistics(df)
    
    # Correlation analysis
    correlation_analysis(df)
    
    # Pareto analysis
    grouped_df, pareto_df = pareto_analysis(df)
    
    # Generate visualizations
    generate_all_plots(df, grouped_df, pareto_df)
    
    # Export report
    export_summary_report(df)
    
    print("=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print()
    print("Outputs:")
    print("  • Plots: data/batch_analysis/*.png")
    print("  • Report: data/batch_analysis/summary_report.txt")
    print()
    print("Next steps:")
    print("  • Review visualizations for insights")
    print("  • Compare across different experiments")
    print("  • Use Pareto frontier to select optimal configurations")
    print()


if __name__ == "__main__":
    main()
