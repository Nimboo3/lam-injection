"""
Quick script to regenerate plots from existing CSV results.
Use this to create plots without re-running the full experiment.
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def create_plots_from_csv(csv_path: str):
    """Load CSV and create all plots with smart overlap handling."""
    
    # Load data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Models found: {df['model'].unique().tolist()}")
    
    # Output directory
    output_dir = Path(csv_path).parent
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300
    
    # Color scheme
    model_colors = {
        'gemini-2.5-flash-lite': '#FF6B6B',
        'phi3': '#4ECDC4',
        'tinyllama': '#FFA500',
        'qwen2:0.5b': '#E74C3C'
    }
    
    # Different marker styles
    model_markers = {
        'gemini-2.5-flash-lite': 'o',
        'phi3': 's',
        'tinyllama': '^',
        'qwen2:0.5b': 'D'
    }
    
    # Line styles
    model_linestyles = {
        'gemini-2.5-flash-lite': '-',
        'phi3': '-',
        'tinyllama': '--',
        'qwen2:0.5b': '-.'
    }
    
    import numpy as np
    
    models = df['model'].unique()
    num_models = len(models)
    jitter_amount = 0.01
    
    print("\nGenerating plots with overlap handling...")
    
    # Plot 1: ASR vs Attack Strength
    print("  [1/4] ASR comparison...")
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for idx, model in enumerate(models):
        model_data = df[df['model'] == model].copy()
        
        x_values = model_data['attack_strength'].values
        if num_models > 2:
            offset = (idx - num_models/2) * jitter_amount
            x_jittered = x_values + offset
        else:
            x_jittered = x_values
        
        color = model_colors.get(model, '#95A5A6')
        marker = model_markers.get(model, 'o')
        linestyle = model_linestyles.get(model, '-')
        
        ax.plot(
            x_jittered,
            model_data['asr'],
            marker=marker,
            linestyle=linestyle,
            linewidth=2.5,
            markersize=10,
            label=model,
            color=color,
            alpha=0.85,
            markeredgewidth=1.5,
            markeredgecolor='white'
        )
    
    ax.set_xlabel('Attack Strength', fontsize=13, fontweight='bold')
    ax.set_ylabel('Attack Success Rate (ASR)', fontsize=13, fontweight='bold')
    ax.set_title('ASR Comparison: Model Robustness\n(Markers show different models)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper left', frameon=True, shadow=True, fontsize=11,
              framealpha=0.95, edgecolor='black', fancybox=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.set_xlim(-0.05, 0.95)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'asr_comparison.png', dpi=300, bbox_inches='tight')
    print(f"    ✓ Saved: {output_dir / 'asr_comparison.png'}")
    plt.close()
    
    # Plot 2: Utility vs Attack Strength
    print("  [2/4] Utility comparison...")
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for idx, model in enumerate(models):
        model_data = df[df['model'] == model].copy()
        
        x_values = model_data['attack_strength'].values
        if num_models > 2:
            offset = (idx - num_models/2) * jitter_amount
            x_jittered = x_values + offset
        else:
            x_jittered = x_values
        
        color = model_colors.get(model, '#95A5A6')
        marker = model_markers.get(model, 'o')
        linestyle = model_linestyles.get(model, '-')
        
        ax.plot(
            x_jittered,
            model_data['utility'],
            marker=marker,
            linestyle=linestyle,
            linewidth=2.5,
            markersize=10,
            label=model,
            color=color,
            alpha=0.85,
            markeredgewidth=1.5,
            markeredgecolor='white'
        )
    
    ax.set_xlabel('Attack Strength', fontsize=13, fontweight='bold')
    ax.set_ylabel('Utility (Goal-Reaching Rate)', fontsize=13, fontweight='bold')
    ax.set_title('Utility Comparison: Model Performance\n(Higher is better, markers distinguish models)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='lower left', frameon=True, shadow=True, fontsize=11,
              framealpha=0.95, edgecolor='black', fancybox=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.set_xlim(-0.05, 0.95)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'utility_comparison.png', dpi=300, bbox_inches='tight')
    print(f"    ✓ Saved: {output_dir / 'utility_comparison.png'}")
    plt.close()
    
    # Plot 3: Combined ASR & Utility
    print("  [3/4] Combined comparison...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Left subplot: ASR
    for idx, model in enumerate(models):
        model_data = df[df['model'] == model].copy()
        
        x_values = model_data['attack_strength'].values
        if num_models > 2:
            offset = (idx - num_models/2) * jitter_amount
            x_jittered = x_values + offset
        else:
            x_jittered = x_values
        
        color = model_colors.get(model, '#95A5A6')
        marker = model_markers.get(model, 'o')
        linestyle = model_linestyles.get(model, '-')
        
        ax1.plot(x_jittered, model_data['asr'],
                marker=marker, linestyle=linestyle, linewidth=2.5, markersize=10, 
                label=model, color=color, alpha=0.85,
                markeredgewidth=1.5, markeredgecolor='white')
    
    # Right subplot: Utility
    for idx, model in enumerate(models):
        model_data = df[df['model'] == model].copy()
        
        x_values = model_data['attack_strength'].values
        if num_models > 2:
            offset = (idx - num_models/2) * jitter_amount
            x_jittered = x_values + offset
        else:
            x_jittered = x_values
        
        color = model_colors.get(model, '#95A5A6')
        marker = model_markers.get(model, 'o')
        linestyle = model_linestyles.get(model, '-')
        
        ax2.plot(x_jittered, model_data['utility'],
                marker=marker, linestyle=linestyle, linewidth=2.5, markersize=10,
                label=model, color=color, alpha=0.85,
                markeredgewidth=1.5, markeredgecolor='white')
    
    ax1.set_xlabel('Attack Strength', fontsize=12, fontweight='bold')
    ax1.set_ylabel('ASR', fontsize=12, fontweight='bold')
    ax1.set_title('Attack Success Rate', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', frameon=True, shadow=True, fontsize=10,
               framealpha=0.95, edgecolor='black')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax1.set_xlim(-0.05, 0.95)
    
    ax2.set_xlabel('Attack Strength', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Utility', fontsize=12, fontweight='bold')
    ax2.set_title('Goal-Reaching Rate', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', frameon=True, shadow=True, fontsize=10,
               framealpha=0.95, edgecolor='black')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax2.set_xlim(-0.05, 0.95)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'combined_comparison.png', dpi=300, bbox_inches='tight')
    print(f"    ✓ Saved: {output_dir / 'combined_comparison.png'}")
    plt.close()
    
    # Plot 4: Vulnerability ranking (Average ASR with tie detection)
    print("  [4/4] Vulnerability ranking...")
    
    # Calculate average ASR across all strengths
    avg_asr = df.groupby('model')['asr'].mean().sort_values()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models_ranked = avg_asr.index.tolist()
    asr_values = avg_asr.values
    
    colors = [model_colors.get(m, '#95A5A6') for m in models_ranked]
    
    # Detect ties (values within 1%)
    hatches = [''] * len(asr_values)
    for i in range(1, len(asr_values)):
        if abs(asr_values[i] - asr_values[i-1]) < 0.01:
            hatches[i-1] = '///'
            hatches[i] = '///'
    
    # Model sizes for labels
    model_sizes = {
        'phi3': '3.8GB',
        'tinyllama': '637MB',
        'qwen2:0.5b': '352MB',
        'gemini-2.5-flash-lite': 'API'
    }
    
    y_labels = [f"{m}\n({model_sizes.get(m, 'Unknown')})" for m in models_ranked]
    
    bars = ax.barh(range(len(models_ranked)), asr_values, color=colors, 
                   edgecolor='black', linewidth=1.5, hatch=hatches)
    
    ax.set_yticks(range(len(models_ranked)))
    ax.set_yticklabels(y_labels, fontsize=11)
    ax.set_xlabel('Average Attack Success Rate (All Strengths)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model (Size)', fontsize=12, fontweight='bold')
    ax.set_title('Model Vulnerability Ranking\n(Lower is More Secure, Patterns show ties within 1%)', 
                 fontsize=13, fontweight='bold', pad=15)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax.grid(True, alpha=0.3, axis='x', linestyle='--')
    
    # Add value labels on bars (smart positioning)
    for i, (bar, val) in enumerate(zip(bars, asr_values)):
        width = bar.get_width()
        label_text = f'{val:.1%}'
        
        if width < 0.5:
            # Place outside (right) for small bars
            ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                   label_text, ha='left', va='center', 
                   fontsize=10, fontweight='bold', color='black')
        else:
            # Place inside (left) for large bars
            ax.text(width - 0.02, bar.get_y() + bar.get_height()/2,
                   label_text, ha='right', va='center',
                   fontsize=10, fontweight='bold', color='white')
    
    plt.tight_layout()
    fig.savefig(output_dir / 'vulnerability_ranking.png', dpi=300, bbox_inches='tight')
    print(f"    ✓ Saved: {output_dir / 'vulnerability_ranking.png'}")
    plt.close()
    
    print(f"\n✓ All plots saved to: {output_dir}")
    print("\nSummary:")
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        print(f"  {model}: Avg ASR = {model_data['asr'].mean():.1%}")


if __name__ == "__main__":
    # Default to the most recent results file
    csv_file = "data/multi_model_comparison/multi_model_results_20251105_040324.csv"
    
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    
    create_plots_from_csv(csv_file)
