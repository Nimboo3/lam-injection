"""
Visualization functions for experiment results.

Provides plotting utilities for ASR curves, transfer matrices, and defense analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any


def plot_asr_vs_strength(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Attack Success Rate vs Attack Strength"
) -> plt.Figure:
    """
    Plot ASR as function of attack strength.
    
    Args:
        results_df: DataFrame with 'attack_strength' and 'asr' columns
        save_path: Optional path to save figure
        figsize: Figure size (width, height)
        title: Plot title
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Group by attack strength and compute statistics
    grouped = results_df.groupby('attack_strength')['asr'].agg(['mean', 'std', 'count'])
    
    # Plot with error bars
    ax.errorbar(
        grouped.index,
        grouped['mean'],
        yerr=grouped['std'],
        marker='o',
        linestyle='-',
        linewidth=2,
        markersize=8,
        capsize=5,
        label='Mean ± Std'
    )
    
    ax.set_xlabel('Attack Strength', fontsize=12)
    ax.set_ylabel('Attack Success Rate (ASR)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig


def plot_utility_vs_strength(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Utility vs Attack Strength"
) -> plt.Figure:
    """
    Plot utility (goal-reaching rate) as function of attack strength.
    
    Args:
        results_df: DataFrame with 'attack_strength' and 'utility' columns
        save_path: Optional path to save figure
        figsize: Figure size
        title: Plot title
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    grouped = results_df.groupby('attack_strength')['utility'].agg(['mean', 'std'])
    
    ax.errorbar(
        grouped.index,
        grouped['mean'],
        yerr=grouped['std'],
        marker='s',
        linestyle='-',
        linewidth=2,
        markersize=8,
        capsize=5,
        color='green',
        label='Mean ± Std'
    )
    
    ax.set_xlabel('Attack Strength', fontsize=12)
    ax.set_ylabel('Utility (Goal-Reaching Rate)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_asr_and_utility(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Plot ASR and utility side-by-side.
    
    Args:
        results_df: DataFrame with results
        save_path: Optional path to save figure
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # ASR plot
    grouped_asr = results_df.groupby('attack_strength')['asr'].agg(['mean', 'std'])
    ax1.errorbar(
        grouped_asr.index,
        grouped_asr['mean'],
        yerr=grouped_asr['std'],
        marker='o',
        linestyle='-',
        linewidth=2,
        markersize=8,
        capsize=5,
        color='red'
    )
    ax1.set_xlabel('Attack Strength', fontsize=11)
    ax1.set_ylabel('ASR', fontsize=11)
    ax1.set_title('Attack Success Rate', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    # Utility plot
    grouped_util = results_df.groupby('attack_strength')['utility'].agg(['mean', 'std'])
    ax2.errorbar(
        grouped_util.index,
        grouped_util['mean'],
        yerr=grouped_util['std'],
        marker='s',
        linestyle='-',
        linewidth=2,
        markersize=8,
        capsize=5,
        color='green'
    )
    ax2.set_xlabel('Attack Strength', fontsize=11)
    ax2.set_ylabel('Utility', fontsize=11)
    ax2.set_title('Goal-Reaching Rate', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_transfer_heatmap(
    transfer_matrix: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    title: str = "Attack Transfer Matrix"
) -> plt.Figure:
    """
    Plot transfer matrix as heatmap.
    
    Args:
        transfer_matrix: DataFrame with transfer scores (source×target)
        save_path: Optional path to save figure
        figsize: Figure size
        title: Plot title
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        transfer_matrix,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Transfer Score'},
        ax=ax
    )
    
    ax.set_xlabel('Target Model', fontsize=12)
    ax.set_ylabel('Source Model', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_defense_effectiveness(
    results_df: pd.DataFrame,
    defense_column: str = 'use_sanitizer',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot ASR with and without defenses.
    
    Args:
        results_df: DataFrame with results
        defense_column: Column indicating defense usage
        save_path: Optional path to save figure
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Split by defense
    no_defense = results_df[~results_df[defense_column]]
    with_defense = results_df[results_df[defense_column]]
    
    if len(no_defense) > 0:
        grouped_no = no_defense.groupby('attack_strength')['asr'].agg(['mean', 'std'])
        ax.errorbar(
            grouped_no.index,
            grouped_no['mean'],
            yerr=grouped_no['std'],
            marker='o',
            linestyle='-',
            linewidth=2,
            markersize=8,
            capsize=5,
            label='No Defense',
            color='red'
        )
    
    if len(with_defense) > 0:
        grouped_with = with_defense.groupby('attack_strength')['asr'].agg(['mean', 'std'])
        ax.errorbar(
            grouped_with.index,
            grouped_with['mean'],
            yerr=grouped_with['std'],
            marker='s',
            linestyle='-',
            linewidth=2,
            markersize=8,
            capsize=5,
            label='With Defense',
            color='blue'
        )
    
    ax.set_xlabel('Attack Strength', fontsize=12)
    ax.set_ylabel('Attack Success Rate', fontsize=12)
    ax.set_title('Defense Effectiveness', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_pareto_frontier(
    results_df: pd.DataFrame,
    objective1: str = 'asr',
    objective2: str = 'utility',
    pareto_df: Optional[pd.DataFrame] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot Pareto frontier for multi-objective optimization.
    
    Args:
        results_df: DataFrame with all results
        objective1: First objective column (minimize)
        objective2: Second objective column (maximize)
        pareto_df: Optional DataFrame with Pareto-optimal points
        save_path: Optional path to save figure
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot all points
    ax.scatter(
        results_df[objective1],
        results_df[objective2],
        alpha=0.5,
        s=50,
        label='All Configurations',
        color='lightblue'
    )
    
    # Plot Pareto frontier if provided
    if pareto_df is not None and len(pareto_df) > 0:
        # Sort for line plot
        pareto_sorted = pareto_df.sort_values(objective1)
        
        ax.scatter(
            pareto_sorted[objective1],
            pareto_sorted[objective2],
            s=100,
            color='red',
            marker='*',
            label='Pareto Frontier',
            zorder=5
        )
        
        ax.plot(
            pareto_sorted[objective1],
            pareto_sorted[objective2],
            color='red',
            linestyle='--',
            linewidth=2,
            alpha=0.7,
            zorder=4
        )
    
    ax.set_xlabel(f'{objective1.upper()}', fontsize=12)
    ax.set_ylabel(f'{objective2.capitalize()}', fontsize=12)
    ax.set_title('Pareto Frontier (ASR vs Utility)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Format axes as percentages if values are in [0, 1]
    if results_df[objective1].max() <= 1:
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    if results_df[objective2].max() <= 1:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_attack_type_comparison(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot ASR comparison across attack types.
    
    Args:
        results_df: DataFrame with 'attack_type' and 'asr' columns
        save_path: Optional path to save figure
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Group by attack type and strength
    for attack_type in results_df['attack_type'].unique():
        subset = results_df[results_df['attack_type'] == attack_type]
        grouped = subset.groupby('attack_strength')['asr'].mean()
        
        ax.plot(
            grouped.index,
            grouped.values,
            marker='o',
            linestyle='-',
            linewidth=2,
            markersize=8,
            label=attack_type.capitalize()
        )
    
    ax.set_xlabel('Attack Strength', fontsize=12)
    ax.set_ylabel('Attack Success Rate', fontsize=12)
    ax.set_title('Attack Type Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_summary_dashboard(
    results_df: pd.DataFrame,
    transfer_matrix: Optional[pd.DataFrame] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 10)
) -> plt.Figure:
    """
    Create comprehensive dashboard with multiple plots.
    
    Args:
        results_df: DataFrame with experiment results
        transfer_matrix: Optional transfer matrix
        save_path: Optional path to save figure
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    if transfer_matrix is not None:
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # ASR vs strength
        ax1 = fig.add_subplot(gs[0, 0])
        grouped_asr = results_df.groupby('attack_strength')['asr'].agg(['mean', 'std'])
        ax1.errorbar(grouped_asr.index, grouped_asr['mean'], yerr=grouped_asr['std'],
                     marker='o', capsize=5, color='red')
        ax1.set_xlabel('Attack Strength')
        ax1.set_ylabel('ASR')
        ax1.set_title('Attack Success Rate', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        # Utility vs strength
        ax2 = fig.add_subplot(gs[0, 1])
        grouped_util = results_df.groupby('attack_strength')['utility'].agg(['mean', 'std'])
        ax2.errorbar(grouped_util.index, grouped_util['mean'], yerr=grouped_util['std'],
                     marker='s', capsize=5, color='green')
        ax2.set_xlabel('Attack Strength')
        ax2.set_ylabel('Utility')
        ax2.set_title('Goal-Reaching Rate', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        # Transfer heatmap
        ax3 = fig.add_subplot(gs[0, 2])
        sns.heatmap(transfer_matrix, annot=True, fmt='.2f', cmap='YlOrRd',
                   vmin=0, vmax=1, ax=ax3, cbar_kws={'label': 'Transfer Score'})
        ax3.set_title('Transfer Matrix', fontweight='bold')
        
        # Attack type comparison (if available)
        if 'attack_type' in results_df.columns:
            ax4 = fig.add_subplot(gs[1, :2])
            for attack_type in results_df['attack_type'].unique():
                subset = results_df[results_df['attack_type'] == attack_type]
                grouped = subset.groupby('attack_strength')['asr'].mean()
                ax4.plot(grouped.index, grouped.values, marker='o', label=attack_type.capitalize())
            ax4.set_xlabel('Attack Strength')
            ax4.set_ylabel('ASR')
            ax4.set_title('Attack Type Comparison', fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        # Defense comparison (if available)
        if 'use_sanitizer' in results_df.columns:
            ax5 = fig.add_subplot(gs[1, 2])
            no_def = results_df[~results_df['use_sanitizer']].groupby('attack_strength')['asr'].mean()
            with_def = results_df[results_df['use_sanitizer']].groupby('attack_strength')['asr'].mean()
            if len(no_def) > 0:
                ax5.plot(no_def.index, no_def.values, marker='o', label='No Defense', color='red')
            if len(with_def) > 0:
                ax5.plot(with_def.index, with_def.values, marker='s', label='With Defense', color='blue')
            ax5.set_xlabel('Attack Strength')
            ax5.set_ylabel('ASR')
            ax5.set_title('Defense Effectiveness', fontweight='bold')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    else:
        # Simpler dashboard without transfer matrix
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        # ASR
        grouped_asr = results_df.groupby('attack_strength')['asr'].agg(['mean', 'std'])
        axes[0].errorbar(grouped_asr.index, grouped_asr['mean'], yerr=grouped_asr['std'],
                        marker='o', capsize=5, color='red')
        axes[0].set_title('Attack Success Rate', fontweight='bold')
        axes[0].set_xlabel('Attack Strength')
        axes[0].set_ylabel('ASR')
        axes[0].grid(True, alpha=0.3)
        
        # Utility
        grouped_util = results_df.groupby('attack_strength')['utility'].agg(['mean', 'std'])
        axes[1].errorbar(grouped_util.index, grouped_util['mean'], yerr=grouped_util['std'],
                        marker='s', capsize=5, color='green')
        axes[1].set_title('Utility', fontweight='bold')
        axes[1].set_xlabel('Attack Strength')
        axes[1].set_ylabel('Utility')
        axes[1].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for ax in axes[2:]:
            ax.axis('off')
    
    plt.suptitle('Experiment Summary Dashboard', fontsize=16, fontweight='bold', y=0.98)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    
    data = {
        'attack_strength': [0.0, 0.3, 0.6, 0.9] * 10,
        'asr': [0.0] * 10 + list(np.random.uniform(0.1, 0.3, 10)) + 
               list(np.random.uniform(0.3, 0.6, 10)) + list(np.random.uniform(0.6, 0.9, 10)),
        'utility': [0.95] * 10 + list(np.random.uniform(0.8, 0.95, 10)) + 
                   list(np.random.uniform(0.6, 0.8, 10)) + list(np.random.uniform(0.4, 0.6, 10)),
        'attack_type': ['direct'] * 20 + ['hidden'] * 20,
        'use_sanitizer': [False] * 20 + [True] * 20
    }
    
    df = pd.DataFrame(data)
    
    print("Visualization Demo")
    print("=" * 70)
    print()
    
    # Create output directory
    output_dir = Path("data/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    print("Generating plots...")
    
    plot_asr_vs_strength(df, save_path=output_dir / "asr_vs_strength.png")
    plot_utility_vs_strength(df, save_path=output_dir / "utility_vs_strength.png")
    plot_asr_and_utility(df, save_path=output_dir / "asr_and_utility.png")
    
    print()
    print(f"Plots saved to {output_dir}/")
    print("=" * 70)

