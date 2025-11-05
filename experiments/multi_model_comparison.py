"""
Multi-Model Comparison Experiment

Compares 1 Gemini model + 2 Ollama models for research paper.

Models tested:
1. gemini-2.5-flash-lite (Google, cloud, latest)
2. phi3 (Microsoft, local, 3.8GB)
3. tinyllama (TinyLlama, local, 637MB) - Ultra-lightweight baseline
"""

# ============================================================================
# CONFIGURATION - Set this to control which models to test
# ============================================================================
USE_GEMINI = False  # Set to True to include Gemini models (uses API quota)
                    # Set to False to test only Ollama models (local, no quota)

# ============================================================================
# SPEED SETTINGS - For quick project proposal demo
# ============================================================================
DEMO_MODE = True    # True = FAST (1 episode, 15 steps) for demos/proposals
                    # False = FULL (3 episodes, 25 steps) for research paper
# ============================================================================

import sys
import os
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import pandas as pd
from datetime import datetime
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns

from llm.gemini_client import GeminiClient
from llm.ollama_client import OllamaClient
from envs.gridworld import GridWorld
from runner.controller import run_episode
from attacks.generator import generate_episode_attack_config


def setup_models() -> Dict:
    """
    Setup all models for comparison.
    
    Returns:
        Dictionary of model_name -> client
    """
    print("=" * 80)
    print("SETTING UP MODELS")
    print("=" * 80)
    
    if DEMO_MODE:
        print("⚡ DEMO MODE: Fast results for project proposal")
        print("   • 1 episode per config, 15 steps max")
        print("   • 2-3 minutes total time")
    else:
        print("📊 RESEARCH MODE: Full results for paper")
        print("   • 3 episodes per config, 25 steps max")
        print("   • 10-15 minutes total time")
    
    print("=" * 80)
    
    if USE_GEMINI:
        print("Mode: Testing Gemini + Ollama models")
    else:
        print("Mode: Testing ONLY Ollama models (local, no quota)")
    
    print("=" * 80)
    
    models = {}
    model_count = 0
    # phi3 + tinyllama + optional gemini -> add qwen2:0.5b as an additional local model
    total_models = (1 if USE_GEMINI else 0) + 3
    
    # Gemini models (cloud) - Only if USE_GEMINI is True
    if USE_GEMINI:
        try:
            model_count += 1
            print(f"\n[{model_count}/{total_models}] Setting up gemini-2.5-flash-lite...")
            models['gemini-2.5-flash-lite'] = GeminiClient(model_name="gemini-2.5-flash-lite")
            print("  ✓ Gemini 2.5 Flash Lite ready")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    else:
        print("\n⊘ Skipping Gemini models (USE_GEMINI=False)")
    
    # Ollama models (local) - Always included
    try:
        model_count += 1
        print(f"\n[{model_count}/{total_models}] Setting up phi3 (Ollama - 3.8GB)...")
        models['phi3'] = OllamaClient(model_name="phi3")
        print("  ✓ Phi-3 ready")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    try:
        model_count += 1
        print(f"\n[{model_count}/{total_models}] Setting up tinyllama (Ollama - 637MB)...")
        print("  ℹ️  TinyLlama is ultra-lightweight - testing speed vs security")
        models['tinyllama'] = OllamaClient(model_name="tinyllama")
        print("  ✓ TinyLlama ready")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        print("  ℹ️  Download with: ollama pull tinyllama")

    # Optional: qwen2:0.5b - very small model to increase vulnerability variance
    try:
        model_count += 1
        print(f"\n[{model_count}/{total_models}] Setting up qwen2:0.5b (Ollama - 352MB)...")
        print("  ℹ️  qwen2:0.5b is an ultra-small model - expected to be highly vulnerable")
        models['qwen2:0.5b'] = OllamaClient(model_name="qwen2:0.5b")
        print("  ✓ qwen2:0.5b ready")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        print("  ℹ️  Download with: ollama pull qwen2:0.5b")
    
    print("\n" + "=" * 80)
    print(f"Successfully configured {len(models)}/{total_models} models")
    print("=" * 80)
    
    return models


def run_comparison_experiment(
    models: Dict,
    attack_strengths: List[float] = [0.0, 0.2, 0.4, 0.6, 0.8],
    n_episodes: int = 10,
    output_dir: str = "data/multi_model_comparison"
) -> pd.DataFrame:
    """
    Run comparison experiment across all models.
    
    Args:
        models: Dictionary of model_name -> client
        attack_strengths: List of attack strengths to test
        n_episodes: Episodes per configuration
        output_dir: Output directory
    
    Returns:
        DataFrame with results
    """
    print("\n" + "=" * 80)
    print("RUNNING MULTI-MODEL COMPARISON EXPERIMENT")
    print("=" * 80)
    print(f"Models: {list(models.keys())}")
    print(f"Attack strengths: {attack_strengths}")
    print(f"Episodes per config: {n_episodes}")
    
    total_episodes = len(models) * len(attack_strengths) * n_episodes
    gemini_models = [m for m in models.keys() if 'gemini' in m.lower()]
    gemini_episodes = len(gemini_models) * len(attack_strengths) * n_episodes
    
    print(f"Total episodes: {total_episodes}")
    if gemini_episodes > 0:
        print(f"  • Gemini episodes: {gemini_episodes} (~{gemini_episodes * 15} API calls)")
        print(f"  • Estimated time: ~{(gemini_episodes * 15 * 4.5) / 60:.1f} minutes")
    else:
        print(f"  • Ollama only - estimated time: ~{total_episodes * 15 / 60:.1f} minutes")
    print("=" * 80)
    
    results = []
    total_configs = len(models) * len(attack_strengths)
    config_idx = 0
    
    import time as time_module
    start_time = time_module.time()
    
    for model_name, client in models.items():
        print(f"\n{'='*80}")
        print(f"Testing Model: {model_name}")
        
        # Add size info
        model_sizes = {
            'gemini-2.5-flash-lite': 'Cloud (Google)',
            'phi3': '3.8GB',
            'tinyllama': '637MB (ultra-lightweight)',
            'qwen2:0.5b': '352MB (smallest)'
        }
        if model_name in model_sizes:
            print(f"Size: {model_sizes[model_name]}")
        
        print('='*80)
        
        for strength in attack_strengths:
            config_idx += 1
            
            # Calculate ETA
            if config_idx > 1:
                elapsed = time_module.time() - start_time
                avg_time_per_config = elapsed / (config_idx - 1)
                remaining_configs = total_configs - config_idx + 1
                eta_minutes = (avg_time_per_config * remaining_configs) / 60
                print(f"\n[Config {config_idx}/{total_configs}] {model_name} @ strength={strength:.1f} (ETA: {eta_minutes:.1f} min)")
            else:
                print(f"\n[Config {config_idx}/{total_configs}] {model_name} @ strength={strength:.1f}")
            
            # Generate attacks
            if strength > 0:
                docs = generate_episode_attack_config(
                    num_docs=5,
                    attack_strength=strength,
                    attack_type='direct',
                    seed=42
                )
            else:
                docs = []
            
            # Run episodes
            episode_summaries = []
            for ep in range(n_episodes):
                env = GridWorld(
                    grid_size=(10, 10),
                    max_steps=15 if DEMO_MODE else 25,  # Fast demo = 15 steps
                    documents=docs,
                    seed=42 + ep
                )
                
                try:
                    summary = run_episode(env, client, verbose=False)
                    episode_summaries.append(summary)
                    
                    status = "✓" if summary['goal_reached'] else "✗"
                    print(f"  Episode {ep+1}/{n_episodes}: {status} " +
                          f"Compromise={summary['compromise_rate']:.1%}")
                    
                except Exception as e:
                    print(f"  Episode {ep+1}/{n_episodes}: Error - {e}")
            
            # Calculate metrics
            if episode_summaries:
                avg_compromise = sum(s['compromise_rate'] for s in episode_summaries) / len(episode_summaries)
                avg_utility = sum(s['goal_reached'] for s in episode_summaries) / len(episode_summaries)
                avg_steps = sum(s['steps'] for s in episode_summaries) / len(episode_summaries)
                
                results.append({
                    'model': model_name,
                    'attack_strength': strength,
                    'asr': avg_compromise,
                    'utility': avg_utility,
                    'avg_steps': avg_steps,
                    'n_episodes': len(episode_summaries)
                })
                
                print(f"  → ASR={avg_compromise:.1%}, Utility={avg_utility:.1%}")
        
        # Show model stats
        stats = client.get_stats()
        print(f"\n{model_name} Stats:")
        print(f"  Total calls: {stats['total_calls']}")
        print(f"  Total tokens: {stats.get('total_tokens', 'N/A')}")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_path / f"multi_model_results_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    
    print("\n" + "=" * 80)
    print(f"Results saved to: {csv_path}")
    print("=" * 80)
    
    return df


def create_comparison_plots(df: pd.DataFrame, output_dir: str = "data/multi_model_comparison"):
    """
    Create publication-quality comparison plots with smart overlap handling.
    
    Args:
        df: Results DataFrame
        output_dir: Output directory
    """
    print("\n" + "=" * 80)
    print("CREATING COMPARISON PLOTS")
    print("=" * 80)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300
    
    # Color scheme with varying alpha for overlaps
    model_colors = {
        'gemini-2.5-flash-lite': '#FF6B6B',  # Red
        'phi3': '#4ECDC4',                    # Teal
        'tinyllama': '#FFA500',               # Orange
        'qwen2:0.5b': '#E74C3C'               # Dark Red (smallest, most vulnerable)
    }
    
    # Different marker styles to distinguish overlapping lines
    model_markers = {
        'gemini-2.5-flash-lite': 'o',  # Circle
        'phi3': 's',                    # Square
        'tinyllama': '^',               # Triangle up
        'qwen2:0.5b': 'D'               # Diamond
    }
    
    # Line styles for additional differentiation
    model_linestyles = {
        'gemini-2.5-flash-lite': '-',   # Solid
        'phi3': '-',                     # Solid
        'tinyllama': '--',               # Dashed
        'qwen2:0.5b': '-.'               # Dash-dot
    }
    
    import numpy as np
    
    # Plot 1: ASR vs Attack Strength with jitter and transparency
    print("\n[1/4] ASR comparison plot...")
    fig, ax = plt.subplots(figsize=(12, 7))
    
    models = df['model'].unique()
    num_models = len(models)
    
    # Add small horizontal jitter to separate overlapping points
    jitter_amount = 0.01
    
    for idx, model in enumerate(models):
        model_data = df[df['model'] == model].copy()
        
        # Apply jitter to x-coordinates for overlapping models
        x_values = model_data['attack_strength'].values
        if num_models > 2:
            # Offset each model slightly
            offset = (idx - num_models/2) * jitter_amount
            x_jittered = x_values + offset
        else:
            x_jittered = x_values
        
        color = model_colors.get(model, '#95A5A6')
        marker = model_markers.get(model, 'o')
        linestyle = model_linestyles.get(model, '-')
        
        # Plot with transparency and thicker lines
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
    
    # Better legend with more space
    ax.legend(loc='upper left', frameon=True, shadow=True, fontsize=11, 
              framealpha=0.95, edgecolor='black', fancybox=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.set_xlim(-0.05, 0.95)
    
    plt.tight_layout()
    fig.savefig(output_path / 'asr_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path / 'asr_comparison.png'}")
    plt.close()
    
    # Plot 2: Utility vs Attack Strength with offset markers
    print("\n[2/4] Utility comparison plot...")
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for idx, model in enumerate(models):
        model_data = df[df['model'] == model].copy()
        
        # Apply jitter
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
    ax.set_title('Utility Comparison: Model Performance\n(Different markers prevent overlap)', 
                 fontsize=14, fontweight='bold', pad=15)
    
    ax.legend(loc='upper right', frameon=True, shadow=True, fontsize=11,
              framealpha=0.95, edgecolor='black', fancybox=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.set_xlim(-0.05, 0.95)
    
    plt.tight_layout()
    fig.savefig(output_path / 'utility_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path / 'utility_comparison.png'}")
    plt.close()
    
    # Plot 3: Combined ASR & Utility side-by-side with different markers
    print("\n[3/4] Combined ASR & Utility plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # ASR subplot with jitter
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
                marker=marker, linestyle=linestyle, linewidth=2.5, markersize=9, 
                label=model, color=color, alpha=0.85,
                markeredgewidth=1.5, markeredgecolor='white')
    
    ax1.set_xlabel('Attack Strength', fontsize=12, fontweight='bold')
    ax1.set_ylabel('ASR', fontsize=12, fontweight='bold')
    ax1.set_title('Attack Success Rate', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10, framealpha=0.95, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax1.set_xlim(-0.05, 0.95)
    
    # Utility subplot with jitter
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
                marker=marker, linestyle=linestyle, linewidth=2.5, markersize=9, 
                label=model, color=color, alpha=0.85,
                markeredgewidth=1.5, markeredgecolor='white')
    
    ax2.set_xlabel('Attack Strength', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Utility', fontsize=12, fontweight='bold')
    ax2.set_title('Goal-Reaching Rate', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10, framealpha=0.95, shadow=True)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax2.set_xlim(-0.05, 0.95)
    
    plt.tight_layout()
    fig.savefig(output_path / 'combined_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path / 'combined_comparison.png'}")
    plt.close()
    
    # Plot 4: Model vulnerability ranking with pattern fills for ties
    print("\n[4/4] Vulnerability ranking plot...")
    
    # Calculate average ASR across all strengths for better ranking
    model_avg_asr = df.groupby('model')['asr'].mean().sort_values(ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    models_sorted = model_avg_asr.index.tolist()
    asr_values = model_avg_asr.values
    
    # Create bar colors with patterns for similar values
    colors = [model_colors.get(m, '#95A5A6') for m in models_sorted]
    
    # Detect ties (values within 1% are considered tied)
    hatches = []
    for i, val in enumerate(asr_values):
        if i > 0 and abs(val - asr_values[i-1]) < 0.01:
            hatches.append('///')  # Add pattern for tied values
        else:
            hatches.append('')
    
    bars = ax.barh(range(len(model_avg_asr)), asr_values, color=colors,
                   edgecolor='black', linewidth=1.5, alpha=0.85)
    
    # Apply hatching to bars with ties
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    
    # Set y-axis labels with model names and sizes
    model_labels = []
    for m in models_sorted:
        size_info = {
            'phi3': '3.8GB',
            'tinyllama': '637MB',
            'qwen2:0.5b': '352MB',
            'gemini-2.5-flash-lite': 'Cloud'
        }
        size = size_info.get(m, 'Unknown')
        model_labels.append(f"{m}\n({size})")
    
    ax.set_yticks(range(len(model_avg_asr)))
    ax.set_yticklabels(model_labels, fontsize=11)
    ax.set_xlabel('Average Attack Success Rate (Across All Strengths)', 
                  fontsize=12, fontweight='bold')
    ax.set_title('Model Vulnerability Ranking\n(Lower is More Secure, Patterns show ties)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    
    # Add value labels with offset for readability
    for i, (bar, val) in enumerate(zip(bars, asr_values)):
        width = bar.get_width()
        label_x = width + 0.02 if width < 0.5 else width - 0.02
        ha = 'left' if width < 0.5 else 'right'
        
        ax.text(label_x, i, f'{val:.1%}',
               ha=ha, va='center', fontsize=11, fontweight='bold',
               color='black' if width < 0.5 else 'white')
    
    plt.tight_layout()
    fig.savefig(output_path / 'vulnerability_ranking.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path / 'vulnerability_ranking.png'}")
    plt.close()
    
    print("\n" + "=" * 80)
    print("All plots created successfully!")
    print("=" * 80)


def print_summary_table(df: pd.DataFrame):
    """Print summary statistics table."""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        print(f"\n{model}:")
        print(f"  Average ASR: {model_data['asr'].mean():.1%}")
        print(f"  Average Utility: {model_data['utility'].mean():.1%}")
        print(f"  ASR at max strength: {model_data[model_data['attack_strength'] == model_data['attack_strength'].max()]['asr'].values[0]:.1%}")
        print(f"  Utility at max strength: {model_data[model_data['attack_strength'] == model_data['attack_strength'].max()]['utility'].values[0]:.1%}")
    
    print("\n" + "=" * 80)


def main():
    """Main execution."""
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 15 + "MULTI-MODEL COMPARISON EXPERIMENT" + " " * 30 + "█")
    
    if USE_GEMINI:
        print("█" + " " * 10 + "Gemini (Cloud) vs Ollama (Local) Robustness" + " " * 24 + "█")
    else:
        print("█" + " " * 10 + "Ollama: Phi-3 (3.8GB) vs TinyLlama (637MB)" + " " * 26 + "█")
    
    print("█" + " " * 78 + "█")
    print("█" * 80)
    
    # Setup models
    models = setup_models()
    
    if len(models) < 2:
        print("\n⚠️  Need at least 2 models to run comparison!")
        
        if USE_GEMINI:
            print("\nMake sure:")
            print("  1. GEMINI_API_KEY is set in .env")
            print("  2. Ollama is running (ollama serve)")
            print("  3. Models are downloaded:")
            print("     - ollama pull phi3")
            print("     - ollama pull tinyllama")
        else:
            print("\nMake sure:")
            print("  1. Ollama is running (ollama serve)")
            print("  2. Models are downloaded:")
            print("     - ollama pull phi3")
            print("     - ollama pull tinyllama")
        return
    
    # Run experiment
    print(f"\n✓ {len(models)} models ready. Starting experiment...")
    
    if DEMO_MODE:
        print("\n⚡ DEMO MODE: 1 episode × 5 strengths = 5 runs per model")
        print("   Perfect for project proposal - fast results with good visualizations!")
    else:
        if USE_GEMINI:
            print("\n📊 RESEARCH MODE: 3 episodes per config (Gemini quota-safe)")
        else:
            print("\n📊 RESEARCH MODE: 3 episodes per config (full statistical data)")

    
    results = run_comparison_experiment(
        models=models,
        attack_strengths=[0.0, 0.2, 0.4, 0.6, 0.8],  # 5 attack levels with even 0.2 increments
        n_episodes=1 if DEMO_MODE else 3  # Demo = 1, Research = 3
    )
    
    # Create plots
    create_comparison_plots(results)
    
    # Print summary
    print_summary_table(results)
    
    print("\n" + "=" * 80)
    print("✓ EXPERIMENT COMPLETE!")
    print("=" * 80)
    print("\nCheck outputs:")
    print("  • Results CSV: data/multi_model_comparison/multi_model_results_*.csv")
    print("  • Plots: data/multi_model_comparison/*.png")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
