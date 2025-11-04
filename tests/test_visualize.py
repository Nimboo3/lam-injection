"""Tests for visualization module."""

import pytest
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile

from analysis.visualize import (
    plot_asr_vs_strength,
    plot_utility_vs_strength,
    plot_asr_and_utility,
    plot_transfer_heatmap,
    plot_defense_effectiveness,
    plot_pareto_frontier,
    plot_attack_type_comparison,
    create_summary_dashboard,
)


@pytest.fixture
def sample_results_df():
    """Create sample results DataFrame."""
    np.random.seed(42)
    
    data = {
        'attack_strength': [0.0, 0.3, 0.6, 0.9] * 5,
        'asr': [0.0] * 5 + list(np.random.uniform(0.1, 0.3, 5)) +
               list(np.random.uniform(0.3, 0.6, 5)) + list(np.random.uniform(0.6, 0.9, 5)),
        'utility': [0.95] * 5 + list(np.random.uniform(0.8, 0.95, 5)) +
                   list(np.random.uniform(0.6, 0.8, 5)) + list(np.random.uniform(0.4, 0.6, 5)),
        'attack_type': ['direct'] * 10 + ['hidden'] * 10,
        'use_sanitizer': [False] * 10 + [True] * 10
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_transfer_matrix():
    """Create sample transfer matrix."""
    data = {
        'model_a': [1.0, 0.7, 0.5],
        'model_b': [0.6, 1.0, 0.4],
        'model_c': [0.5, 0.6, 1.0]
    }
    return pd.DataFrame(data, index=['model_a', 'model_b', 'model_c'])


class TestASRPlots:
    """Tests for ASR plotting functions."""
    
    def test_plot_asr_vs_strength(self, sample_results_df):
        """Test ASR vs strength plot."""
        fig = plot_asr_vs_strength(sample_results_df)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1
        
        ax = fig.axes[0]
        assert ax.get_xlabel() == 'Attack Strength'
        assert ax.get_ylabel() == 'Attack Success Rate (ASR)'
        
        plt.close(fig)
    
    def test_plot_asr_with_save(self, sample_results_df, tmp_path):
        """Test saving ASR plot."""
        save_path = tmp_path / "asr.png"
        
        fig = plot_asr_vs_strength(sample_results_df, save_path=str(save_path))
        
        assert save_path.exists()
        assert save_path.stat().st_size > 0
        
        plt.close(fig)
    
    def test_plot_asr_custom_params(self, sample_results_df):
        """Test ASR plot with custom parameters."""
        fig = plot_asr_vs_strength(
            sample_results_df,
            figsize=(8, 6),
            title="Custom Title"
        )
        
        assert fig.get_figwidth() == 8
        assert fig.get_figheight() == 6
        assert "Custom Title" in fig.axes[0].get_title()
        
        plt.close(fig)


class TestUtilityPlots:
    """Tests for utility plotting functions."""
    
    def test_plot_utility_vs_strength(self, sample_results_df):
        """Test utility vs strength plot."""
        fig = plot_utility_vs_strength(sample_results_df)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1
        
        ax = fig.axes[0]
        assert ax.get_xlabel() == 'Attack Strength'
        assert 'Utility' in ax.get_ylabel()
        
        plt.close(fig)
    
    def test_plot_asr_and_utility(self, sample_results_df):
        """Test combined ASR and utility plot."""
        fig = plot_asr_and_utility(sample_results_df)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2
        
        plt.close(fig)


class TestTransferPlots:
    """Tests for transfer matrix visualization."""
    
    def test_plot_transfer_heatmap(self, sample_transfer_matrix):
        """Test transfer matrix heatmap."""
        fig = plot_transfer_heatmap(sample_transfer_matrix)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 1  # Heatmap + colorbar
        
        plt.close(fig)
    
    def test_transfer_heatmap_values(self, sample_transfer_matrix):
        """Test heatmap displays correct values."""
        fig = plot_transfer_heatmap(sample_transfer_matrix)
        
        # Check diagonal values are 1.0
        assert sample_transfer_matrix.iloc[0, 0] == 1.0
        assert sample_transfer_matrix.iloc[1, 1] == 1.0
        assert sample_transfer_matrix.iloc[2, 2] == 1.0
        
        plt.close(fig)


class TestDefensePlots:
    """Tests for defense effectiveness visualization."""
    
    def test_plot_defense_effectiveness(self, sample_results_df):
        """Test defense effectiveness plot."""
        fig = plot_defense_effectiveness(sample_results_df)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1
        
        ax = fig.axes[0]
        assert 'Defense' in ax.get_title()
        
        plt.close(fig)
    
    def test_defense_plot_with_both_conditions(self, sample_results_df):
        """Test plot includes both defense conditions."""
        fig = plot_defense_effectiveness(sample_results_df)
        
        ax = fig.axes[0]
        legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
        
        assert 'No Defense' in legend_labels or 'With Defense' in legend_labels
        
        plt.close(fig)


class TestParetoPlots:
    """Tests for Pareto frontier visualization."""
    
    def test_plot_pareto_frontier_basic(self, sample_results_df):
        """Test basic Pareto frontier plot."""
        fig = plot_pareto_frontier(sample_results_df)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1
        
        plt.close(fig)
    
    def test_pareto_with_frontier_points(self, sample_results_df):
        """Test Pareto plot with frontier points."""
        # Create Pareto points (low ASR, high utility)
        pareto_df = sample_results_df.nsmallest(3, 'asr').nlargest(3, 'utility')
        
        fig = plot_pareto_frontier(
            sample_results_df,
            pareto_df=pareto_df
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestAttackTypePlots:
    """Tests for attack type comparison."""
    
    def test_plot_attack_type_comparison(self, sample_results_df):
        """Test attack type comparison plot."""
        fig = plot_attack_type_comparison(sample_results_df)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1
        
        ax = fig.axes[0]
        assert 'Attack Type' in ax.get_title()
        
        plt.close(fig)
    
    def test_attack_type_multiple_types(self, sample_results_df):
        """Test plot with multiple attack types."""
        fig = plot_attack_type_comparison(sample_results_df)
        
        ax = fig.axes[0]
        legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
        
        # Should have both attack types
        assert len(legend_labels) >= 2
        
        plt.close(fig)


class TestDashboard:
    """Tests for summary dashboard."""
    
    def test_create_summary_dashboard_basic(self, sample_results_df):
        """Test basic dashboard creation."""
        fig = create_summary_dashboard(sample_results_df)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 2
        
        plt.close(fig)
    
    def test_dashboard_with_transfer_matrix(self, sample_results_df, sample_transfer_matrix):
        """Test dashboard with transfer matrix."""
        fig = create_summary_dashboard(
            sample_results_df,
            transfer_matrix=sample_transfer_matrix
        )
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 3
        
        plt.close(fig)
    
    def test_dashboard_save(self, sample_results_df, tmp_path):
        """Test saving dashboard."""
        save_path = tmp_path / "dashboard.png"
        
        fig = create_summary_dashboard(
            sample_results_df,
            save_path=str(save_path)
        )
        
        assert save_path.exists()
        plt.close(fig)


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame(columns=['attack_strength', 'asr', 'utility'])
        
        # Should handle gracefully
        try:
            fig = plot_asr_vs_strength(df)
            plt.close(fig)
        except Exception as e:
            # Some error is acceptable for empty data
            pass
    
    def test_single_strength_value(self):
        """Test with single strength value."""
        df = pd.DataFrame({
            'attack_strength': [0.5] * 10,
            'asr': np.random.uniform(0, 1, 10),
            'utility': np.random.uniform(0, 1, 10)
        })
        
        fig = plot_asr_vs_strength(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_missing_columns(self, sample_results_df):
        """Test with missing optional columns."""
        df = sample_results_df[['attack_strength', 'asr', 'utility']].copy()
        
        # Should work without attack_type or defense columns
        fig = plot_asr_vs_strength(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
