# Chunk 7 — Experiment Harness & Metrics ✅ COMPLETE

**Created Files**:
- `experiments/batch.py`: Batch episode runner (200+ lines)
- `experiments/runner.py`: Parameter grid experiment runner (300+ lines)
- `experiments/__init__.py`: Package exports
- `analysis/metrics.py`: Metrics calculation module (400+ lines)
- `tests/test_experiments.py`: Comprehensive tests (500+ lines, 23 tests)
- `scripts/demo_experiments.py`: Interactive demo

**Key Components**:

1. **Batch Runner** (`experiments/batch.py`):
   - `run_batch_episodes(env_config, llm_config, n_episodes, max_steps, log_dir, verbose)`: Run multiple episodes with same config
   - `save_batch_results(summaries, output_file, metadata)`: Save results to JSON
   - `load_batch_results(input_file)`: Load results from JSON
   - Returns: List of episode summary dicts with batch metadata

2. **Experiment Runner** (`experiments/runner.py`):
   - `ExperimentConfig`: Dataclass for experiment configuration
   - `run_experiment_grid(config, verbose)`: Execute parameter grid sweep
   - Returns: pandas DataFrame with results
   - Saves: CSV with all results, JSON with configuration
   - Grid parameters: grid_sizes, models, attack_strengths, attack_types, defenses, etc.

3. **Metrics Module** (`analysis/metrics.py`):
   - `calculate_asr(episodes)`: Attack Success Rate (fraction of compromised steps)
   - `calculate_utility(episodes)`: Goal-reaching rate
   - `calculate_steps_to_compromise(episodes)`: Average steps until first compromise
   - `calculate_defense_effectiveness(baseline_asr, defended_asr)`: Defense performance
   - `aggregate_metrics(results_df, group_by)`: Aggregate with mean/std/median
   - `compute_pareto_frontier(results_df, objective1, objective2)`: Multi-objective optimization
   - `calculate_transfer_score(source_results, target_results)`: Attack transferability
   - `print_metrics_summary(results_df)`: Formatted summary output

**Testing Commands**:
```bash
# Run all experiment tests (23 tests)
pytest tests/test_experiments.py -v

# Run demo
python scripts/demo_experiments.py
```

**Expected Output**:
- All 23 tests pass in ~0.86 seconds
- Demo runs 30 episodes across 6 configurations in ~5 seconds
- CSV and JSON saved to `data/experiments/`

**Integration**: Combines all previous chunks (GridWorld, LLM, Controller, Attacks, Defenses) to run systematic parameter sweeps and calculate robustness metrics.
