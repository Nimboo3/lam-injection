"""
Batch episode runner for experiments.

Runs multiple episodes with same configuration and collects results.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
from typing import List, Dict, Any, Optional
from datetime import datetime

from envs.gridworld import GridWorld
from llm.wrapper import LLMClient
from runner.controller import run_episode


def run_batch_episodes(
    env_config: Dict[str, Any],
    llm_config: Dict[str, Any],
    n_episodes: int = 10,
    max_steps: int = 50,
    log_dir: Optional[str] = None,
    verbose: bool = False
) -> List[Dict[str, Any]]:
    """
    Run multiple episodes with same configuration.
    
    Args:
        env_config: GridWorld configuration dict
        llm_config: LLM client configuration dict
        n_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        log_dir: Directory for episode logs (None = no logging)
        verbose: Print episode progress
    
    Returns:
        List of episode summary dicts
    """
    summaries = []
    
    for i in range(n_episodes):
        if verbose:
            print(f"Running episode {i+1}/{n_episodes}...", end=" ")
        
        # Create fresh environment
        env = GridWorld(**env_config)
        env.reset()
        
        # Create LLM client
        client = LLMClient(**llm_config)
        
        # Setup logging if requested
        logger = None
        if log_dir:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            log_file = log_path / f"episode_{i:03d}.jsonl"
            logger = open(log_file, "w")
        
        try:
            # Run episode
            summary = run_episode(
                env=env,
                llm_client=client,
                max_steps=max_steps,
                logger=logger,
                verbose=False
            )
            
            # Add batch metadata
            summary['batch_index'] = i
            summary['batch_size'] = n_episodes
            
            summaries.append(summary)
            
            if verbose:
                status = "✓ Goal" if summary['goal_reached'] else "✗ Timeout"
                print(f"{status} (steps={summary['steps']}, " 
                      f"compromised={summary['compromised_steps']}/{summary['steps']})")
        
        finally:
            if logger:
                logger.close()
    
    return summaries


def save_batch_results(
    summaries: List[Dict[str, Any]],
    output_file: str,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Save batch results to JSON file.
    
    Args:
        summaries: List of episode summaries
        output_file: Path to output JSON file
        metadata: Optional metadata dict to include
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = {
        'metadata': metadata or {},
        'timestamp': datetime.now().isoformat(),
        'n_episodes': len(summaries),
        'summaries': summaries
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)


def load_batch_results(input_file: str) -> Dict[str, Any]:
    """
    Load batch results from JSON file.
    
    Args:
        input_file: Path to input JSON file
    
    Returns:
        Results dict with metadata and summaries
    """
    with open(input_file, 'r') as f:
        return json.load(f)


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Batch Runner Demo")
    print("=" * 70)
    print()
    
    # Configuration
    env_config = {
        'grid_size': (10, 10),
        'max_steps': 30,
        'documents': [],  # No attacks for demo
        'seed': 42
    }
    
    llm_config = {
        'model_name': 'mock',
        'injection_threshold': 0.5,
        'seed': 42
    }
    
    # Run batch
    print("Running 5 episodes...")
    print("-" * 70)
    
    summaries = run_batch_episodes(
        env_config=env_config,
        llm_config=llm_config,
        n_episodes=5,
        max_steps=30,
        verbose=True
    )
    
    # Summary stats
    print()
    print("=" * 70)
    print("Batch Summary")
    print("=" * 70)
    
    goal_reached = sum(1 for s in summaries if s['goal_reached'])
    total_steps = sum(s['steps'] for s in summaries)
    total_compromised = sum(s['compromised_steps'] for s in summaries)
    
    print(f"Episodes: {len(summaries)}")
    print(f"Goal reached: {goal_reached}/{len(summaries)} ({goal_reached/len(summaries):.1%})")
    print(f"Avg steps: {total_steps/len(summaries):.1f}")
    print(f"Total compromised: {total_compromised}/{total_steps} ({total_compromised/total_steps:.1%})")
    
    print()
    print("=" * 70)
    print("Demo complete!")
