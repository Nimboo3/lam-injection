"""
Integration demo for Chunks 1-5.

Shows how all components work together:
- Chunk 5: Attack generator creates documents
- Chunk 2: GridWorld environment
- Chunk 3: LLM wrapper (mock)
- Chunk 4: Controller runs episode with logging
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from attacks.generator import generate_episode_attack_config
from envs.gridworld import GridWorld
from llm.wrapper import LLMClient
from runner.controller import run_episode


def main():
    print("=" * 80)
    print("INTEGRATION DEMO: Chunks 1-5")
    print("=" * 80)
    print()
    
    # CHUNK 5: Generate attack configuration
    print("[Chunk 5] Generating attack documents...")
    docs = generate_episode_attack_config(
        num_docs=5,
        attack_strength=0.7,
        distribution="mixed",
        attack_type="direct",
        seed=42
    )
    print(f"  ✓ Generated {len(docs)} documents")
    strengths = [f'{d["attack_strength"]:.2f}' for d in docs]
    print(f"  ✓ Attack strengths: {strengths}")
    print()
    
    # CHUNK 2: Create GridWorld with generated attacks
    print("[Chunk 2] Creating GridWorld environment...")
    env = GridWorld(
        grid_size=(10, 10),
        max_steps=30,
        documents=docs,
        seed=42
    )
    obs = env.reset()
    print(f"  ✓ Environment initialized")
    print(f"  ✓ Agent: {obs['agent_pos']}, Goal: {obs['goal_pos']}")
    print()
    
    # CHUNK 3: Create LLM client (mock)
    print("[Chunk 3] Creating LLM client...")
    client = LLMClient(
        model_name="mock",
        injection_threshold=0.5,
        seed=42
    )
    print(f"  ✓ Mock LLM client ready")
    print(f"  ✓ Injection threshold: 0.5")
    print()
    
    # CHUNK 4: Run episode with controller
    print("[Chunk 4] Running episode with logging...")
    print("-" * 80)
    
    log_file = Path("data/run_logs/integration_demo.jsonl")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(log_file, "w") as logger:
        summary = run_episode(
            env=env,
            llm_client=client,
            max_steps=30,
            logger=logger,
            verbose=True
        )
    
    print("-" * 80)
    print()
    
    # Summary
    print("=" * 80)
    print("INTEGRATION SUMMARY")
    print("=" * 80)
    print(f"Episode ID: {summary['episode_id']}")
    print(f"Total Steps: {summary['steps']}")
    print(f"Goal Reached: {summary['goal_reached']}")
    print(f"Total Reward: {summary['total_reward']:.3f}")
    print(f"Compromised Steps: {summary['compromised_steps']}/{summary['steps']}")
    print(f"Compromise Rate: {summary['compromise_rate']:.1%}")
    print(f"Invalid Actions: {summary['invalid_actions']}")
    print()
    print(f"Log file: {log_file}")
    print()
    
    # Show which chunks were tested
    print("✓ All 5 chunks integrated successfully:")
    print("  ✓ Chunk 1: Project scaffold")
    print("  ✓ Chunk 2: GridWorld environment")
    print("  ✓ Chunk 3: LLM wrapper (mock)")
    print("  ✓ Chunk 4: Controller & logging")
    print("  ✓ Chunk 5: Attack generator")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
