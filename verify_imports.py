#!/usr/bin/env python3
"""
Quick verification script to test if imports work correctly.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    print("=" * 60)
    print("Testing imports...")
    print("=" * 60)
    
    try:
        print("\n[1/3] Importing envs.gridworld...")
        from envs.gridworld import GridWorld
        print("      ✓ Success!")
        
        print("\n[2/3] Creating GridWorld instance...")
        env = GridWorld(grid_size=(5, 5), seed=42)
        print("      ✓ Success!")
        
        print("\n[3/3] Testing reset and step...")
        obs = env.reset()
        obs, reward, done, info = env.step(GridWorld.RIGHT)
        print("      ✓ Success!")
        
        print("\n" + "=" * 60)
        print("All imports working correctly! ✓")
        print("=" * 60)
        print("\nYou can now run:")
        print("  - pytest tests/test_gridworld.py -v")
        print("  - python scripts/demo_gridworld.py")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nTry running: pip install -e .")
        return 1


if __name__ == "__main__":
    sys.exit(main())
