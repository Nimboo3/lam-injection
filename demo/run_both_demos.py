"""
Automated Demo Runner

Runs both baseline and defense experiments automatically.
"""

import sys
import subprocess
from pathlib import Path
import time

def run_command(cmd, description):
    """Run a command and handle errors."""
    print("\n" + "="*80)
    print(f"  {description}")
    print("="*80 + "\n")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\n❌ ERROR: {description} failed!")
        sys.exit(1)
    
    print(f"\n✓ {description} complete!")
    return True

def main():
    """Run both demo experiments."""
    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + "AUTOMATED DEMO RUNNER".center(78) + "█")
    print("█" + " "*78 + "█")
    print("█"*80)
    print()
    print("This script will run BOTH experiments:")
    print("  1. Baseline (no defense)")
    print("  2. With defenses enabled")
    print()
    print("Total estimated time: ~7 minutes")
    print()
    
    input("Press ENTER to start...")
    
    # Get script directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Change to project root
    import os
    os.chdir(project_root)
    
    # Step 1: Run baseline
    print("\n" + "█"*80)
    print("█" + "STEP 1/2: Baseline Experiment (No Defense)".center(78) + "█")
    print("█"*80)
    
    # Read original script and ensure USE_DEFENSE = False
    demo_script = script_dir / "run_demo.py"
    with open(demo_script, 'r', encoding='utf-8') as f:
        baseline_code = f.read()
    
    # Ensure it's set to False
    if "USE_DEFENSE = True" in baseline_code:
        baseline_code = baseline_code.replace("USE_DEFENSE = True", "USE_DEFENSE = False")
        with open(demo_script, 'w', encoding='utf-8') as f:
            f.write(baseline_code)
        print("✓ Reset USE_DEFENSE to False")
    
    run_command(
        f"python {demo_script}",
        "Baseline Experiment"
    )
    
    print("\n⏳ Waiting 3 seconds before starting defense demo...")
    time.sleep(3)
    
    # Step 2: Run with defense
    print("\n" + "█"*80)
    print("█" + "STEP 2/2: Defense Experiment (With Sanitizer + Detector)".center(78) + "█")
    print("█"*80)
    
    # Read and modify for defense
    with open(demo_script, 'r', encoding='utf-8') as f:
        defense_code = f.read()
    
    defense_code = defense_code.replace("USE_DEFENSE = False", "USE_DEFENSE = True")
    
    with open(demo_script, 'w', encoding='utf-8') as f:
        f.write(defense_code)
    
    print("✓ Enabled defenses (USE_DEFENSE = True)")
    
    run_command(
        f"python {demo_script}",
        "Defense Experiment"
    )
    
    # Reset to baseline
    with open(demo_script, 'w', encoding='utf-8') as f:
        f.write(baseline_code)
    print("\n✓ Reset script to baseline mode")
    
    # Success message
    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + "SUCCESS! Both experiments complete".center(78) + "█")
    print("█" + " "*78 + "█")
    print("█"*80)
    print()
    print("📊 Results saved to:")
    print("   • data/demo_results/no_defense/")
    print("   • data/demo_results/with_defense/")
    print()
    print("📈 Next steps:")
    print("   1. View plots: Open data/demo_results/ folder")
    print("   2. Compare results: Run notebooks/demo_comparison.ipynb")
    print()
    print("="*80)
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        sys.exit(1)
