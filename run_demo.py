#!/usr/bin/env python3
"""
Scaffold verification script for the Agentic Prompt-Injection Robustness Benchmark.

This script confirms that the project scaffold is correctly set up.
"""

import sys
import os


def main():
    """Print scaffold status and verify basic setup."""
    print("=" * 60)
    print("=== Agentic Prompt-Injection Robustness Benchmark ===")
    print("=" * 60)
    print()
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 10):
        print("⚠ Warning: Python 3.10+ recommended")
    else:
        print("✓ Python version compatible")
    
    print()
    
    # Check for key files
    expected_files = [
        "README.md",
        "requirements.txt",
        ".gitignore",
        ".env.example",
        "CONTEXT.md"
    ]
    
    print("Checking scaffold files:")
    all_present = True
    for filename in expected_files:
        exists = os.path.exists(filename)
        status = "✓" if exists else "✗"
        print(f"  {status} {filename}")
        if not exists:
            all_present = False
    
    print()
    
    # Check for .env (optional)
    if os.path.exists(".env"):
        print("✓ .env file found (API key configured)")
    else:
        print("ℹ .env file not found (using mock LLM mode)")
    
    print()
    print("=" * 60)
    
    if all_present:
        print("Status: Scaffold ready ✓")
        print("=" * 60)
        print()
        print("Next steps:")
        print("  1. Activate virtual environment: .venv\\Scripts\\activate")
        print("  2. Install dependencies: pip install -r requirements.txt")
        print("  3. (Optional) Configure .env with GEMINI_API_KEY")
        print("  4. Continue with Chunk 2 (GridWorld environment)")
        return 0
    else:
        print("Status: Scaffold incomplete ✗")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
