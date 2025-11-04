"""
Quick test to verify Gemini API connection and model availability.

Run this before running the full experiment to catch issues early.
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

def test_api_key():
    """Check if API key is set."""
    print("=" * 70)
    print("Step 1: Checking API Key")
    print("=" * 70)
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not found!")
        print("\nSet it in .env file:")
        print("  GEMINI_API_KEY=your_key_here")
        return False
    
    print(f"✓ API key found (length: {len(api_key)} chars)")
    return True


def test_gemini_client():
    """Test Gemini client initialization and basic call."""
    print("\n" + "=" * 70)
    print("Step 2: Testing Gemini Client")
    print("=" * 70)
    
    try:
        from llm.gemini_client import GeminiClient
        
        # Test only gemini-2.5-flash-lite
        print("\n[1/1] Testing gemini-2.5-flash-lite...")
        client = GeminiClient(model_name="gemini-2.5-flash-lite")
        
        test_prompt = "What is 2+2? Answer with just the number."
        print(f"  Sending test prompt...")
        response = client.generate(test_prompt, max_tokens=50, temperature=0.0)
        print(f"  ✓ Response: {response}")
        print(f"  ✓ Stats: {client.get_stats()}")
        
        print("\n✅ Gemini model working!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ollama():
    """Test Ollama connection."""
    print("\n" + "=" * 70)
    print("Step 3: Testing Ollama (Optional)")
    print("=" * 70)
    
    try:
        import subprocess
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            print("✓ Ollama service is running")
            print("\nAvailable models:")
            print(result.stdout)
            
            # Check for required models
            models = result.stdout.lower()
            has_phi3 = "phi3" in models
            has_llama = "llama3.2" in models
            
            if has_phi3 and has_llama:
                print("✅ Both phi3 and llama3.2 are ready!")
            else:
                print("\n⚠️  Some models missing:")
                if not has_phi3:
                    print("  - phi3 not found. Download with: ollama pull phi3")
                if not has_llama:
                    print("  - llama3.2 not found. Download with: ollama pull llama3.2")
            
            return True
        else:
            print("⚠️  Ollama service not responding")
            print("  Start it with: ollama serve")
            return False
            
    except FileNotFoundError:
        print("⚠️  Ollama not installed")
        print("  Install with: winget install Ollama.Ollama")
        return False
    except Exception as e:
        print(f"⚠️  Error checking Ollama: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + " " * 15 + "GEMINI API CONNECTION TEST" + " " * 26 + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    print()
    
    results = []
    
    # Test API key
    results.append(("API Key", test_api_key()))
    
    # Test Gemini
    if results[0][1]:  # Only if API key exists
        results.append(("Gemini API", test_gemini_client()))
    else:
        print("\n⏭️  Skipping Gemini test (no API key)")
        results.append(("Gemini API", False))
    
    # Test Ollama
    results.append(("Ollama", test_ollama()))
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for name, passed in results:
        status = "✓" if passed else "✗"
        print(f"  {status} {name}")
    
    passed = sum(1 for _, p in results if p)
    total = len(results)
    
    print(f"\n{passed}/{total} tests passed")
    print("=" * 70)
    
    if results[1][1]:  # Gemini working
        print("\n✅ Ready to run experiments!")
        print("\nRun:")
        print("  python experiments/multi_model_comparison.py")
    else:
        print("\n⚠️  Fix issues above before running experiments")
    
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(1)
