"""
System Readiness Check for Multi-Model Research Experiment

Validates that all components are properly configured before running experiments.
"""

import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()


def print_header(title):
    """Print section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def check_mark(passed):
    """Return check mark or X based on status."""
    return "✓" if passed else "✗"


def check_python_packages():
    """Check if required Python packages are installed."""
    print_header("Python Packages Check")
    
    required = [
        'numpy',
        'pandas', 
        'matplotlib',
        'seaborn',
        'scipy',
        'requests',
        'gym'
    ]
    
    all_installed = True
    for package in required:
        try:
            __import__(package)
            print(f"  {check_mark(True)} {package:20s} installed")
        except ImportError:
            print(f"  {check_mark(False)} {package:20s} MISSING")
            all_installed = False
    
    if not all_installed:
        print("\n  Install missing packages with:")
        print("    pip install -r requirements.txt")
    
    return all_installed


def check_gemini_api():
    """Check if Gemini API key is configured."""
    print_header("Gemini API Configuration")
    
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print(f"  {check_mark(False)} GEMINI_API_KEY not found")
        print("\n  Set your API key in .env file:")
        print("    GEMINI_API_KEY=your_key_here")
        return False
    
    if len(api_key) < 20:
        print(f"  {check_mark(False)} GEMINI_API_KEY looks invalid (too short)")
        return False
    
    print(f"  {check_mark(True)} GEMINI_API_KEY configured")
    print(f"  Key length: {len(api_key)} characters")
    
    # Test API connection (optional)
    try:
        from llm.gemini_client import GeminiClient
        print("\n  Testing connection...")
        client = GeminiClient(model_name="gemini-1.5-flash")
        print(f"  {check_mark(True)} GeminiClient initialized")
        return True
    except Exception as e:
        print(f"  {check_mark(False)} Connection test failed: {e}")
        return False


def check_ollama_installation():
    """Check if Ollama is installed."""
    print_header("Ollama Installation")
    
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"  {check_mark(True)} Ollama installed: {version}")
            return True
        else:
            print(f"  {check_mark(False)} Ollama command failed")
            return False
    except FileNotFoundError:
        print(f"  {check_mark(False)} Ollama not installed")
        print("\n  Install with:")
        print("    winget install Ollama.Ollama")
        return False
    except Exception as e:
        print(f"  {check_mark(False)} Error checking Ollama: {e}")
        return False


def check_ollama_service():
    """Check if Ollama service is running."""
    print_header("Ollama Service Status")
    
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            print(f"  {check_mark(True)} Ollama service is running")
            return True
        else:
            print(f"  {check_mark(False)} Ollama service not responding")
            print("\n  Start the service with:")
            print("    ollama serve")
            return False
    except Exception as e:
        print(f"  {check_mark(False)} Cannot connect to Ollama: {e}")
        print("\n  Start the service with:")
        print("    ollama serve")
        return False


def check_ollama_models():
    """Check if required Ollama models are downloaded."""
    print_header("Ollama Models")
    
    required_models = ["phi3", "llama3.2"]
    alternative_models = ["tinyllama", "gemma2:2b"]
    
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode != 0:
            print(f"  {check_mark(False)} Cannot list models (service not running?)")
            return False
        
        output = result.stdout.lower()
        
        # Check required models
        all_found = True
        for model in required_models:
            found = model in output
            print(f"  {check_mark(found)} {model:20s} {'downloaded' if found else 'NOT FOUND'}")
            if not found:
                all_found = False
        
        if not all_found:
            print("\n  Download recommended models:")
            for model in required_models:
                if model not in output:
                    print(f"    ollama pull {model}")
            
            print("\n  Or use alternative (smaller) models:")
            for model in alternative_models:
                size = "637MB" if model == "tinyllama" else "1.6GB"
                print(f"    ollama pull {model}  # {size}")
        
        return all_found
        
    except Exception as e:
        print(f"  {check_mark(False)} Error checking models: {e}")
        return False


def check_project_structure():
    """Check if project files exist."""
    print_header("Project Structure")
    
    required_files = [
        "experiments/multi_model_comparison.py",
        "llm/gemini_client.py",
        "llm/ollama_client.py",
        "llm/wrapper.py",
        "scripts/setup_ollama_models.py",
        "requirements.txt"
    ]
    
    all_exist = True
    for file_path in required_files:
        exists = Path(file_path).exists()
        print(f"  {check_mark(exists)} {file_path}")
        if not exists:
            all_exist = False
    
    return all_exist


def check_output_directory():
    """Check if output directory exists or can be created."""
    print_header("Output Directory")
    
    output_dir = Path("data/multi_model_comparison")
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"  {check_mark(True)} Output directory ready: {output_dir}")
        
        # Check write permissions
        test_file = output_dir / "test.txt"
        test_file.write_text("test")
        test_file.unlink()
        print(f"  {check_mark(True)} Write permissions verified")
        
        return True
    except Exception as e:
        print(f"  {check_mark(False)} Cannot create/write to output directory: {e}")
        return False


def run_all_checks():
    """Run all system checks."""
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + " " * 15 + "SYSTEM READINESS CHECK" + " " * 30 + "█")
    print("█" + " " * 10 + "Multi-Model Research Experiment" + " " * 28 + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    
    checks = {
        "Python Packages": check_python_packages(),
        "Gemini API": check_gemini_api(),
        "Ollama Installation": check_ollama_installation(),
        "Ollama Service": check_ollama_service(),
        "Ollama Models": check_ollama_models(),
        "Project Structure": check_project_structure(),
        "Output Directory": check_output_directory()
    }
    
    # Summary
    print_header("SUMMARY")
    
    passed = sum(checks.values())
    total = len(checks)
    
    print()
    for name, result in checks.items():
        print(f"  {check_mark(result)} {name}")
    
    print("\n" + "=" * 70)
    print(f"  {passed}/{total} checks passed")
    print("=" * 70)
    
    if passed == total:
        print("\n  ✅ ALL CHECKS PASSED!")
        print("\n  You're ready to run the experiment:")
        print("    python experiments/multi_model_comparison.py")
        print("\n  Or use the helper script:")
        print("    run_research_experiment.bat")
    else:
        print("\n  ⚠️  SOME CHECKS FAILED")
        print("\n  Please fix the issues above before running the experiment.")
        print("\n  See RESEARCH_QUICKSTART.md for detailed setup instructions.")
    
    print("\n" + "=" * 70)
    
    return passed == total


def main():
    """Main execution."""
    try:
        success = run_all_checks()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nCheck interrupted by user.")
        sys.exit(1)


if __name__ == "__main__":
    main()
