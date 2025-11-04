"""
Helper script to download and verify Ollama models for research.

Downloads recommended models for 16GB RAM + Intel UHD hardware:
- phi3 (3.8GB)
- llama3.2 (2GB)
"""

import subprocess
import sys
import time


def check_ollama_installed():
    """Check if Ollama is installed."""
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print(f"✓ Ollama installed: {result.stdout.strip()}")
            return True
        return False
    except Exception as e:
        print(f"✗ Ollama not found: {e}")
        return False


def check_ollama_running():
    """Check if Ollama service is running."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print("✓ Ollama service is running")
            return True
        else:
            print("✗ Ollama service not running")
            print("  Start it with: ollama serve")
            return False
    except Exception as e:
        print(f"✗ Cannot connect to Ollama: {e}")
        print("  Start it with: ollama serve")
        return False


def list_downloaded_models():
    """List currently downloaded Ollama models."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print("\n" + "=" * 60)
            print("Currently Downloaded Models:")
            print("=" * 60)
            print(result.stdout)
            
            # Parse to check if our models exist
            models = result.stdout.lower()
            has_phi3 = "phi3" in models
            has_llama = "llama3.2" in models
            
            return has_phi3, has_llama
        return False, False
    except Exception as e:
        print(f"Error listing models: {e}")
        return False, False


def download_model(model_name: str, size: str):
    """
    Download an Ollama model.
    
    Args:
        model_name: Name of the model
        size: Human-readable size
    """
    print("\n" + "=" * 60)
    print(f"Downloading {model_name} ({size})")
    print("=" * 60)
    print("This may take several minutes depending on your internet speed...")
    
    try:
        # Run ollama pull with real-time output
        process = subprocess.Popen(
            ["ollama", "pull", model_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Stream output
        for line in process.stdout:
            print(line, end='', flush=True)
        
        process.wait()
        
        if process.returncode == 0:
            print(f"\n✓ Successfully downloaded {model_name}")
            return True
        else:
            print(f"\n✗ Failed to download {model_name}")
            return False
            
    except Exception as e:
        print(f"\n✗ Error downloading {model_name}: {e}")
        return False


def main():
    """Main execution."""
    print("=" * 60)
    print("Ollama Model Setup for Research Paper")
    print("=" * 60)
    
    # Check Ollama installation
    if not check_ollama_installed():
        print("\n❌ Ollama is not installed!")
        print("\nInstall it with:")
        print("  winget install Ollama.Ollama")
        print("\nThen restart this script.")
        sys.exit(1)
    
    # Check Ollama service
    if not check_ollama_running():
        print("\n❌ Ollama service is not running!")
        print("\nOpen a new terminal and run:")
        print("  ollama serve")
        print("\nThen restart this script.")
        sys.exit(1)
    
    # List current models
    has_phi3, has_llama = list_downloaded_models()
    
    # Determine what to download
    download_queue = []
    
    if not has_phi3:
        download_queue.append(("phi3", "3.8GB"))
    else:
        print("\n✓ phi3 already downloaded")
    
    if not has_llama:
        download_queue.append(("llama3.2", "2.0GB"))
    else:
        print("\n✓ llama3.2 already downloaded")
    
    # All models ready?
    if not download_queue:
        print("\n" + "=" * 60)
        print("✅ All models are ready!")
        print("=" * 60)
        print("\nYou can now run:")
        print("  python experiments/multi_model_comparison.py")
        return
    
    # Ask user confirmation
    print("\n" + "=" * 60)
    print("Models to Download:")
    print("=" * 60)
    total_size = 0
    for name, size in download_queue:
        print(f"  • {name} ({size})")
        # Parse size to GB
        gb = float(size.replace("GB", ""))
        total_size += gb
    
    print(f"\nTotal download size: ~{total_size:.1f}GB")
    print(f"Estimated time: {int(total_size * 2)}-{int(total_size * 3)} minutes")
    
    response = input("\nProceed with download? (y/n): ")
    if response.lower() != 'y':
        print("\nDownload cancelled.")
        print("\nYou can download manually with:")
        for name, _ in download_queue:
            print(f"  ollama pull {name}")
        return
    
    # Download models
    success_count = 0
    for name, size in download_queue:
        if download_model(name, size):
            success_count += 1
        time.sleep(1)  # Brief pause between downloads
    
    # Final summary
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    print(f"Successfully downloaded: {success_count}/{len(download_queue)} models")
    
    if success_count == len(download_queue):
        print("\n✅ All models ready!")
        print("\nNext steps:")
        print("  1. Verify .env has GEMINI_API_KEY")
        print("  2. Run: python experiments/multi_model_comparison.py")
        print("  3. Check outputs in: data/multi_model_comparison/")
    else:
        print("\n⚠️  Some downloads failed. Try manually:")
        for name, _ in download_queue:
            print(f"  ollama pull {name}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user.")
        sys.exit(0)
