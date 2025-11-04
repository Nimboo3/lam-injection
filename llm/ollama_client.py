"""
Ollama Client - Local LLM integration for Ollama models.

Supports local models like Phi, Llama, Mistral, etc.
"""

import requests
import json
from typing import Optional, Dict, Any
from llm.wrapper import LLMClient


class OllamaClient(LLMClient):
    """
    Client for Ollama local LLM models.
    
    Supports models that can run on consumer hardware:
    - phi3 (3.8GB) - Microsoft's efficient model
    - llama3.2 (2GB) - Meta's latest small model
    - tinyllama (637MB) - Ultra-lightweight
    - gemma2:2b (1.6GB) - Google's small model
    """
    
    def __init__(
        self,
        model_name: str = "phi3",
        base_url: str = "http://localhost:11434",
        injection_threshold: float = 0.5,
        seed: int = 42
    ):
        """
        Initialize Ollama client.
        
        Args:
            model_name: Ollama model name (phi3, llama3.2, tinyllama, gemma2:2b)
            base_url: Ollama API endpoint
            injection_threshold: Threshold for mock fallback
            seed: Random seed
        """
        self.model_name = model_name
        self.base_url = base_url
        self.injection_threshold = injection_threshold
        self.seed = seed
        self.backend = "ollama"
        
        # Track usage
        self.total_calls = 0
        self.total_tokens = 0
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test if Ollama is running and model is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            
            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]
            
            if not any(self.model_name in name for name in model_names):
                print(f"⚠️  Model '{self.model_name}' not found in Ollama.")
                print(f"Available models: {model_names}")
                print(f"\nTo download: ollama pull {self.model_name}")
                
        except requests.exceptions.RequestException as e:
            print(f"⚠️  Cannot connect to Ollama at {self.base_url}")
            print(f"Error: {e}")
            print(f"\nMake sure Ollama is running:")
            print(f"  1. Open Ollama application")
            print(f"  2. Or run: ollama serve")
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.0,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate response from Ollama model.
        
        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = deterministic)
            system_prompt: Optional system prompt
        
        Returns:
            Generated text response
        """
        self.total_calls += 1
        
        # Combine prompts
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        try:
            # Call Ollama API
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                        "top_k": 1 if temperature == 0.0 else 40,
                        "top_p": 1.0,
                        "seed": self.seed
                    }
                },
                timeout=60  # Longer timeout for local inference
            )
            response.raise_for_status()
            
            result = response.json()
            text = result.get('response', '').strip()
            
            # Track tokens
            self.total_tokens += result.get('prompt_eval_count', 0)
            self.total_tokens += result.get('eval_count', 0)
            
            return text
            
        except requests.exceptions.RequestException as e:
            print(f"⚠️  Ollama API error: {e}")
            print(f"Falling back to simple response...")
            
            # Fallback: Extract action from prompt
            return self._fallback_response(full_prompt)
    
    def _fallback_response(self, prompt: str) -> str:
        """Simple fallback when Ollama is unavailable."""
        # Try to find action in prompt
        actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'PICK', 'DROP']
        prompt_upper = prompt.upper()
        
        # Look for goal direction
        if 'GOAL' in prompt_upper:
            # Try to parse positions
            if '(' in prompt and ')' in prompt:
                return "RIGHT"  # Default navigation
        
        return "RIGHT"  # Safe default
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "backend": self.backend,
            "model_name": self.model_name,
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "avg_tokens_per_call": (
                self.total_tokens / self.total_calls if self.total_calls > 0 else 0
            )
        }
    
    def reset_stats(self):
        """Reset usage statistics."""
        self.total_calls = 0
        self.total_tokens = 0


def list_ollama_models() -> list:
    """List all available Ollama models."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        response.raise_for_status()
        models = response.json().get('models', [])
        return [m['name'] for m in models]
    except Exception as e:
        print(f"Error listing Ollama models: {e}")
        return []


def download_ollama_model(model_name: str):
    """Download an Ollama model (shows progress)."""
    import subprocess
    print(f"Downloading {model_name}...")
    subprocess.run(['ollama', 'pull', model_name])


# Recommended models for 16GB RAM + Intel UHD
RECOMMENDED_MODELS = {
    'phi3': {
        'size': '3.8GB',
        'description': 'Microsoft Phi-3 - Excellent for reasoning tasks',
        'download': 'ollama pull phi3'
    },
    'llama3.2': {
        'size': '2GB', 
        'description': 'Meta Llama 3.2 - Latest efficient model',
        'download': 'ollama pull llama3.2'
    },
    'tinyllama': {
        'size': '637MB',
        'description': 'TinyLlama - Ultra lightweight, fast inference',
        'download': 'ollama pull tinyllama'
    },
    'gemma2:2b': {
        'size': '1.6GB',
        'description': 'Google Gemma 2B - Efficient and capable',
        'download': 'ollama pull gemma2:2b'
    }
}


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Ollama Client Demo")
    print("=" * 70)
    print()
    
    # List available models
    print("Available Ollama models:")
    models = list_ollama_models()
    if models:
        for model in models:
            print(f"  • {model}")
    else:
        print("  No models found or Ollama not running")
    
    print()
    print("Recommended models for your hardware (16GB RAM + Intel UHD):")
    for name, info in RECOMMENDED_MODELS.items():
        print(f"\n  {name} ({info['size']})")
        print(f"    {info['description']}")
        print(f"    Download: {info['download']}")
    
    print()
    print("=" * 70)
    
    # Test if Ollama is available
    if models:
        print("\nTesting Ollama client...")
        client = OllamaClient(model_name=models[0].split(':')[0])
        
        test_prompt = """
        Agent position: (2, 3)
        Goal position: (5, 3)
        
        What action should the agent take? (UP/DOWN/LEFT/RIGHT)
        """
        
        response = client.generate(test_prompt, max_tokens=50, temperature=0.0)
        print(f"\nPrompt: Navigate from (2,3) to (5,3)")
        print(f"Response: {response}")
        print(f"\nStats: {client.get_stats()}")
    
    print("\n" + "=" * 70)
