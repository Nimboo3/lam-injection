"""
Enhanced Gemini Client - Support for multiple Gemini models.

Optimized for FREE TIER with aggressive rate limiting.
Free tier limits: 15 RPM, 1,500 RPD, 1M TPM
"""

import os
import requests
import json
import time
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()


class GeminiClient:
    """
    Client for Google Gemini API with support for multiple models.
    
    Optimized for free tier: 15 RPM, 1,500 RPD limits.
    Rate limited to ~13 RPM (4.5s interval) for safety margin.
    """
    
    # Class variable to track last API call time (shared across instances)
    _last_call_time = 0
    _min_call_interval = 4.5  # 4.5 seconds = ~13 RPM (safe margin for 15 RPM limit)
    _daily_call_count = 0
    _daily_limit = 1400  # Leave 100 calls margin from 1,500 RPD
    
    def __init__(
        self,
        model_name: str = "gemini-2.5-flash-lite",
        api_key: Optional[str] = None,
        api_key_env: str = "GEMINI_API_KEY"
    ):
        """
        Initialize Gemini client.
        
        Args:
            model_name: Model identifier (e.g., gemini-2.5-flash-lite)
            api_key: API key (or None to read from environment)
            api_key_env: Environment variable name for API key
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv(api_key_env)
        self.backend = "gemini"
        
        # Track usage
        self.total_calls = 0
        self.total_tokens = 0
        
        # Validate setup
        if not self.api_key:
            raise ValueError(
                f"Gemini API key not found. Set {api_key_env} environment variable "
                "or pass api_key parameter."
            )
        
        # API endpoint
        self.api_endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/"
            f"models/{self.model_name}:generateContent"
        )
        
        print(f"  📊 Free tier limits: 15 RPM, 1,500 RPD")
        print(f"  ⏱️  Rate limit: {self._min_call_interval}s between calls")
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
        max_retries: int = 3
    ) -> str:
        """
        Generate response from Gemini with aggressive rate limiting for free tier.
        
        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = deterministic)
            system_prompt: Optional system prompt (prepended to prompt)
            max_retries: Number of retries on quota errors
        
        Returns:
            Generated text response
        """
        # Check daily limit
        if GeminiClient._daily_call_count >= GeminiClient._daily_limit:
            print(f"⚠️  Daily limit reached ({GeminiClient._daily_limit} calls)")
            print(f"   Returning fallback response")
            return "RIGHT"  # Safe fallback
        
        # Retry loop for quota errors
        for attempt in range(max_retries):
            try:
                # Rate limiting: Wait if needed
                current_time = time.time()
                time_since_last_call = current_time - GeminiClient._last_call_time
                
                if time_since_last_call < GeminiClient._min_call_interval:
                    wait_time = GeminiClient._min_call_interval - time_since_last_call
                    time.sleep(wait_time)
                
                GeminiClient._last_call_time = time.time()
                self.total_calls += 1
                GeminiClient._daily_call_count += 1
                
                # Combine prompts
                full_prompt = prompt
                if system_prompt:
                    full_prompt = f"{system_prompt}\n\n{prompt}"
                
                # Build request
                headers = {
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "contents": [{
                        "parts": [{"text": full_prompt}]
                    }],
                    "generationConfig": {
                        "temperature": temperature,
                        "maxOutputTokens": max_tokens,
                        "topP": 1.0,
                        "topK": 1 if temperature == 0.0 else 40
                    }
                }
                
                url = f"{self.api_endpoint}?key={self.api_key}"
                
                # Make request
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                
                # Handle response
                if response.status_code == 200:
                    result = response.json()
                    
                    # Parse response
                    if "candidates" in result and len(result["candidates"]) > 0:
                        candidate = result["candidates"][0]
                        if "content" in candidate and "parts" in candidate["content"]:
                            text = candidate["content"]["parts"][0].get("text", "")
                            
                            # Track token usage
                            if "usageMetadata" in result:
                                self.total_tokens += result["usageMetadata"].get("totalTokenCount", 0)
                            
                            return text.strip() if text else "RIGHT"
                    
                    # Fallback if unexpected structure
                    print(f"⚠️  Unexpected response structure from {self.model_name}")
                    return "RIGHT"
                
                elif response.status_code == 429:
                    # Quota exceeded - exponential backoff
                    wait_time = (2 ** attempt) * 30  # 30s, 60s, 120s
                    print(f"⚠️  Quota exceeded (429). Waiting {wait_time}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                
                else:
                    # Other HTTP error
                    print(f"⚠️  HTTP {response.status_code}: {response.text[:200]}")
                    if attempt < max_retries - 1:
                        time.sleep(5)
                        continue
                    else:
                        return "RIGHT"
                        
            except requests.exceptions.Timeout:
                print(f"⚠️  Timeout (attempt {attempt+1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(10)
                    continue
                else:
                    return "RIGHT"
                    
            except requests.exceptions.RequestException as e:
                print(f"⚠️  Network error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                else:
                    return "RIGHT"
                    
            except Exception as e:
                print(f"⚠️  Unexpected error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                else:
                    return "RIGHT"
        
        # Max retries exceeded
        print(f"⚠️  Max retries exceeded")
        return "RIGHT"
    
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


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Gemini Client Demo")
    print("=" * 70)
    print()
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("⚠️  GEMINI_API_KEY not found in environment")
        print("Set it in .env file or environment variables")
    else:
        print("✓ API key found")
        print()
        
        # Test both models
        models = ["gemini-1.5-flash", "gemini-1.5-flash-8b"]
        
        for model_name in models:
            print(f"Testing {model_name}...")
            client = GeminiClient(model_name=model_name)
            
            test_prompt = """
            Agent position: (2, 3)
            Goal position: (5, 3)
            
            What action should the agent take? (UP/DOWN/LEFT/RIGHT)
            """
            
            response = client.generate(test_prompt, max_tokens=50, temperature=0.0)
            print(f"  Response: {response}")
            print(f"  Stats: {client.get_stats()}")
            print()
    
    print("=" * 70)
