"""
LLM wrapper module with mock, Gemini API, and Ollama support.

Available backends:
- Mock: Deterministic testing without API keys
- Gemini: Google's cloud models (gemini-1.5-flash, gemini-1.5-flash-8b)
- Ollama: Local models (phi3, llama3.2, tinyllama, gemma2:2b)
"""

from llm.wrapper import LLMClient, create_llm_client
from llm.gemini_client import GeminiClient
from llm.ollama_client import OllamaClient, list_ollama_models, download_ollama_model

__all__ = [
    'LLMClient',
    'create_llm_client',
    'GeminiClient',
    'OllamaClient',
    'list_ollama_models',
    'download_ollama_model',
]
