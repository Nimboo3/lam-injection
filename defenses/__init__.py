"""
Defense mechanisms package for prompt injection protection.

Provides three defense layers:
1. Sanitizer: Remove/neutralize attack patterns
2. Detector: Flag suspicious documents
3. Verifier: Check LLM reasoning
"""

from .sanitizer import sanitize_document, sanitize_documents
from .detector import detect_malicious, is_malicious
from .verifier import verify_with_llm, should_reject_response

__all__ = [
    "sanitize_document",
    "sanitize_documents",
    "detect_malicious",
    "is_malicious",
    "verify_with_llm",
    "should_reject_response"
]
