"""attacks package - Prompt injection attack generation."""

from .generator import (
    generate_document,
    generate_episode_attack_config,
    generate_attack_suite,
    save_seed_texts,
    load_seed_texts,
    ATTACK_TEMPLATES,
    INSTRUCTIONS,
    BENIGN_TEMPLATES
)

__all__ = [
    "generate_document",
    "generate_episode_attack_config", 
    "generate_attack_suite",
    "save_seed_texts",
    "load_seed_texts",
    "ATTACK_TEMPLATES",
    "INSTRUCTIONS",
    "BENIGN_TEMPLATES"
]
