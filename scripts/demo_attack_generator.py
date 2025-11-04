"""
Demo script for attack generator.

Shows various attack generation capabilities.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from attacks.generator import (
    generate_document,
    generate_episode_attack_config,
    generate_attack_suite,
    save_seed_texts,
    load_seed_texts
)


def demo_single_documents():
    """Demonstrate single document generation."""
    print("=" * 80)
    print("DEMO 1: Single Document Generation")
    print("=" * 80)
    print()
    
    print("Generating documents with varying attack strengths:")
    print("-" * 80)
    
    strengths = [0.0, 0.2, 0.5, 0.8, 1.0]
    for strength in strengths:
        doc = generate_document(
            attack_strength=strength,
            attack_type="direct",
            target_action="RIGHT",
            seed=42
        )
        print(f"\n[Strength: {strength:.1f}]")
        print(f"Text: {doc}")
        print(f"Length: {len(doc)} chars")


def demo_attack_types():
    """Demonstrate different attack types."""
    print("\n" + "=" * 80)
    print("DEMO 2: Attack Types")
    print("=" * 80)
    print()
    
    attack_types = ["direct", "polite", "hidden", "encoding"]
    
    print("Same strength (0.8), different types:")
    print("-" * 80)
    
    for attack_type in attack_types:
        doc = generate_document(
            attack_strength=0.8,
            attack_type=attack_type,
            target_action="UP",
            seed=42
        )
        print(f"\n[{attack_type.upper()}]")
        print(f"Text: {doc}")


def demo_episode_configs():
    """Demonstrate episode configuration generation."""
    print("\n" + "=" * 80)
    print("DEMO 3: Episode Configurations")
    print("=" * 80)
    print()
    
    distributions = ["uniform", "mixed", "escalating"]
    
    for distribution in distributions:
        print(f"\n[Distribution: {distribution}]")
        print("-" * 80)
        
        docs = generate_episode_attack_config(
            num_docs=5,
            attack_strength=0.8,
            distribution=distribution,
            attack_type="direct",
            seed=42
        )
        
        print(f"Generated {len(docs)} documents:")
        for i, doc in enumerate(docs, 1):
            print(f"\n  Doc {i}:")
            print(f"    Position: {doc['pos']}")
            print(f"    Strength: {doc['attack_strength']:.2f}")
            print(f"    Text: {doc['text'][:60]}...")


def demo_attack_suite():
    """Demonstrate attack suite generation."""
    print("\n" + "=" * 80)
    print("DEMO 4: Attack Suite")
    print("=" * 80)
    print()
    
    suite = generate_attack_suite(seed=42)
    
    print("Generated attack suite with configurations:")
    print("-" * 80)
    
    for config_name, docs in suite.items():
        print(f"\n[{config_name}]")
        print(f"  Number of documents: {len(docs)}")
        
        if docs:
            strengths = [d['attack_strength'] for d in docs]
            print(f"  Strength range: {min(strengths):.2f} - {max(strengths):.2f}")
            
            # Show first document
            print(f"  Example: {docs[0]['text'][:70]}...")


def demo_seed_texts():
    """Demonstrate seed text utilities."""
    print("\n" + "=" * 80)
    print("DEMO 5: Seed Texts")
    print("=" * 80)
    print()
    
    # Save seed texts
    output_dir = "data/seed_texts"
    print(f"Saving seed texts to: {output_dir}")
    print("-" * 80)
    save_seed_texts(output_dir)
    
    # Load them back
    print(f"\nLoading seed texts from: {output_dir}")
    print("-" * 80)
    texts = load_seed_texts(output_dir)
    
    print(f"\nLoaded {len(texts)} seed texts:")
    for i, text in enumerate(texts, 1):
        print(f"\n  Text {i}:")
        lines = text.split('\n')
        for line in lines[:2]:  # Show first 2 lines
            print(f"    {line}")
        if len(lines) > 2:
            print(f"    ... ({len(lines)} lines total)")


def demo_strength_correlation():
    """Demonstrate attack strength correlation."""
    print("\n" + "=" * 80)
    print("DEMO 6: Attack Strength Correlation")
    print("=" * 80)
    print()
    
    print("Analyzing injection intensity vs. attack strength:")
    print("-" * 80)
    
    strengths = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    for strength in strengths:
        doc = generate_document(
            attack_strength=strength,
            attack_type="direct",
            target_action="DOWN",
            seed=123
        )
        
        # Count injection keywords
        keywords = ["IGNORE", "DISREGARD", "STOP", "OVERRIDE", "URGENT"]
        keyword_count = sum(1 for kw in keywords if kw in doc.upper())
        
        print(f"\nStrength {strength:.2f}:")
        print(f"  Injection keywords found: {keyword_count}")
        print(f"  Document length: {len(doc)} chars")
        print(f"  Preview: {doc[:80]}...")


def main():
    """Run all demos."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "ATTACK GENERATOR DEMO" + " " * 37 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    try:
        demo_single_documents()
        demo_attack_types()
        demo_episode_configs()
        demo_attack_suite()
        demo_seed_texts()
        demo_strength_correlation()
        
        print("\n" + "=" * 80)
        print("All demos completed successfully!")
        print("=" * 80)
        print()
        
    except Exception as e:
        print(f"\n[ERROR] Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
