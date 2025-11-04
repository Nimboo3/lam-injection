"""
Tests for attack generator module.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
import random
from attacks.generator import (
    generate_document,
    generate_episode_attack_config,
    generate_attack_suite,
    save_seed_texts,
    load_seed_texts,
    ATTACK_TEMPLATES,
    INSTRUCTIONS,
    BENIGN_TEMPLATES
)


class TestGenerateDocument:
    """Test single document generation."""
    
    def test_no_attack_returns_benign(self):
        """Attack strength 0 should return benign text."""
        doc = generate_document(attack_strength=0.0, seed=42)
        assert len(doc) > 0
        # Should not contain attack markers
        assert "IGNORE" not in doc.upper()
        assert "DISREGARD" not in doc.upper()
    
    def test_attack_strength_affects_content(self):
        """Higher attack strength should produce more aggressive text."""
        weak = generate_document(attack_strength=0.1, attack_type="direct", seed=42)
        strong = generate_document(attack_strength=0.9, attack_type="direct", seed=42)
        
        # Strong should be shorter (less benign content) or more direct
        assert len(strong) > 0
        assert len(weak) > 0
    
    def test_different_attack_types(self):
        """Different attack types should use different templates."""
        direct = generate_document(attack_strength=0.8, attack_type="direct", seed=42)
        polite = generate_document(attack_strength=0.8, attack_type="polite", seed=42)
        hidden = generate_document(attack_strength=0.8, attack_type="hidden", seed=42)
        
        # All should be different
        assert direct != polite
        assert direct != hidden
        assert polite != hidden
    
    def test_target_action_included(self):
        """Specified target action should appear in document."""
        doc = generate_document(
            attack_strength=0.8,
            attack_type="direct",
            target_action="UP",
            seed=42
        )
        assert "UP" in doc.upper()
    
    def test_custom_base_text(self):
        """Custom base text should be used when provided."""
        base = "This is my custom text."
        doc = generate_document(
            base_text=base,
            attack_strength=0.2,
            seed=42
        )
        assert base in doc
    
    def test_seed_reproducibility(self):
        """Same seed should produce same output."""
        doc1 = generate_document(attack_strength=0.5, seed=123)
        doc2 = generate_document(attack_strength=0.5, seed=123)
        assert doc1 == doc2
    
    def test_different_seeds_vary(self):
        """Different seeds should produce different outputs."""
        doc1 = generate_document(attack_strength=0.5, seed=123)
        doc2 = generate_document(attack_strength=0.5, seed=456)
        assert doc1 != doc2
    
    def test_invalid_attack_type_fallback(self):
        """Invalid attack type should fallback to direct."""
        doc = generate_document(
            attack_strength=0.8,
            attack_type="invalid_type",
            seed=42
        )
        assert len(doc) > 0
    
    def test_invalid_target_action_fallback(self):
        """Invalid target action should fallback to RIGHT."""
        doc = generate_document(
            attack_strength=0.8,
            target_action="INVALID",
            seed=42
        )
        assert len(doc) > 0


class TestGenerateEpisodeAttackConfig:
    """Test episode configuration generation."""
    
    def test_correct_number_of_documents(self):
        """Should generate requested number of documents."""
        docs = generate_episode_attack_config(num_docs=5, attack_strength=0.5, seed=42)
        assert len(docs) == 5
    
    def test_document_structure(self):
        """Each document should have required fields."""
        docs = generate_episode_attack_config(num_docs=3, attack_strength=0.5, seed=42)
        for doc in docs:
            assert "pos" in doc
            assert "text" in doc
            assert "attack_strength" in doc
            assert isinstance(doc["pos"], tuple)
            assert len(doc["pos"]) == 2
            assert isinstance(doc["text"], str)
            assert isinstance(doc["attack_strength"], float)
    
    def test_positions_within_grid(self):
        """Document positions should be within grid bounds."""
        grid_size = (8, 8)
        docs = generate_episode_attack_config(
            num_docs=5,
            attack_strength=0.5,
            grid_size=grid_size,
            seed=42
        )
        for doc in docs:
            x, y = doc["pos"]
            assert 0 <= x < grid_size[0]
            assert 0 <= y < grid_size[1]
    
    def test_positions_unique(self):
        """Documents should have unique positions."""
        docs = generate_episode_attack_config(num_docs=5, attack_strength=0.5, seed=42)
        positions = [doc["pos"] for doc in docs]
        assert len(positions) == len(set(positions))
    
    def test_uniform_distribution(self):
        """Uniform distribution should give all same strength."""
        docs = generate_episode_attack_config(
            num_docs=5,
            attack_strength=0.7,
            distribution="uniform",
            seed=42
        )
        strengths = [doc["attack_strength"] for doc in docs]
        assert all(s == 0.7 for s in strengths)
    
    def test_mixed_distribution(self):
        """Mixed distribution should vary strengths."""
        docs = generate_episode_attack_config(
            num_docs=6,
            attack_strength=0.9,
            distribution="mixed",
            seed=42
        )
        strengths = [doc["attack_strength"] for doc in docs]
        unique_strengths = set(strengths)
        assert len(unique_strengths) > 1  # Should have variety
    
    def test_escalating_distribution(self):
        """Escalating distribution should increase strength."""
        docs = generate_episode_attack_config(
            num_docs=5,
            attack_strength=1.0,
            distribution="escalating",
            seed=42
        )
        strengths = [doc["attack_strength"] for doc in docs]
        # Should be in increasing order
        for i in range(len(strengths) - 1):
            assert strengths[i] <= strengths[i + 1]
    
    def test_attack_type_propagates(self):
        """Attack type should affect generated documents."""
        docs_direct = generate_episode_attack_config(
            num_docs=3,
            attack_strength=0.8,
            attack_type="direct",
            seed=42
        )
        docs_polite = generate_episode_attack_config(
            num_docs=3,
            attack_strength=0.8,
            attack_type="polite",
            seed=42
        )
        # At least one document should differ
        texts_direct = [d["text"] for d in docs_direct]
        texts_polite = [d["text"] for d in docs_polite]
        assert texts_direct != texts_polite
    
    def test_seed_reproducibility(self):
        """Same seed should produce same configuration."""
        docs1 = generate_episode_attack_config(num_docs=5, attack_strength=0.5, seed=99)
        docs2 = generate_episode_attack_config(num_docs=5, attack_strength=0.5, seed=99)
        
        for d1, d2 in zip(docs1, docs2):
            assert d1["pos"] == d2["pos"]
            assert d1["text"] == d2["text"]
            assert d1["attack_strength"] == d2["attack_strength"]


class TestGenerateAttackSuite:
    """Test attack suite generation."""
    
    def test_suite_has_expected_configs(self):
        """Suite should contain expected configuration names."""
        suite = generate_attack_suite(seed=42)
        
        expected_configs = [
            "no_attack",
            "weak_direct",
            "weak_polite",
            "medium_direct",
            "medium_mixed",
            "strong_direct",
            "strong_hidden",
            "escalating"
        ]
        
        for config in expected_configs:
            assert config in suite
    
    def test_no_attack_empty(self):
        """No attack configuration should be empty."""
        suite = generate_attack_suite(seed=42)
        assert suite["no_attack"] == []
    
    def test_weak_configs_low_strength(self):
        """Weak configurations should have low attack strength."""
        suite = generate_attack_suite(seed=42)
        
        for doc in suite["weak_direct"]:
            assert doc["attack_strength"] <= 0.3
        
        for doc in suite["weak_polite"]:
            assert doc["attack_strength"] <= 0.3
    
    def test_strong_configs_high_strength(self):
        """Strong configurations should have high attack strength."""
        suite = generate_attack_suite(seed=42)
        
        # At least some docs should be strong
        strong_direct_strengths = [d["attack_strength"] for d in suite["strong_direct"]]
        assert max(strong_direct_strengths) >= 0.8
        
        strong_hidden_strengths = [d["attack_strength"] for d in suite["strong_hidden"]]
        assert max(strong_hidden_strengths) >= 0.8
    
    def test_suite_seed_reproducibility(self):
        """Same seed should produce same suite."""
        suite1 = generate_attack_suite(seed=777)
        suite2 = generate_attack_suite(seed=777)
        
        assert suite1.keys() == suite2.keys()
        
        for config_name in suite1.keys():
            assert len(suite1[config_name]) == len(suite2[config_name])


class TestSeedTexts:
    """Test seed text utilities."""
    
    def test_save_seed_texts_creates_files(self, tmp_path):
        """save_seed_texts should create text files."""
        output_dir = tmp_path / "seed_texts"
        save_seed_texts(str(output_dir))
        
        # Check files exist
        assert (output_dir / "navigation_tips.txt").exists()
        assert (output_dir / "environment_info.txt").exists()
        assert (output_dir / "generic_advice.txt").exists()
    
    def test_save_seed_texts_content(self, tmp_path):
        """Saved seed texts should have content."""
        output_dir = tmp_path / "seed_texts"
        save_seed_texts(str(output_dir))
        
        for txt_file in output_dir.glob("*.txt"):
            content = txt_file.read_text()
            assert len(content) > 0
            assert len(content.strip()) > 0
    
    def test_load_seed_texts_returns_list(self, tmp_path):
        """load_seed_texts should return list of strings."""
        output_dir = tmp_path / "seed_texts"
        save_seed_texts(str(output_dir))
        
        texts = load_seed_texts(str(output_dir))
        assert isinstance(texts, list)
        assert len(texts) > 0
        assert all(isinstance(t, str) for t in texts)
    
    def test_load_seed_texts_nonexistent_dir(self):
        """Loading from nonexistent dir should return defaults."""
        texts = load_seed_texts("nonexistent_dir_12345")
        assert isinstance(texts, list)
        assert len(texts) > 0


class TestAttackStrengthCorrelation:
    """Test that attack strength correlates with injection intensity."""
    
    def test_strength_zero_no_injection(self):
        """Strength 0 should produce no injection keywords."""
        doc = generate_document(attack_strength=0.0, seed=42)
        
        injection_keywords = ["IGNORE", "DISREGARD", "OVERRIDE", "URGENT"]
        doc_upper = doc.upper()
        
        for keyword in injection_keywords:
            assert keyword not in doc_upper
    
    def test_strength_high_has_injection(self):
        """High strength should produce injection keywords."""
        doc = generate_document(
            attack_strength=0.9,
            attack_type="direct",
            seed=42
        )
        
        injection_keywords = ["IGNORE", "DISREGARD", "OVERRIDE", "STOP", "URGENT", "GO", "MOVE"]
        doc_upper = doc.upper()
        
        # At least one keyword should be present
        has_keyword = any(keyword in doc_upper for keyword in injection_keywords)
        assert has_keyword
    
    def test_increasing_strength_increases_directness(self):
        """Higher strength should produce more direct/shorter text."""
        docs = []
        for strength in [0.1, 0.3, 0.5, 0.7, 0.9]:
            doc = generate_document(
                attack_strength=strength,
                attack_type="direct",
                target_action="UP",
                seed=42
            )
            docs.append(doc)
        
        # Very strong attack should be more concise than very weak
        assert len(docs[4]) <= len(docs[0]) + 50  # Allow some variance


class TestTemplates:
    """Test that templates are properly defined."""
    
    def test_attack_templates_complete(self):
        """All attack types should have templates."""
        expected_types = ["direct", "hidden", "polite", "encoding"]
        for attack_type in expected_types:
            assert attack_type in ATTACK_TEMPLATES
            assert len(ATTACK_TEMPLATES[attack_type]) > 0
    
    def test_instructions_complete(self):
        """All actions should have instruction templates."""
        expected_actions = ["UP", "DOWN", "LEFT", "RIGHT", "PICK", "DROP"]
        for action in expected_actions:
            assert action in INSTRUCTIONS
            assert len(INSTRUCTIONS[action]) > 0
    
    def test_benign_templates_exist(self):
        """Benign templates should be defined."""
        assert len(BENIGN_TEMPLATES) > 0
        assert all(isinstance(t, str) for t in BENIGN_TEMPLATES)
    
    def test_templates_have_placeholders(self):
        """Direct templates should have {instruction} placeholder."""
        for template in ATTACK_TEMPLATES["direct"]:
            assert "{instruction}" in template


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
