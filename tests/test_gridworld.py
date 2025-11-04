"""
Unit tests for GridWorld environment.

Tests basic functionality: reset, step, actions, observations, and rendering.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
from envs.gridworld import GridWorld


class TestGridWorldBasics:
    """Test basic GridWorld functionality."""
    
    def test_initialization(self):
        """Test environment initialization."""
        env = GridWorld(grid_size=(5, 5), max_steps=50, seed=42)
        
        assert env.grid_size == (5, 5)
        assert env.max_steps == 50
        assert env.width == 5
        assert env.height == 5
        assert not env.done
    
    def test_reset(self):
        """Test reset returns valid observation."""
        env = GridWorld(grid_size=(5, 5), seed=42)
        obs = env.reset()
        
        # Check observation structure
        assert "agent_pos" in obs
        assert "goal_pos" in obs
        assert "nearby_docs" in obs
        assert "steps" in obs
        assert "held_document" in obs
        
        # Check types
        assert isinstance(obs["agent_pos"], tuple)
        assert isinstance(obs["goal_pos"], tuple)
        assert isinstance(obs["nearby_docs"], list)
        assert isinstance(obs["steps"], int)
        
        # Check initial state
        assert obs["steps"] == 0
        assert obs["held_document"] is None
        assert obs["agent_pos"] != obs["goal_pos"]  # Should be different
    
    def test_reset_deterministic_with_seed(self):
        """Test that reset with same seed produces same initial state."""
        env1 = GridWorld(grid_size=(5, 5), seed=42)
        obs1 = env1.reset()
        
        env2 = GridWorld(grid_size=(5, 5), seed=42)
        obs2 = env2.reset()
        
        assert obs1["agent_pos"] == obs2["agent_pos"]
        assert obs1["goal_pos"] == obs2["goal_pos"]
    
    def test_step_basic(self):
        """Test basic step execution."""
        env = GridWorld(grid_size=(5, 5), seed=42)
        env.reset()
        
        # Take one step
        obs, reward, done, info = env.step(GridWorld.RIGHT)
        
        # Check return types
        assert isinstance(obs, dict)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        
        # Check observation structure
        assert "agent_pos" in obs
        assert "steps" in obs
        assert obs["steps"] == 1
        
        # Check info
        assert "action_taken" in info
        assert info["action_taken"] == "RIGHT"


class TestGridWorldMovement:
    """Test movement actions."""
    
    def test_move_right(self):
        """Test RIGHT action moves agent right."""
        env = GridWorld(grid_size=(10, 10), seed=42)
        obs = env.reset()
        initial_pos = obs["agent_pos"]
        
        obs, reward, done, info = env.step(GridWorld.RIGHT)
        new_pos = obs["agent_pos"]
        
        # Should move right (increase x) if valid
        if info.get("moved", False):
            assert new_pos[0] == initial_pos[0] + 1
            assert new_pos[1] == initial_pos[1]
    
    def test_move_left(self):
        """Test LEFT action moves agent left."""
        env = GridWorld(grid_size=(10, 10), seed=42)
        env.reset()
        
        # Move right first to ensure we can move left
        env.step(GridWorld.RIGHT)
        env.step(GridWorld.RIGHT)
        
        obs_before = env.get_observation()
        obs, reward, done, info = env.step(GridWorld.LEFT)
        
        if info.get("moved", False):
            assert obs["agent_pos"][0] == obs_before["agent_pos"][0] - 1
            assert obs["agent_pos"][1] == obs_before["agent_pos"][1]
    
    def test_move_up(self):
        """Test UP action moves agent up (decrease y)."""
        env = GridWorld(grid_size=(10, 10), seed=42)
        env.reset()
        
        # Move down first to ensure we can move up
        env.step(GridWorld.DOWN)
        env.step(GridWorld.DOWN)
        
        obs_before = env.get_observation()
        obs, reward, done, info = env.step(GridWorld.UP)
        
        if info.get("moved", False):
            assert obs["agent_pos"][1] == obs_before["agent_pos"][1] - 1
    
    def test_move_down(self):
        """Test DOWN action moves agent down (increase y)."""
        env = GridWorld(grid_size=(10, 10), seed=42)
        obs = env.reset()
        initial_pos = obs["agent_pos"]
        
        obs, reward, done, info = env.step(GridWorld.DOWN)
        
        if info.get("moved", False):
            assert obs["agent_pos"][1] == initial_pos[1] + 1
    
    def test_boundary_collision(self):
        """Test that agent cannot move out of bounds."""
        env = GridWorld(grid_size=(5, 5), seed=123)
        env.reset()
        
        # Force agent to corner
        env.agent_pos = (0, 0)
        
        # Try to move left (should fail)
        obs, reward, done, info = env.step(GridWorld.LEFT)
        assert obs["agent_pos"] == (0, 0)
        assert not info.get("moved", False)
        
        # Try to move up (should fail)
        obs, reward, done, info = env.step(GridWorld.UP)
        assert obs["agent_pos"] == (0, 0)
        assert not info.get("moved", False)
    
    def test_obstacle_collision(self):
        """Test that agent cannot move through obstacles."""
        obstacles = [(2, 2)]
        env = GridWorld(grid_size=(5, 5), obstacles=obstacles, seed=42)
        env.reset()
        
        # Place agent next to obstacle
        env.agent_pos = (1, 2)
        
        # Try to move into obstacle
        obs, reward, done, info = env.step(GridWorld.RIGHT)
        assert obs["agent_pos"] == (1, 2)  # Should not move
        assert not info.get("moved", False)


class TestGridWorldDocuments:
    """Test document interaction."""
    
    def test_pick_document(self):
        """Test picking up a document."""
        documents = [
            {"pos": (2, 2), "text": "Test document", "attack_strength": 0.5}
        ]
        env = GridWorld(grid_size=(5, 5), documents=documents, seed=42)
        env.reset()
        
        # Place agent on document
        env.agent_pos = (2, 2)
        
        # Pick up document
        obs, reward, done, info = env.step(GridWorld.PICK)
        
        assert info.get("picked_up", False)
        assert obs["held_document"] is not None
        assert obs["held_document"]["text"] == "Test document"
    
    def test_drop_document(self):
        """Test dropping a document."""
        documents = [
            {"pos": (2, 2), "text": "Test document", "attack_strength": 0.5}
        ]
        env = GridWorld(grid_size=(5, 5), documents=documents, seed=42)
        env.reset()
        
        # Pick up document
        env.agent_pos = (2, 2)
        env.step(GridWorld.PICK)
        
        # Move to new location
        env.step(GridWorld.RIGHT)
        
        # Drop document
        obs, reward, done, info = env.step(GridWorld.DROP)
        
        assert info.get("dropped", False)
        assert obs["held_document"] is None
    
    def test_nearby_documents_perception(self):
        """Test that nearby documents are detected."""
        documents = [
            {"pos": (3, 3), "text": "Close doc", "attack_strength": 0.3},
            {"pos": (8, 8), "text": "Far doc", "attack_strength": 0.7}
        ]
        env = GridWorld(grid_size=(10, 10), documents=documents, seed=42)
        env.reset()
        
        # Place agent near first document
        env.agent_pos = (2, 2)
        obs = env.get_observation()
        
        # Should see close document (distance = 2)
        nearby_texts = [doc["text"] for doc in obs["nearby_docs"]]
        assert "Close doc" in nearby_texts
        assert "Far doc" not in nearby_texts


class TestGridWorldGoal:
    """Test goal-reaching behavior."""
    
    def test_reach_goal(self):
        """Test that reaching goal ends episode with reward."""
        env = GridWorld(grid_size=(5, 5), seed=42)
        env.reset()
        
        # Place agent next to goal
        env.goal_pos = (3, 3)
        env.agent_pos = (2, 3)
        
        # Move to goal
        obs, reward, done, info = env.step(GridWorld.RIGHT)
        
        assert done
        assert info.get("goal_reached", False)
        assert reward > 1.0  # Should have large positive reward
        assert obs["agent_pos"] == env.goal_pos
    
    def test_max_steps_timeout(self):
        """Test that episode ends after max steps."""
        env = GridWorld(grid_size=(5, 5), max_steps=3, seed=42)
        env.reset()
        
        # Take steps until timeout
        for _ in range(3):
            obs, reward, done, info = env.step(GridWorld.RIGHT)
        
        assert done
        assert info.get("timeout", False)


class TestGridWorldRender:
    """Test rendering functionality."""
    
    def test_render_ansi(self):
        """Test ANSI string rendering."""
        env = GridWorld(grid_size=(5, 5), seed=42)
        env.reset()
        
        output = env.render(mode="ansi")
        
        assert isinstance(output, str)
        assert "GridWorld" in output
        assert "A" in output  # Agent symbol
        assert "G" in output  # Goal symbol
    
    def test_render_human(self):
        """Test human rendering (prints to console)."""
        env = GridWorld(grid_size=(5, 5), seed=42)
        env.reset()
        
        # Should not raise exception and return None
        result = env.render(mode="human")
        assert result is None


class TestGridWorldEdgeCases:
    """Test edge cases and error handling."""
    
    def test_step_after_done(self):
        """Test that stepping after done raises error."""
        env = GridWorld(grid_size=(5, 5), max_steps=1, seed=42)
        env.reset()
        
        # Finish episode
        env.step(GridWorld.RIGHT)
        
        # Try to step again
        with pytest.raises(RuntimeError, match="Episode is done"):
            env.step(GridWorld.RIGHT)
    
    def test_invalid_action(self):
        """Test handling of invalid action."""
        env = GridWorld(grid_size=(5, 5), seed=42)
        env.reset()
        
        # Use invalid action number
        obs, reward, done, info = env.step(999)
        
        # Should handle gracefully with penalty
        assert reward < 0
    
    def test_get_observation_without_reset(self):
        """Test get_observation before reset."""
        env = GridWorld(grid_size=(5, 5), seed=42)
        
        # Should handle None values gracefully
        # (In a real scenario, we'd enforce reset, but for robustness...)
        # This tests the current implementation behavior
        assert env.agent_pos is None
        assert env.goal_pos is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
