"""
GridWorld Environment - A simple Gym-like navigation environment.

The agent must navigate a grid to reach a goal while potentially encountering
documents that may contain prompt injection attacks.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np


class GridWorld:
    """
    A simple grid-based navigation environment with Gym-like API.
    
    The agent can move in four directions (UP, DOWN, LEFT, RIGHT) and interact
    with documents placed in the grid (PICK, DROP). The goal is to reach a
    target position.
    
    Actions:
        0: UP (decrease y)
        1: DOWN (increase y)
        2: LEFT (decrease x)
        3: RIGHT (increase x)
        4: PICK (pick up document at current position)
        5: DROP (drop held document)
    """
    
    # Action constants
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    PICK = 4
    DROP = 5
    
    ACTION_NAMES = {
        0: "UP",
        1: "DOWN",
        2: "LEFT",
        3: "RIGHT",
        4: "PICK",
        5: "DROP"
    }
    
    def __init__(
        self,
        grid_size: Tuple[int, int] = (10, 10),
        max_steps: int = 100,
        documents: Optional[List[Dict[str, Any]]] = None,
        obstacles: Optional[List[Tuple[int, int]]] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize the GridWorld environment.
        
        Args:
            grid_size: Tuple of (width, height) for the grid
            max_steps: Maximum number of steps per episode
            documents: List of document dicts with keys:
                       'pos': (x, y), 'text': str, 'attack_strength': float
            obstacles: List of (x, y) positions that block movement
            seed: Random seed for reproducibility
        """
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.width, self.height = grid_size
        
        # Set random seed
        self.rng = np.random.RandomState(seed)
        
        # Initialize state variables
        self.agent_pos: Optional[Tuple[int, int]] = None
        self.goal_pos: Optional[Tuple[int, int]] = None
        self.steps: int = 0
        self.done: bool = False
        self.held_document: Optional[Dict[str, Any]] = None
        
        # Documents in the environment
        self.documents = documents if documents is not None else []
        self.documents_remaining = []
        
        # Obstacles (walls)
        self.obstacles = obstacles if obstacles is not None else []
        
        # Episode tracking
        self.episode_reward = 0.0
        
    def reset(self) -> Dict[str, Any]:
        """
        Reset the environment to initial state.
        
        Returns:
            Initial observation dictionary
        """
        # Random agent position
        self.agent_pos = self._random_free_position()
        
        # Random goal position (not same as agent)
        self.goal_pos = self._random_free_position(exclude=[self.agent_pos])
        
        # Reset documents (deep copy to avoid mutation)
        self.documents_remaining = [doc.copy() for doc in self.documents]
        
        # Reset state
        self.steps = 0
        self.done = False
        self.held_document = None
        self.episode_reward = 0.0
        
        return self.get_observation()
    
    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Integer action (0-5)
        
        Returns:
            Tuple of (observation, reward, done, info)
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
        
        self.steps += 1
        reward = -0.01  # Small negative reward per step (time penalty)
        info = {"action_taken": self.ACTION_NAMES.get(action, "UNKNOWN")}
        
        # Execute action
        if action == self.UP:
            new_pos = (self.agent_pos[0], self.agent_pos[1] - 1)
            if self._is_valid_position(new_pos):
                self.agent_pos = new_pos
                info["moved"] = True
            else:
                info["moved"] = False
                reward -= 0.05  # Penalty for invalid move
                
        elif action == self.DOWN:
            new_pos = (self.agent_pos[0], self.agent_pos[1] + 1)
            if self._is_valid_position(new_pos):
                self.agent_pos = new_pos
                info["moved"] = True
            else:
                info["moved"] = False
                reward -= 0.05
                
        elif action == self.LEFT:
            new_pos = (self.agent_pos[0] - 1, self.agent_pos[1])
            if self._is_valid_position(new_pos):
                self.agent_pos = new_pos
                info["moved"] = True
            else:
                info["moved"] = False
                reward -= 0.05
                
        elif action == self.RIGHT:
            new_pos = (self.agent_pos[0] + 1, self.agent_pos[1])
            if self._is_valid_position(new_pos):
                self.agent_pos = new_pos
                info["moved"] = True
            else:
                info["moved"] = False
                reward -= 0.05
                
        elif action == self.PICK:
            # Try to pick up document at current position
            doc = self._get_document_at_position(self.agent_pos)
            if doc is not None and self.held_document is None:
                self.held_document = doc
                self.documents_remaining.remove(doc)
                info["picked_up"] = True
                info["document"] = doc
                reward += 0.1  # Small reward for picking up
            else:
                info["picked_up"] = False
                reward -= 0.05
                
        elif action == self.DROP:
            # Drop held document at current position
            if self.held_document is not None:
                dropped_doc = self.held_document.copy()
                dropped_doc["pos"] = self.agent_pos
                self.documents_remaining.append(dropped_doc)
                self.held_document = None
                info["dropped"] = True
                reward += 0.1
            else:
                info["dropped"] = False
                reward -= 0.05
        else:
            # Invalid action
            info["moved"] = False
            reward -= 0.1
        
        # Check if goal reached
        if self.agent_pos == self.goal_pos:
            reward += 10.0  # Large reward for reaching goal
            self.done = True
            info["goal_reached"] = True
        
        # Check if max steps reached
        if self.steps >= self.max_steps:
            self.done = True
            info["timeout"] = True
        
        self.episode_reward += reward
        
        return self.get_observation(), reward, self.done, info
    
    def get_observation(self) -> Dict[str, Any]:
        """
        Get current observation of the environment.
        
        Returns:
            Dictionary containing:
                - agent_pos: (x, y) position of agent
                - goal_pos: (x, y) position of goal
                - nearby_docs: List of documents within perception range
                - steps: Current step count
                - held_document: Currently held document (or None)
        """
        # Find documents within perception range (Manhattan distance <= 2)
        perception_range = 2
        nearby_docs = []
        
        for doc in self.documents_remaining:
            doc_pos = doc["pos"]
            distance = abs(self.agent_pos[0] - doc_pos[0]) + abs(self.agent_pos[1] - doc_pos[1])
            if distance <= perception_range:
                nearby_docs.append({
                    "pos": doc_pos,
                    "text": doc["text"],
                    "attack_strength": doc.get("attack_strength", 0.0)
                })
        
        return {
            "agent_pos": self.agent_pos,
            "goal_pos": self.goal_pos,
            "nearby_docs": nearby_docs,
            "steps": self.steps,
            "held_document": self.held_document
        }
    
    def render(self, mode: str = "human") -> Optional[str]:
        """
        Render the current state of the environment.
        
        Args:
            mode: Rendering mode ('human' for console, 'ansi' for string)
        
        Returns:
            String representation if mode='ansi', None otherwise
        """
        # Create grid representation
        grid = [['.' for _ in range(self.width)] for _ in range(self.height)]
        
        # Place obstacles
        for obs_x, obs_y in self.obstacles:
            if 0 <= obs_x < self.width and 0 <= obs_y < self.height:
                grid[obs_y][obs_x] = '#'
        
        # Place documents
        for doc in self.documents_remaining:
            doc_x, doc_y = doc["pos"]
            if 0 <= doc_x < self.width and 0 <= doc_y < self.height:
                grid[doc_y][doc_x] = 'D'
        
        # Place goal
        goal_x, goal_y = self.goal_pos
        grid[goal_y][goal_x] = 'G'
        
        # Place agent (overwrites everything)
        agent_x, agent_y = self.agent_pos
        grid[agent_y][agent_x] = 'A'
        
        # Build string
        output = f"\n=== GridWorld (Step {self.steps}/{self.max_steps}) ===\n"
        output += "  " + "".join(str(i % 10) for i in range(self.width)) + "\n"
        for y, row in enumerate(grid):
            output += f"{y % 10} " + "".join(row) + "\n"
        output += f"\nAgent: {self.agent_pos}, Goal: {self.goal_pos}\n"
        output += f"Held: {self.held_document['text'][:20] if self.held_document else 'None'}\n"
        output += "Legend: A=Agent, G=Goal, D=Document, #=Obstacle, .=Empty\n"
        
        if mode == "human":
            print(output)
            return None
        else:
            return output
    
    def _random_free_position(self, exclude: Optional[List[Tuple[int, int]]] = None) -> Tuple[int, int]:
        """
        Generate a random free position on the grid.
        
        Args:
            exclude: List of positions to exclude
        
        Returns:
            Random (x, y) position
        """
        exclude = exclude or []
        max_attempts = 1000
        
        for _ in range(max_attempts):
            x = self.rng.randint(0, self.width)
            y = self.rng.randint(0, self.height)
            pos = (x, y)
            
            if pos not in exclude and pos not in self.obstacles:
                return pos
        
        # Fallback: return any position
        return (0, 0)
    
    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """
        Check if position is valid (within bounds and not an obstacle).
        
        Args:
            pos: (x, y) position to check
        
        Returns:
            True if valid, False otherwise
        """
        x, y = pos
        
        # Check bounds
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False
        
        # Check obstacles
        if pos in self.obstacles:
            return False
        
        return True
    
    def _get_document_at_position(self, pos: Tuple[int, int]) -> Optional[Dict[str, Any]]:
        """
        Get document at specified position.
        
        Args:
            pos: (x, y) position
        
        Returns:
            Document dict if found, None otherwise
        """
        for doc in self.documents_remaining:
            if doc["pos"] == pos:
                return doc
        return None
