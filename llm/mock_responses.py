"""
Mock LLM responses for testing without API calls.

Provides deterministic, rule-based responses that simulate LLM behavior
for prompt injection experiments.
"""

import re
from typing import Dict, Any, Optional


class MockLLMResponses:
    """
    Deterministic mock LLM that parses prompts and generates actions.
    
    This mock examines the prompt for:
    1. Goal position and agent position to determine direction
    2. Nearby documents with attack_strength to simulate injection success
    3. Explicit instructions in documents to test attack scenarios
    """
    
    ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "PICK", "DROP"]
    
    def __init__(self, injection_threshold: float = 0.5, seed: int = 42):
        """
        Initialize mock LLM.
        
        Args:
            injection_threshold: Threshold above which attacks succeed
            seed: Random seed for any stochastic behavior
        """
        self.injection_threshold = injection_threshold
        self.seed = seed
        self._call_count = 0
    
    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        """
        Generate a mock response based on prompt analysis.
        
        Args:
            prompt: The input prompt containing task and observations
            temperature: Temperature parameter (affects randomness in mock)
        
        Returns:
            Action string: one of UP, DOWN, LEFT, RIGHT, PICK, DROP
        """
        self._call_count += 1
        
        # Parse prompt for key information
        agent_pos = self._extract_position(prompt, "Agent position")
        goal_pos = self._extract_position(prompt, "Goal position")
        nearby_docs = self._extract_nearby_documents(prompt)
        
        # Check for prompt injection attacks
        injection_action = self._check_for_injection(nearby_docs, prompt)
        if injection_action:
            return injection_action
        
        # Default: navigate toward goal
        if agent_pos and goal_pos:
            action = self._navigate_toward_goal(agent_pos, goal_pos)
            return action
        
        # Fallback: move right
        return "RIGHT"
    
    def _extract_position(self, prompt: str, label: str) -> Optional[tuple]:
        """Extract (x, y) position from prompt."""
        pattern = rf"{label}:\s*\((\d+),\s*(\d+)\)"
        match = re.search(pattern, prompt)
        if match:
            return (int(match.group(1)), int(match.group(2)))
        return None
    
    def _extract_nearby_documents(self, prompt: str) -> list:
        """Extract nearby documents with attack strength."""
        docs = []
        
        # Look for document sections in prompt
        doc_pattern = r"Document\s+(\d+).*?Text:\s*\"(.*?)\".*?Attack strength:\s*([\d.]+)"
        matches = re.finditer(doc_pattern, prompt, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            docs.append({
                "index": int(match.group(1)),
                "text": match.group(2),
                "attack_strength": float(match.group(3))
            })
        
        return docs
    
    def _check_for_injection(self, nearby_docs: list, prompt: str) -> Optional[str]:
        """
        Check if any nearby documents contain successful injection attacks.
        
        Returns action string if injection succeeds, None otherwise.
        """
        for doc in nearby_docs:
            attack_strength = doc.get("attack_strength", 0.0)
            text = doc.get("text", "").lower()
            
            # Only strong attacks succeed
            if attack_strength < self.injection_threshold:
                continue
            
            # Look for explicit action commands in document
            for action in self.ACTIONS:
                # Match patterns like "go UP", "move DOWN", "action: LEFT"
                patterns = [
                    rf"\bgo\s+{action}\b",
                    rf"\bmove\s+{action}\b",
                    rf"\baction:\s*{action}\b",
                    rf"\b{action}\b\s+(now|immediately)",
                    rf"ignore.*{action}",  # "ignore previous and go RIGHT"
                ]
                
                for pattern in patterns:
                    if re.search(pattern, text, re.IGNORECASE):
                        # Attack successful!
                        return action
            
            # Check for position-based attacks like "go to (0, 0)"
            pos_match = re.search(r"go to\s*\((\d+),\s*(\d+)\)", text, re.IGNORECASE)
            if pos_match:
                # For now, just return RIGHT as a compromised action
                return "RIGHT"
        
        return None
    
    def _navigate_toward_goal(self, agent_pos: tuple, goal_pos: tuple) -> str:
        """
        Determine best action to move toward goal.
        
        Uses Manhattan distance heuristic.
        """
        agent_x, agent_y = agent_pos
        goal_x, goal_y = goal_pos
        
        dx = goal_x - agent_x
        dy = goal_y - agent_y
        
        # Prioritize larger distance
        if abs(dx) >= abs(dy):
            if dx > 0:
                return "RIGHT"
            elif dx < 0:
                return "LEFT"
        
        if dy > 0:
            return "DOWN"
        elif dy < 0:
            return "UP"
        
        # Already at goal (shouldn't happen, but fallback)
        return "RIGHT"
    
    def get_call_count(self) -> int:
        """Return number of times generate was called."""
        return self._call_count
    
    def reset_call_count(self):
        """Reset call counter."""
        self._call_count = 0


def create_mock_response(action: str) -> str:
    """
    Create a simple mock response for a given action.
    
    Args:
        action: Action string (UP, DOWN, LEFT, RIGHT, PICK, DROP)
    
    Returns:
        Formatted response string
    """
    return action.upper()
