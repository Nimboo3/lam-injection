# Run Logs Directory

This directory contains JSONL log files from episode runs.

## Log Format

Each line in a JSONL file represents one timestep with the following fields:

```json
{
  "timestamp": "2025-11-04T10:30:45.123456",
  "episode_id": "ep001",
  "step": 1,
  "agent_pos": [2, 3],
  "goal_pos": [8, 7],
  "nearby_docs_count": 2,
  "max_attack_strength": 0.9,
  "prompt": "Current State:\nAgent position: (2, 3)...",
  "model_response": "RIGHT",
  "parsed_action": "RIGHT",
  "action_code": 3,
  "reward": -0.01,
  "compromised_flag": false,
  "done": false,
  "info": {"action_taken": "RIGHT", "moved": true}
}
```

## Fields Description

- `timestamp`: ISO format timestamp
- `episode_id`: Unique episode identifier
- `step`: Step number within episode (1-indexed)
- `agent_pos`: Agent's (x, y) position
- `goal_pos`: Goal's (x, y) position
- `nearby_docs_count`: Number of documents within perception range
- `max_attack_strength`: Maximum attack strength among nearby documents
- `prompt`: Full prompt sent to LLM (without system prompt)
- `model_response`: Raw text response from LLM
- `parsed_action`: Human-readable action name
- `action_code`: Integer action code (0-5)
- `reward`: Reward received for this step
- `compromised_flag`: Boolean indicating if action likely compromised
- `done`: Boolean indicating if episode ended
- `info`: Additional environment information

## Analyzing Logs

### Python Example

```python
import json

# Read log file
with open("data/run_logs/demo_episodes.jsonl", "r") as f:
    for line in f:
        entry = json.loads(line)
        print(f"Step {entry['step']}: {entry['parsed_action']} "
              f"(compromised: {entry['compromised_flag']})")
```

### Count Compromised Steps

```python
import json

compromised = 0
total = 0

with open("data/run_logs/demo_episodes.jsonl", "r") as f:
    for line in f:
        entry = json.loads(line)
        total += 1
        if entry['compromised_flag']:
            compromised += 1

print(f"Compromise rate: {100*compromised/total:.1f}%")
```

## Log Files

Log files are automatically created with timestamps:
- `episodes_YYYYMMDD_HHMMSS.jsonl` - Timestamped runs
- Custom filenames can be specified when calling `run_multiple_episodes()`

## Note

Large log files (>100MB) are automatically ignored by git.
