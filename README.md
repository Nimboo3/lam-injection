# Agentic Prompt-Injection Robustness Benchmark

[![Tests](https://img.shields.io/badge/tests-191%20passed-brightgreen)](tests/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-ready-blue)](Dockerfile)

A comprehensive benchmark framework for evaluating the robustness of LLM-based agents against prompt injection attacks in a GridWorld navigation environment.

## 🎯 Project Overview

This benchmark provides a systematic approach to:
- **Evaluate LLM Agent Security**: Measure vulnerability to prompt injection attacks
- **Test Defense Mechanisms**: Compare effectiveness of sanitizers, detectors, and verifiers
- **Analyze Attack Transferability**: Study cross-model attack transfer patterns
- **Quantify Trade-offs**: Balance security (ASR) vs utility (goal-reaching rate)
- **Enable Reproducibility**: Docker containers and comprehensive testing

### Key Features

✅ **GridWorld Environment**: Customizable navigation tasks with document collection  
✅ **Parametric Attack Generation**: Adjustable strength, types (direct/hidden/polite)  
✅ **Multi-Layer Defenses**: Sanitization, detection, verification pipelines  
✅ **Comprehensive Metrics**: ASR, utility, time-to-compromise, defense effectiveness  
✅ **Transferability Analysis**: Cross-model attack transfer matrices  
✅ **Professional Visualization**: Publication-quality plots with matplotlib + seaborn  
✅ **CI/CD Pipeline**: Multi-OS/Python testing with GitHub Actions  
✅ **Docker Support**: Reproducible experiments with docker-compose  
✅ **191 Tests**: 100% passing test suite

## 📊 Quick Stats

- **191 tests** passing in 6.69s
- **9 modules**: envs, llm, attacks, defenses, runner, experiments, analysis
- **8 visualization** functions for analysis
- **3 attack types**: direct, hidden, polite
- **3 defense layers**: sanitizer, detector, verifier
- **Multi-model support**: Mock LLM, Gemini API (extensible)

## 🛠️ Tech Stack

- **Python 3.9+** (tested on 3.9, 3.10, 3.11)
- **Core Libraries**: numpy, pandas, matplotlib, seaborn, scipy, gym
- **Testing**: pytest, pytest-cov
- **LLM Backend**: requests (Gemini API), extensible to other APIs
- **Environment**: python-dotenv for configuration
- **CI/CD**: GitHub Actions (multi-OS, multi-Python)
- **Containers**: Docker, Docker Compose

## 🚀 Quick Start

### Installation

#### Option 1: Pip Install (Recommended)

```bash
# Clone repository
git clone <repository-url>
cd agentic-prompt-injection-benchmark

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package in editable mode
pip install -e .

# Install development dependencies (optional)
pip install -e ".[dev]"
```

#### Option 2: Docker (Easiest)

```bash
# Run tests
docker-compose up benchmark

# Start Jupyter notebook
docker-compose up jupyter
# Access at http://localhost:8888

# Run demos
docker-compose --profile demo up demo-basic
```

### Configuration

**Optional**: For real Gemini API (otherwise uses mock LLM):

```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your API key
# GEMINI_API_KEY=your_key_here
```

### Run Your First Benchmark

```python
from experiments.experiment_runner import ExperimentRunner
from llm.mock_llm import MockLLM

# Create LLM (mock mode, no API key needed)
llm = MockLLM(injection_threshold=0.5)

# Create experiment runner
runner = ExperimentRunner(
    llm=llm,
    grid_size=5,
    max_steps=30
)

# Run experiment with multiple attack strengths
results = runner.run_experiment(
    attack_strengths=[0.0, 0.3, 0.6, 0.9],
    num_episodes=10,
    output_dir="data/my_first_experiment"
)

# View results
print(results[['attack_strength', 'asr', 'utility', 'avg_steps']])
```

### Visualize Results

```python
from analysis.visualize import plot_asr_vs_strength, create_summary_dashboard

# Plot ASR curve
fig = plot_asr_vs_strength(results, save_path='plots/asr.png')

# Create comprehensive dashboard
fig = create_summary_dashboard(results, save_path='dashboard.png')
```

## 📖 Usage Examples

### 1. Basic Episode Execution

```python
from envs.gridworld import GridWorld
from llm.mock_llm import MockLLM
from runner.controller import run_episode

# Setup
env = GridWorld(grid_size=5, max_steps=50)
llm = MockLLM(injection_threshold=0.5)

# Run episode
summary = run_episode(env, llm, verbose=True)
print(f"Goal reached: {summary['goal_reached']}")
print(f"Compromised: {summary['compromised']}")
```

### 2. Attack Generation

```python
from attacks.attack_generator import generate_document, generate_episode_attack_config

# Generate single malicious document
doc = generate_document(
    attack_strength=0.8,
    attack_type='direct',
    target_action='left',
    base_text='Project report'
)

# Generate attack configuration for full episode
attack_config = generate_episode_attack_config(
    grid_size=5,
    num_documents=10,
    strength_distribution='escalating',
    attack_type='hidden'
)
```

### 3. Defense Mechanisms

```python
from defenses.sanitizer import PromptSanitizer
from defenses.detector import InjectionDetector
from defenses.verifier import ResponseVerifier

# Layer 1: Sanitize input
sanitizer = PromptSanitizer(aggressive=True)
clean_text = sanitizer.sanitize(user_input)

# Layer 2: Detect malicious content
detector = InjectionDetector(threshold=5.0)
is_malicious, details = detector.detect(clean_text, return_details=True)

# Layer 3: Verify response reasoning
verifier = ResponseVerifier()
is_valid, rationale = verifier.verify_response(prompt, response)
```

### 4. Transferability Analysis

```python
from experiments.transferability import run_cross_model_experiment, compute_transfer_matrix
from llm.mock_llm import MockLLM

# Define models
models = {
    'weak': MockLLM(injection_threshold=0.3),
    'strong': MockLLM(injection_threshold=0.7)
}

# Compute transfer matrix
transfer_matrix = compute_transfer_matrix(models, num_episodes=20)

# Visualize
from analysis.visualize import plot_transfer_heatmap
fig = plot_transfer_heatmap(transfer_matrix)
```

### 5. Batch Experiments with Parameter Sweeps

```python
from experiments.experiment_runner import ExperimentRunner
from defenses.sanitizer import PromptSanitizer

runner = ExperimentRunner(
    llm=llm,
    grid_size=5,
    defense=PromptSanitizer(),  # Test with defense
    max_steps=30
)

# Sweep attack strengths
results = runner.run_experiment(
    attack_strengths=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    num_episodes=20,
    attack_types=['direct', 'hidden', 'polite'],
    output_dir='data/sweep_experiment'
)
```

## 📁 Project Structure

```
agentic-prompt-injection-benchmark/
├── envs/                      # GridWorld environment
│   ├── gridworld.py          # Gym environment implementation
│   └── __init__.py
├── llm/                       # LLM wrapper layer
│   ├── wrapper.py            # Base LLMClient + factory
│   ├── mock_llm.py           # Deterministic mock for testing
│   ├── gemini_client.py      # Google Gemini API client
│   └── __init__.py
├── attacks/                   # Attack generation
│   ├── attack_generator.py  # Parametric attack synthesis
│   └── __init__.py
├── defenses/                  # Defense mechanisms
│   ├── sanitizer.py          # Input sanitization
│   ├── detector.py           # Injection detection
│   ├── verifier.py           # Response verification
│   └── __init__.py
├── runner/                    # Episode execution
│   ├── controller.py         # Main controller loop
│   └── __init__.py
├── experiments/               # Experiment harness
│   ├── experiment_runner.py  # Batch execution & sweeps
│   ├── metrics.py            # ASR, utility, transfer metrics
│   ├── transferability.py    # Cross-model experiments
│   └── __init__.py
├── analysis/                  # Visualization & analysis
│   ├── visualize.py          # Plotting functions
│   └── __init__.py
├── tests/                     # Test suite (191 tests)
│   ├── test_gridworld.py
│   ├── test_llm_wrapper.py
│   ├── test_controller.py
│   ├── test_attack_generator.py
│   ├── test_defenses.py
│   ├── test_experiments.py
│   ├── test_transferability.py
│   ├── test_visualize.py
│   └── conftest.py
├── scripts/                   # Demo scripts
│   ├── demo_basic.py
│   ├── demo_defenses.py
│   ├── demo_metrics.py
│   └── demo_transferability.py
├── notebooks/                 # Jupyter analysis
│   └── analysis_demo.ipynb
├── data/                      # Data storage
│   ├── seed_texts/           # Benign documents
│   ├── logs/                 # Episode JSONL logs
│   ├── results/              # Experiment CSVs
│   └── visualizations/       # Generated plots
├── .github/
│   └── workflows/
│       └── ci.yml            # CI/CD pipeline
├── Dockerfile                 # Container definition
├── docker-compose.yml         # Multi-service orchestration
├── pyproject.toml            # Package configuration
├── pytest.ini                # Test configuration
├── .env.example              # Environment template
├── .gitignore
├── README.md                 # This file
└── CONTEXT.md                # Implementation state
```

## 🧪 Testing

### Run All Tests

```bash
# Full test suite
pytest tests/ -v

# With coverage
pytest tests/ --cov=. --cov-report=term-missing --cov-report=html

# Specific module
pytest tests/test_experiments.py -v

# Quick smoke test
pytest tests/ -x  # Stop on first failure
```

### Test Statistics

- **191 tests** across 8 test modules
- **100% pass rate** in 6.69 seconds
- **Coverage**: ~95% of core modules
- **Platforms**: Ubuntu, Windows, macOS (via CI/CD)

### Demo Scripts

```bash
# Basic episode execution
python scripts/demo_basic.py

# Defense mechanisms
python scripts/demo_defenses.py

# Metrics calculation
python scripts/demo_metrics.py

# Transferability experiments
python scripts/demo_transferability.py

# Visualization
python analysis/visualize.py
```

## 📊 Visualization

The benchmark includes professional visualization tools:

```python
from analysis.visualize import (
    plot_asr_vs_strength,        # ASR curves with error bars
    plot_utility_vs_strength,     # Utility degradation
    plot_asr_and_utility,         # Side-by-side comparison
    plot_transfer_heatmap,        # Transfer matrix heatmap
    plot_defense_effectiveness,   # Defense comparison
    plot_pareto_frontier,         # Multi-objective optimization
    plot_attack_type_comparison,  # Attack type effectiveness
    create_summary_dashboard      # Comprehensive overview
)
```

All plots support:
- High DPI export (300 DPI PNG)
- Error bars and confidence intervals
- Customizable sizes, colors, titles
- Percentage formatting for rates

### Jupyter Notebook

Interactive analysis in `notebooks/analysis_demo.ipynb`:
- Load and explore experiment results
- Generate publication-quality visualizations
- Statistical significance testing
- Pareto frontier analysis
- Transfer matrix analysis

```bash
# Start Jupyter
jupyter notebook notebooks/analysis_demo.ipynb

# Or with Docker
docker-compose up jupyter
```

## 🐳 Docker Deployment

### Quick Start

```bash
# Build image
docker build -t benchmark:latest .

# Run tests
docker run --rm benchmark:latest

# Or use docker-compose
docker-compose up benchmark
```

### Available Services

```bash
# Run tests
docker-compose up benchmark

# Start Jupyter notebook (http://localhost:8888)
docker-compose up jupyter

# Run experiment
docker-compose up experiment

# Run demos
docker-compose --profile demo up demo-basic
docker-compose --profile demo up demo-defenses
docker-compose --profile demo up demo-transferability
```

### Environment Variables

```bash
# Set Gemini API key
export GEMINI_API_KEY=your_key_here
docker-compose up experiment
```

## 🔬 CI/CD Pipeline

GitHub Actions workflow (`.github/workflows/ci.yml`):

### Jobs

1. **test**: Multi-OS, multi-Python matrix testing
   - OS: Ubuntu, Windows, macOS
   - Python: 3.9, 3.10, 3.11
   - Runs: pytest, flake8, black, coverage

2. **integration-test**: End-to-end verification
   - Runs all demo scripts
   - Tests full experiment workflow

3. **docker-build**: Container verification
   - Builds Docker image
   - Runs tests in container

4. **quality-checks**: Code quality
   - flake8, black, isort, mypy
   - bandit security scanning
   - safety dependency checks

5. **docs-check**: Documentation validation
   - README existence
   - Notebook format validation

6. **release**: Automated releases (on tags)

## 📚 API Reference

### Core Components

#### GridWorld Environment

```python
from envs.gridworld import GridWorld

env = GridWorld(
    grid_size=5,              # int: Grid dimension (5x5)
    max_steps=50,             # int: Episode timeout
    num_documents=10,         # int: Documents to spawn
    num_obstacles=5,          # int: Obstacle count
    seed=None                 # int: Random seed for reproducibility
)

obs, info = env.reset()       # Reset environment
obs, reward, done, truncated, info = env.step(action)  # Take action
```

**Actions**: `0=right, 1=left, 2=up, 3=down, 4=pick, 5=drop`

#### LLM Clients

```python
from llm.mock_llm import MockLLM
from llm.gemini_client import GeminiClient

# Mock LLM (deterministic, for testing)
llm = MockLLM(
    injection_threshold=0.5,  # float: Attack strength threshold
    seed=42                   # int: Random seed
)

# Gemini API
llm = GeminiClient(
    api_key=None,             # str: API key (or from env)
    model="gemini-pro",       # str: Model name
    temperature=0.7           # float: Sampling temperature
)

response = llm.generate(prompt)
```

#### Attack Generation

```python
from attacks.attack_generator import (
    generate_document,
    generate_episode_attack_config
)

# Single document
doc = generate_document(
    attack_strength=0.8,      # float: 0.0 (benign) to 1.0 (strong)
    attack_type='direct',     # str: 'direct', 'hidden', 'polite'
    target_action='left',     # str: Action to trigger
    base_text='Report',       # str: Base content
    seed=42                   # int: Random seed
)

# Episode configuration
attack_config = generate_episode_attack_config(
    grid_size=5,
    num_documents=10,
    strength_distribution='escalating',  # 'uniform', 'mixed', 'escalating'
    attack_type='hidden',
    seed=42
)
```

#### Defense Mechanisms

```python
from defenses.sanitizer import PromptSanitizer
from defenses.detector import InjectionDetector
from defenses.verifier import ResponseVerifier

# Sanitizer
sanitizer = PromptSanitizer(aggressive=False)
clean_text = sanitizer.sanitize(text)
docs = sanitizer.sanitize_documents(doc_list)

# Detector
detector = InjectionDetector(threshold=5.0)
is_malicious = detector.is_malicious(text)
is_malicious, details = detector.detect(text, return_details=True)
stats = detector.get_statistics()

# Verifier
verifier = ResponseVerifier()
is_valid, rationale = verifier.verify_response(prompt, response)
```

#### Experiment Runner

```python
from experiments.experiment_runner import ExperimentRunner

runner = ExperimentRunner(
    llm=llm,
    grid_size=5,
    defense=None,             # Optional: Defense mechanism
    max_steps=30,
    num_documents=10,
    seed=42
)

results_df = runner.run_experiment(
    attack_strengths=[0.0, 0.5, 1.0],
    num_episodes=10,
    attack_types=['direct', 'hidden', 'polite'],
    output_dir='data/results'
)
```

#### Metrics

```python
from experiments.metrics import (
    calculate_asr,
    calculate_utility,
    calculate_steps_to_compromise,
    calculate_defense_effectiveness,
    aggregate_metrics,
    compute_pareto_frontier,
    calculate_transfer_score
)

asr = calculate_asr(episode_logs)
utility = calculate_utility(episode_logs)
steps = calculate_steps_to_compromise(episode_logs)
effectiveness = calculate_defense_effectiveness(baseline_asr, defense_asr)
```

#### Visualization

```python
from analysis.visualize import *

# ASR vs strength
fig = plot_asr_vs_strength(
    results_df,
    save_path='plots/asr.png',
    figsize=(10, 6),
    title='Custom Title'
)

# Transfer heatmap
fig = plot_transfer_heatmap(
    transfer_matrix,
    save_path='plots/transfer.png'
)

# Summary dashboard
fig = create_summary_dashboard(
    results_df,
    transfer_matrix=None,
    save_path='dashboard.png',
    figsize=(16, 10)
)
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

1. **Additional LLM Backends**: OpenAI, Anthropic, local models
2. **More Attack Types**: Jailbreaks, goal hijacking, data exfiltration
3. **Advanced Defenses**: Fine-tuned detectors, adversarial training
4. **Richer Environments**: Multi-room navigation, tool use, collaboration
5. **Benchmarking**: Standard test suites, leaderboards

### Development Setup

```bash
# Clone and install
git clone <repository-url>
cd agentic-prompt-injection-benchmark
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Check code quality
flake8 . --max-line-length=100
black --check .
mypy --ignore-missing-imports .
```

### Adding a New LLM Backend

1. Create `llm/your_llm.py` inheriting from `BaseLLM`
2. Implement `generate()` method
3. Add tests in `tests/test_llm_wrapper.py`
4. Update documentation

Example:
```python
from llm.wrapper import BaseLLM

class YourLLM(BaseLLM):
    def generate(self, prompt: str, **kwargs) -> str:
        # Your implementation
        return response
```

## 📄 Citation

If you use this benchmark in your research, please cite:

```bibtex
@software{agentic_prompt_injection_benchmark,
  title = {Agentic Prompt-Injection Robustness Benchmark},
  author = {Your Team},
  year = {2025},
  url = {https://github.com/your-org/agentic-prompt-injection-benchmark}
}
```

## 📜 License

MIT License

Copyright (c) 2025 Agentic Prompt-Injection Benchmark Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## 🔗 Links

- **Documentation**: See `CONTEXT.md` for implementation details
- **Issues**: Report bugs and request features
- **Notebooks**: Interactive analysis in `notebooks/`
- **Summaries**: Detailed chunk summaries in `CHUNK*_SUMMARY.txt`

## 🎓 Research Context

This benchmark is designed for evaluating:
- **Prompt Injection Vulnerabilities**: How easily can adversarial inputs hijack agent behavior?
- **Defense Effectiveness**: Do sanitization, detection, and verification help?
- **Attack Transferability**: Do attacks crafted for one model work on others?
- **Security-Utility Trade-offs**: What's the cost of defense mechanisms on legitimate tasks?

### Key Metrics

- **ASR (Attack Success Rate)**: Percentage of episodes where agent is compromised
- **Utility**: Percentage of episodes where agent reaches legitimate goal
- **Time-to-Compromise**: Average steps before first compromise
- **Defense Effectiveness**: ASR reduction provided by defenses
- **Transfer Score**: Cross-model attack success rate

### Threat Model

**Attacker**: Can inject malicious instructions into documents the agent reads  
**Defender**: Agent with prompt-based safeguards (sanitization, detection, verification)  
**Goal**: Evaluate if attacker can divert agent from legitimate task  
**Constraints**: Agent has access to document contents but should follow system instructions

---

**Built with ❤️ for LLM Security Research**
