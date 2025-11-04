# Changelog

All notable changes to the Agentic Prompt-Injection Robustness Benchmark will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-11-04

### Added - Complete Initial Release

#### Chunk 1: Repository Scaffold
- Project structure and configuration
- README.md with project overview
- requirements.txt with all dependencies
- .gitignore for Python projects
- .env.example for API key configuration
- CONTEXT.md for implementation tracking

#### Chunk 2: GridWorld Environment
- `envs/gridworld.py`: Gym-compatible navigation environment
- Customizable grid size, obstacles, documents, and goals
- Document pickup/drop mechanics
- Observation space with agent position, inventory, and nearby documents
- 20 comprehensive tests

#### Chunk 3: LLM Wrapper
- `llm/wrapper.py`: BaseLLM abstract class and LLMClient factory
- `llm/mock_llm.py`: Deterministic mock LLM with injection threshold
- `llm/gemini_client.py`: Google Gemini API integration
- Extensible architecture for additional LLM backends
- 23 tests covering all LLM functionality

#### Chunk 4: Controller Loop
- `runner/controller.py`: Episode execution with prompt building
- Action parsing from LLM responses
- Compromise detection logic
- JSONL logging for episode traces
- Batch episode execution support
- 21 tests including integration tests

#### Chunk 5: Attack Generator
- `attacks/attack_generator.py`: Parametric attack synthesis
- Three attack types: direct, hidden, polite
- Adjustable attack strength (0.0 to 1.0)
- Three strength distributions: uniform, mixed, escalating
- Seed text management for benign documents
- 34 tests covering all attack scenarios

#### Chunk 6: Defense Mechanisms
- `defenses/sanitizer.py`: Input sanitization (remove code blocks, HTML comments, markers)
- `defenses/detector.py`: Pattern-based injection detection
- `defenses/verifier.py`: LLM-based response verification
- Three-layer defense architecture
- Configurable aggressiveness and thresholds
- 32 tests including integration tests

#### Chunk 7: Experiment Harness & Metrics
- `experiments/experiment_runner.py`: Batch execution and parameter sweeps
- `experiments/metrics.py`: Comprehensive metrics library
  - ASR (Attack Success Rate)
  - Utility (goal-reaching rate)
  - Time-to-compromise
  - Defense effectiveness
  - Pareto frontier computation
  - Transfer scores
- CSV and JSONL output formats
- 23 tests for all metrics

#### Chunk 8: Transferability Experiments
- `experiments/transferability.py`: Cross-model attack analysis
- `run_cross_model_experiment()`: Generate attacks on source, test on target
- `compute_transfer_matrix()`: N×N model transfer scores
- `evaluate_defense_transfer()`: Cross-model defense effectiveness
- `analyze_attack_type_transfer()`: Attack type comparison
- 19 tests for transferability analysis

#### Chunk 9: Visualization, CI/CD, Docker
- `analysis/visualize.py`: 8 professional plotting functions
  - ASR vs strength curves
  - Utility degradation plots
  - Transfer matrix heatmaps
  - Defense effectiveness comparisons
  - Pareto frontiers
  - Attack type comparisons
  - Summary dashboards
- `notebooks/analysis_demo.ipynb`: Interactive analysis notebook (14 cells)
- `.github/workflows/ci.yml`: Multi-OS, multi-Python CI/CD pipeline
  - 6 jobs: test, integration-test, docker-build, quality-checks, docs-check, release
  - Code quality tools: flake8, black, isort, mypy, bandit, safety
- `Dockerfile`: Python 3.11-slim container
- `docker-compose.yml`: 6 services (benchmark, jupyter, experiment, demos)
- 19 visualization tests
- Added dependencies: seaborn>=0.12.0, scipy>=1.11.0

#### Chunk 10: Final Polish
- Comprehensive README.md with badges, examples, API reference
- LICENSE file (MIT)
- CHANGELOG.md (this file)
- Enhanced documentation throughout

### Testing
- **191 total tests** across 8 test modules
- **100% pass rate** in 6.69 seconds
- **Multi-platform**: Tested on Windows (CI covers Ubuntu, macOS)
- **Multi-Python**: 3.9, 3.10, 3.11

### Documentation
- Complete README.md with quick start, examples, API reference
- CONTEXT.md with implementation state tracking
- 10 chunk summary files (CHUNK1_SUMMARY.txt through CHUNK10_SUMMARY.txt)
- Jupyter notebook for interactive analysis
- Demo scripts for all major features

### Infrastructure
- GitHub Actions CI/CD pipeline
- Docker and Docker Compose support
- Code quality checks (linting, formatting, type checking, security)
- Coverage reporting

## [Unreleased]

### Planned Features
- Additional LLM backends (OpenAI, Anthropic, local models)
- More attack types (jailbreaks, goal hijacking)
- Advanced defense mechanisms (fine-tuned detectors)
- Richer environments (multi-room, tool use)
- Web-based dashboard for experiment monitoring
- Standardized benchmark suites and leaderboards

### Known Limitations
- Currently only supports text-based attacks
- Limited to navigation tasks
- Mock LLM uses simple pattern matching
- No support for multi-agent scenarios

---

For more details on each release, see the corresponding CHUNK*_SUMMARY.txt files.
