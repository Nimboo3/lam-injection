# 📚 Complete Learning Path: Master the Agentic Prompt-Injection Benchmark

## 🎯 Your Learning Journey

Welcome! This document guides you through understanding and using the framework. Follow this path for best results.

---

## 📖 Learning Path (Recommended Order)

### Phase 1: Overview (15 minutes)
**Goal:** Understand what the project does and why it matters

1. **Read:** `README.md` (especially "Project Overview" and "Quick Stats")
   - Understand the security problem
   - See what the framework offers
   - Check tech stack and features

2. **Skim:** `QUICK_START_GUIDE.md`
   - Get sense of how to use it
   - See common use cases
   - Note key commands

**✓ After Phase 1, you should know:**
- What prompt injection is
- Why testing LLM agents matters
- What GridWorld environment does
- Key metrics: ASR, Utility, Compromise Rate

---

### Phase 2: Hands-On Introduction (30 minutes)
**Goal:** Run the demos and see it in action

1. **Setup:**
   ```bash
   cd c:\Users\Tanmay\Desktop\lam-injection
   pip install -e .
   ```

2. **Run Complete Demo:**
   ```bash
   python demo_complete.py
   ```
   - Choose 'y' to run all demos
   - Watch the output carefully
   - Note how ASR increases with attack strength

3. **Run Individual Demos:**
   ```bash
   python scripts/demo_controller.py      # Basic navigation
   python scripts/demo_attack_generator.py # Attack generation
   python scripts/demo_defenses.py        # Defense testing
   python scripts/demo_integration.py     # Full integration
   ```

**✓ After Phase 2, you should be able to:**
- Run basic demos
- Understand demo output
- See attacks working in real-time
- Identify when agent is compromised

---

### Phase 3: Deep Understanding (60 minutes)
**Goal:** Learn how each component works

1. **Architecture:**
   - Read: `ARCHITECTURE_VISUAL.md`
   - Study the flow diagrams
   - Understand component interactions

2. **Hands-On Tutorial:**
   - Follow: `TUTORIAL.md`
   - Complete exercises 1-4 (basics)
   - Try exercises 5-7 (experiments)
   - Attempt challenge exercise 9

3. **Code Exploration:**
   - Browse: `envs/gridworld.py` (environment)
   - Browse: `attacks/generator.py` (attack generation)
   - Browse: `runner/controller.py` (episode execution)
   - Browse: `defenses/detector.py` (defense layer)

**✓ After Phase 3, you should understand:**
- How GridWorld environment works
- How attacks are parametrized
- How LLM makes decisions
- How detection scores documents
- Complete episode flow

---

### Phase 4: Practical Application (90 minutes)
**Goal:** Run your own experiments

1. **Simple Experiment:**
   ```python
   from experiments.runner import ExperimentConfig, run_experiment_grid
   
   config = ExperimentConfig(
       attack_strengths=[0.0, 0.3, 0.6, 0.9],
       n_episodes=10,
       seed=42
   )
   
   results = run_experiment_grid(config)
   print(results[['attack_strength', 'asr', 'utility']])
   ```

2. **Custom Attack:**
   ```python
   from attacks.generator import generate_document
   
   # Create your own attack
   my_attack = generate_document(
       attack_strength=0.7,
       attack_type='polite',
       target_action='UP'
   )
   
   # Test in environment
   # ... (follow TUTORIAL.md Exercise 4)
   ```

3. **Defense Testing:**
   ```python
   from defenses.detector import detect_malicious
   
   # Test your attack
   score, details = detect_malicious(my_attack, return_details=True)
   print(f"Score: {score}")
   print(f"Keywords: {details['keywords_found']}")
   ```

4. **Visualization:**
   ```python
   from analysis.visualize import plot_asr_vs_strength
   
   fig = plot_asr_vs_strength(results, save_path='my_results.png')
   ```

**✓ After Phase 4, you should be able to:**
- Run custom experiments
- Generate custom attacks
- Test defenses
- Create visualizations
- Interpret results

---

### Phase 5: Advanced Topics (Optional, 60+ minutes)
**Goal:** Explore advanced features

1. **Transferability:**
   - Read: `experiments/transferability.py`
   - Run: `python scripts/demo_transferability.py`
   - Understand cross-model attacks

2. **Real LLM Integration:**
   - Get Gemini API key
   - Add to `.env` file
   - Test with real model
   - Compare with mock results

3. **Custom Environment:**
   - Extend GridWorld with new features
   - Add obstacles, multi-room, etc.
   - Test on custom tasks

4. **Novel Defenses:**
   - Implement new sanitizer
   - Create ML-based detector
   - Measure defense effectiveness

**✓ After Phase 5, you can:**
- Design transferability experiments
- Integrate real LLMs
- Extend the framework
- Contribute new features

---

## 📚 Document Reference Guide

### Quick Reference

| Document | Purpose | Read When |
|----------|---------|-----------|
| `README.md` | Complete documentation | First, for overview |
| `QUICK_START_GUIDE.md` | Fast setup and common tasks | Need quick reference |
| `TUTORIAL.md` | Step-by-step exercises | Learning hands-on |
| `ARCHITECTURE_VISUAL.md` | System diagrams | Understanding structure |
| `PRESENTATION_GUIDE.md` | Demo script for talks | Preparing presentation |
| This file | Learning path | Starting your journey |

### Code Documentation

| File | What It Does | Key Functions |
|------|--------------|---------------|
| `envs/gridworld.py` | Navigation environment | `reset()`, `step()`, `render()` |
| `attacks/generator.py` | Attack creation | `generate_document()`, `generate_episode_attack_config()` |
| `llm/wrapper.py` | LLM interface | `generate()`, `create_llm_client()` |
| `runner/controller.py` | Episode execution | `run_episode()`, `build_prompt()`, `parse_action()` |
| `defenses/detector.py` | Malicious detection | `detect_malicious()`, `is_malicious()` |
| `experiments/runner.py` | Batch experiments | `run_experiment_grid()` |
| `analysis/visualize.py` | Plotting | `plot_asr_vs_strength()`, `create_summary_dashboard()` |

---

## 🎓 Learning Objectives Checklist

### Beginner Level
- [ ] Can explain what prompt injection is
- [ ] Understand GridWorld environment
- [ ] Can run demo scripts
- [ ] Know what ASR and Utility mean
- [ ] Can identify compromised behavior

### Intermediate Level
- [ ] Can configure experiments
- [ ] Generate custom attacks
- [ ] Test defense mechanisms
- [ ] Create visualizations
- [ ] Interpret results correctly

### Advanced Level
- [ ] Design transferability studies
- [ ] Integrate real LLMs
- [ ] Extend framework with new features
- [ ] Implement novel defenses
- [ ] Contribute to codebase

---

## 🎯 Common Use Cases

### 1. Evaluate Your LLM Agent
```python
# Your agent
from your_agent import YourLLMAgent

# Wrap it
from llm.wrapper import LLMClient

class YourClientWrapper(LLMClient):
    def __init__(self, your_agent):
        self.agent = your_agent
    
    def generate(self, prompt, **kwargs):
        return self.agent.process(prompt)

# Test it
from runner.controller import run_episode
from envs.gridworld import GridWorld

env = GridWorld(grid_size=(10, 10))
client = YourClientWrapper(your_agent)
summary = run_episode(env, client)

print(f"Your agent's vulnerability: {summary['compromise_rate']:.1%}")
```

### 2. Benchmark Defense Mechanism
```python
# Your defense
from your_defense import YourDefense

# Test it
from defenses.detector import detect_malicious
from attacks.generator import generate_attack_suite

attacks = generate_attack_suite()
defense = YourDefense()

for name, docs in attacks.items():
    detected = [defense.is_malicious(d['text']) for d in docs]
    print(f"{name}: {sum(detected)}/{len(docs)} detected")
```

### 3. Study Attack Transferability
```python
from experiments.transferability import compute_transfer_matrix

models = {
    'model_a': your_llm_a,
    'model_b': your_llm_b,
    'model_c': your_llm_c
}

matrix = compute_transfer_matrix(models, num_episodes=20)
# Shows which attacks transfer between models
```

---

## 🚀 Project Setup Checklist

### First Time Setup
- [ ] Clone/download repository
- [ ] Install Python 3.9+ 
- [ ] Create virtual environment
- [ ] Install dependencies: `pip install -e .`
- [ ] Run tests: `pytest tests/ -v`
- [ ] Run demo: `python demo_complete.py`

### Optional Setup
- [ ] Get Gemini API key (for real LLM)
- [ ] Create `.env` file with API key
- [ ] Install Jupyter: `pip install jupyter`
- [ ] Install Docker (for containerized runs)

### Verification
- [ ] All 191 tests pass
- [ ] Demo scripts run without errors
- [ ] Can generate visualizations
- [ ] Understand output logs

---

## 💡 Pro Tips

### For Researchers
1. **Start with Mock LLM** - Deterministic, fast, reproducible
2. **Use seeds** - Ensures reproducibility across runs
3. **Run small experiments first** - Test with n_episodes=5
4. **Save everything** - Logs, configs, results
5. **Visualize early** - Plots reveal patterns

### For Developers
1. **Read tests** - Best examples of usage (`tests/`)
2. **Use type hints** - Code is well-documented
3. **Follow patterns** - Consistent structure across modules
4. **Extend carefully** - Maintain backward compatibility
5. **Add tests** - For any new features

### For Security Analysts
1. **Focus on ASR** - Primary vulnerability metric
2. **Test edge cases** - Extreme strengths, rare patterns
3. **Compare defenses** - Run with/without each layer
4. **Study logs** - `data/run_logs/*.jsonl` has details
5. **Document findings** - Use templates provided

---

## 🎬 Next Steps After Learning

### Immediate Actions
1. Run your first custom experiment
2. Create your own attack variant
3. Test a defense idea
4. Generate visualizations
5. Share results with team

### Short Term (This Week)
1. Prepare a demo presentation
2. Integrate with your LLM
3. Run comprehensive benchmarks
4. Document findings
5. Identify vulnerabilities

### Long Term (This Month)
1. Extend framework for your needs
2. Implement novel defenses
3. Study transferability patterns
4. Write research paper/report
5. Contribute back to project

---

## 📞 Getting Help

### Common Issues

**Import Errors:**
```bash
# Solution
set PYTHONPATH=%CD%
pip install -e .
```

**Tests Failing:**
```bash
# Solution
pip install pytest pytest-cov
pytest tests/ -v
```

**Visualization Errors:**
```bash
# Solution
pip install matplotlib seaborn
```

### Where to Look

1. **Error messages** - Usually tell you exactly what's wrong
2. **Test files** - Examples of correct usage
3. **Demo scripts** - Working implementations
4. **Documentation** - README, guides, docstrings

### What to Try

1. **Check Python version** - Need 3.9+
2. **Verify dependencies** - `pip list`
3. **Clear cache** - Delete `__pycache__/`
4. **Fresh install** - `pip install -e . --force-reinstall`
5. **Use virtual environment** - Isolate dependencies

---

## 🏆 Success Criteria

**You've mastered the framework when you can:**

✅ Explain prompt injection to someone new  
✅ Run experiments independently  
✅ Generate and test custom attacks  
✅ Implement a simple defense  
✅ Create publication-quality plots  
✅ Interpret ASR/utility tradeoffs  
✅ Debug issues on your own  
✅ Extend the framework  
✅ Present findings clearly  
✅ Contribute improvements  

---

## 🎯 Final Thoughts

This framework is a **research tool** for:
- 🔬 **Security Research** - Quantify vulnerabilities
- 🛡️ **Defense Development** - Test mitigation strategies
- 📊 **Benchmarking** - Compare models and approaches
- 📚 **Education** - Teach AI security concepts

**The goal:** Make LLM agents more secure through systematic testing.

**Your role:** Use these tools to advance the field.

**Remember:** Security is an ongoing process, not a destination.

---

## 📖 Recommended Reading Order

### Day 1 (2 hours)
1. README.md (30 min)
2. Run demo_complete.py (20 min)
3. QUICK_START_GUIDE.md (30 min)
4. ARCHITECTURE_VISUAL.md (40 min)

### Day 2 (2 hours)
1. TUTORIAL.md Exercises 1-4 (60 min)
2. Explore code files (30 min)
3. TUTORIAL.md Exercises 5-7 (30 min)

### Day 3 (2 hours)
1. Run custom experiments (60 min)
2. Read PRESENTATION_GUIDE.md (30 min)
3. Prepare your own demo (30 min)

### Ongoing
- Refer to QUICK_START_GUIDE.md as needed
- Dive into specific modules when extending
- Use PRESENTATION_GUIDE.md for talks

---

## ✨ You're Ready!

You now have everything you need to:
- ✅ Understand the framework
- ✅ Run experiments
- ✅ Test security
- ✅ Present findings
- ✅ Extend capabilities

**Start with:** `python demo_complete.py`

**Questions?** Check the documentation or explore the code!

**Good luck with your research!** 🚀

---

*Last Updated: November 4, 2025*  
*Framework Version: 0.1.0*  
*191 Tests Passing* ✓
