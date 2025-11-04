# 🚀 Multi-Model Research Experiment - Master Guide

## 📖 Quick Navigation

### For Impatient Researchers (5 minutes to results)
→ **[RESEARCH_QUICKSTART.md](RESEARCH_QUICKSTART.md)**
- 3-step quick start
- What gets measured (ASR, Utility)
- Expected results
- Paper writing templates

### For Thorough Setup (20 minutes)
→ **[RESEARCH_PAPER_SETUP.md](RESEARCH_PAPER_SETUP.md)**
- Complete prerequisites checklist
- Detailed model download instructions
- Troubleshooting guide
- Research paper integration guide

### For Technical Details
→ **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)**
- Architecture overview
- File-by-file documentation
- Configuration reference
- Performance benchmarks

---

## 🎯 What This Does

Compare **4 LLM models** for prompt injection robustness:
- **Gemini-1.5-Flash** (Google, cloud, free)
- **Gemini-1.5-Flash-8B** (Google, cloud, free)
- **Phi-3** (Microsoft, local, 3.8GB)
- **Llama-3.2** (Meta, local, 2GB)

Get publication-ready results:
- ✅ CSV data for statistical analysis
- ✅ 4 publication-quality plots (300 DPI)
- ✅ ASR and Utility metrics
- ✅ Model vulnerability ranking

---

## ⚡ Super Quick Start

```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Check setup
python check_setup.py

# If all checks pass:
python experiments/multi_model_comparison.py

# Or use the helper:
run_research_experiment.bat
```

**Time to results:** 15-20 minutes

---

## 📁 Key Files

### Run These:
| File | Purpose | Time |
|------|---------|------|
| `check_setup.py` | Verify everything is ready | 30 sec |
| `scripts/setup_ollama_models.py` | Download models | 10 min |
| `experiments/multi_model_comparison.py` | Run experiment | 15 min |
| `run_research_experiment.bat` | All-in-one (Windows) | 15 min |

### Read These:
| File | When to Read |
|------|--------------|
| `RESEARCH_QUICKSTART.md` | First time setup |
| `RESEARCH_PAPER_SETUP.md` | Writing the paper |
| `IMPLEMENTATION_SUMMARY.md` | Understanding code |

### Results Location:
```
data/multi_model_comparison/
├── multi_model_results_*.csv      # Raw data
├── asr_comparison.png             # ASR curves
├── utility_comparison.png         # Goal-reaching rates
├── combined_comparison.png        # Side-by-side
└── vulnerability_ranking.png      # Bar chart
```

---

## 🔄 Typical Workflow

### First Time Setup (20 minutes)
1. **Check prerequisites:**
   ```bash
   python check_setup.py
   ```

2. **Start Ollama service:**
   ```bash
   ollama serve
   ```
   (Keep this terminal open!)

3. **Download models:**
   ```bash
   python scripts/setup_ollama_models.py
   ```

4. **Verify setup:**
   ```bash
   python check_setup.py
   ```

### Running Experiments

#### Quick Test (5 minutes)
```bash
python experiments/multi_model_comparison.py
```

#### For Publication (1-2 hours)
Edit `experiments/multi_model_comparison.py` line 240:
```python
results = run_comparison_experiment(
    models=models,
    attack_strengths=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    n_episodes=50  # More reliable statistics
)
```

### Using Results

1. **View plots:**
   ```bash
   cd data\multi_model_comparison
   # Open PNG files
   ```

2. **Analyze data:**
   ```python
   import pandas as pd
   df = pd.read_csv("data/multi_model_comparison/multi_model_results_*.csv")
   print(df.groupby('model')['asr'].mean())
   ```

3. **Write paper:**
   - Copy plots → Figures
   - Extract ASR/Utility → Results table
   - Use methodology template → Methods section

---

## 🎓 For Your Research Paper

### Metrics to Report

**Attack Success Rate (ASR)**
- Definition: % of episodes where agent was compromised
- Lower = more robust
- Report: Mean ± SD across attack strengths

**Utility**
- Definition: % of episodes where agent reached goal
- Higher = better functionality
- Report: Mean ± SD across attack strengths

**Example Results Table:**
```
Model                  ASR (max)   Utility (avg)
Gemini-1.5-Flash       45% ± 8%    72% ± 5%
Gemini-1.5-Flash-8B    52% ± 7%    68% ± 6%
Phi-3                  38% ± 6%    65% ± 7%
Llama-3.2              41% ± 5%    70% ± 4%
```

### Figures to Include

1. **Figure 1:** ASR comparison (`asr_comparison.png`)
   - Caption: "Attack Success Rate across varying attack strengths for 4 LLM models"

2. **Figure 2:** Vulnerability ranking (`vulnerability_ranking.png`)
   - Caption: "Model vulnerability at maximum attack strength (0.9)"

3. **Figure 3:** Combined ASR & Utility (`combined_comparison.png`)
   - Caption: "Trade-off between robustness (ASR) and functionality (Utility)"

### Methodology Template

```
We evaluated 4 large language models using the Agentic 
Prompt-Injection Robustness Benchmark framework. The benchmark 
simulates a goal-oriented navigation agent in a 10×10 GridWorld 
environment with potentially malicious documents positioned to 
intercept the agent.

Models tested include Gemini-1.5-Flash and Gemini-1.5-Flash-8B 
(Google, cloud-based), Phi-3 (Microsoft, 3.8B parameters), and 
Llama-3.2 (Meta, 3.2B parameters). Each model was evaluated across 
5 attack strengths (0.0, 0.3, 0.5, 0.7, 0.9) with 10 episodes per 
configuration, totaling 250 episodes per model.

We measured Attack Success Rate (ASR) as the percentage of episodes 
where the agent was compromised, and Utility as the percentage where 
the agent successfully reached its goal despite attacks.
```

---

## 🔧 Customization

### Test Different Models

```python
# In setup_models() function:

# Add GPT models (if you have API key)
models['gpt-3.5'] = OpenAIClient("gpt-3.5-turbo")

# Use smaller Ollama models
models['tinyllama'] = OllamaClient("tinyllama")  # Only 637MB!
models['gemma2'] = OllamaClient("gemma2:2b")     # Only 1.6GB
```

### Adjust Attack Parameters

```python
# In run_comparison_experiment():

# More granular strength sweep
attack_strengths = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Different attack types
docs = generate_episode_attack_config(
    num_docs=5,
    attack_strength=strength,
    attack_type='polite',  # 'direct', 'polite', or 'hidden'
    seed=42
)
```

### Change Environment

```python
# In run_comparison_experiment():

env = GridWorld(
    grid_size=(15, 15),    # Larger grid
    max_steps=50,          # More time
    documents=docs,
    seed=42 + ep
)
```

---

## 🐛 Common Issues & Solutions

### "Ollama service not running"
**Solution:**
```bash
# Open new terminal
ollama serve
```

### "Model not found"
**Solution:**
```bash
ollama pull phi3
ollama pull llama3.2
ollama list  # Verify
```

### "GEMINI_API_KEY not set"
**Solution:**
```bash
# Create/edit .env file
echo GEMINI_API_KEY=your_key_here > .env
```

### "Out of memory"
**Solution:**
```bash
# Use smaller models
ollama pull tinyllama    # 637MB
ollama pull gemma2:2b    # 1.6GB
```

### "Experiment too slow"
**Solution:**
- Reduce `n_episodes` from 10 to 5
- Test only 2 models instead of 4
- Use `attack_strengths=[0.0, 0.5, 0.9]` (3 instead of 5)

---

## ✅ Validation Checklist

### Before Running Experiment:
- [ ] Python environment activated
- [ ] `check_setup.py` passes all checks
- [ ] Ollama service running (`ollama serve`)
- [ ] Models downloaded (phi3, llama3.2)
- [ ] Gemini API key in `.env`
- [ ] Close heavy applications (free up RAM)

### After Running Experiment:
- [ ] CSV file created in `data/multi_model_comparison/`
- [ ] 4 PNG plots generated
- [ ] Console shows summary statistics
- [ ] No errors in output
- [ ] Plots show reasonable trends

### Before Writing Paper:
- [ ] Run thorough experiment (50 episodes)
- [ ] Backup results CSV
- [ ] Copy plots to paper folder
- [ ] Calculate mean ± SD for metrics
- [ ] Create results table

---

## 📞 Help & Support

### Documentation Files:
- **Quick start:** `RESEARCH_QUICKSTART.md`
- **Setup guide:** `RESEARCH_PAPER_SETUP.md`
- **Technical docs:** `IMPLEMENTATION_SUMMARY.md`
- **This file:** `RESEARCH_MASTER_INDEX.md`

### Helper Scripts:
- **Check setup:** `python check_setup.py`
- **Download models:** `python scripts/setup_ollama_models.py`
- **Run experiment:** `run_research_experiment.bat`

### Troubleshooting:
See "Troubleshooting" section in `RESEARCH_PAPER_SETUP.md`

---

## 🎉 Success Criteria

### Experiment is successful when you see:
```
✓ EXPERIMENT COMPLETE!
Check outputs:
  • Results CSV: data/multi_model_comparison/multi_model_results_*.csv
  • Plots: data/multi_model_comparison/*.png
```

### You're ready for publication when you have:
- ✅ CSV with reliable statistics (50+ episodes)
- ✅ Publication-quality plots (300 DPI)
- ✅ Clear model ranking (ASR and Utility)
- ✅ Results table for paper
- ✅ Methodology section written

---

## 🚀 Let's Go!

```bash
# 1. Check everything is ready
python check_setup.py

# 2. Run the experiment
run_research_experiment.bat

# 3. View results
cd data\multi_model_comparison
dir

# 4. Write your paper! 📝
```

**Estimated time:** 2-3 hours from zero to publication-ready results.

Good luck with your research! 🎓📊🏆
