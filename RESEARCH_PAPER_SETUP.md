# Research Paper Multi-Model Comparison Setup

## 🎯 Goal
Compare **4 LLM models** for your research paper:
- **2 Gemini models** (cloud, free tier): gemini-1.5-flash, gemini-1.5-flash-8b
- **2 Ollama models** (local): phi3, llama3.2

Generate real experimental data comparing robustness against prompt injection attacks.

---

## ✅ Prerequisites Checklist

### 1. Python Environment
```bash
# Already configured with all dependencies
pip install -r requirements.txt
```

### 2. Gemini API Key
```bash
# Check if set:
type .env
```
Should show: `GEMINI_API_KEY=your_key_here`

✅ Already configured!

### 3. Ollama Installation
```bash
# Verify Ollama is installed
ollama --version
```

✅ Already installed!

---

## 📥 Download Ollama Models

### Step 1: Start Ollama Service
```bash
# Open a new terminal and run:
ollama serve
```
Keep this terminal open while running experiments.

### Step 2: Download Recommended Models
Open another terminal and run:

```bash
# Download phi3 (3.8GB - Microsoft's efficient model)
ollama pull phi3

# Download llama3.2 (2GB - Meta's latest compact model)
ollama pull llama3.2
```

**Expected download times** (on typical internet):
- phi3: ~5-10 minutes
- llama3.2: ~3-5 minutes

**Why these models?**
- ✅ Optimized for 16GB RAM + Intel UHD graphics
- ✅ Small enough to run efficiently
- ✅ Strong performance for agent tasks
- ✅ From reputable sources (Microsoft, Meta)

### Alternative Models (if phi3/llama3.2 too slow)
```bash
# Tiny models for faster testing:
ollama pull tinyllama      # 637MB - fastest
ollama pull gemma2:2b      # 1.6GB - small but capable
```

### Step 3: Verify Models
```bash
# List downloaded models:
ollama list
```
Should show:
```
NAME            ID         SIZE    MODIFIED
phi3:latest     ... 3.8GB  ...
llama3.2:latest ... 2.0GB  ...
```

---

## 🚀 Running the Comparison Experiment

### Quick Test (5 episodes, ~5 minutes)
```bash
python experiments/multi_model_comparison.py
```

This will:
1. ✓ Test all 4 models
2. ✓ Run 5 attack strengths: [0.0, 0.3, 0.5, 0.7, 0.9]
3. ✓ 10 episodes per configuration
4. ✓ Generate 4 publication-quality plots
5. ✓ Save results to CSV

**Expected runtime:**
- With 4 models: ~10-15 minutes
- With 2 models only: ~5-7 minutes

### Thorough Experiment (for publication)
Edit `experiments/multi_model_comparison.py` line 240:
```python
# Change n_episodes from 10 to 20 or 50
results = run_comparison_experiment(
    models=models,
    attack_strengths=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],  # More granular
    n_episodes=50  # More reliable statistics
)
```

**Runtime:** ~1-2 hours (but publication-quality!)

---

## 📊 Output Files

### Results Location
All outputs saved to: `data/multi_model_comparison/`

### Generated Files

1. **CSV Results**
   - `multi_model_results_YYYYMMDD_HHMMSS.csv`
   - Contains: model, attack_strength, asr, utility, avg_steps
   - Use this for statistical analysis

2. **Plots** (Publication-ready, 300 DPI)
   - `asr_comparison.png` - ASR curves for all models
   - `utility_comparison.png` - Goal-reaching rates
   - `combined_comparison.png` - Side-by-side ASR & Utility
   - `vulnerability_ranking.png` - Bar chart ranking models

3. **Console Summary**
   - Average ASR per model
   - Average Utility per model
   - Performance at max attack strength
   - Token usage statistics

---

## 📈 Using Results in Your Paper

### Key Metrics to Report

1. **Attack Success Rate (ASR)**
   - Percentage of episodes where agent was compromised
   - Higher = more vulnerable

2. **Utility**
   - Percentage of episodes where agent reached goal
   - Higher = more robust (maintains functionality)

3. **ASR-Utility Tradeoff**
   - Compare models: low ASR + high utility = best
   - Plot both on same graph

### Example Paper Findings
```
We evaluated 4 LLM models on prompt injection robustness:
- Gemini-1.5-flash: ASR 45%, Utility 72%
- Gemini-1.5-flash-8b: ASR 52%, Utility 68%
- Phi-3: ASR 38%, Utility 65%
- Llama-3.2: ASR 41%, Utility 70%

Phi-3 showed highest robustness (lowest ASR) while maintaining
acceptable utility. Gemini-1.5-flash achieved best utility but
higher vulnerability.
```

### Citing the Framework
```
We used the Agentic Prompt-Injection Robustness Benchmark,
a standardized framework for evaluating LLM agent security
in document-based attack scenarios.
```

---

## 🔧 Troubleshooting

### Issue: Ollama models not found
**Solution:**
```bash
# Restart Ollama service
ollama serve

# Re-download models
ollama pull phi3
ollama pull llama3.2
```

### Issue: Gemini API quota exceeded
**Solution:**
1. Wait 60 seconds (rate limit reset)
2. Or use only Ollama models for testing
3. Free tier: 15 requests/minute

### Issue: Out of memory
**Symptoms:** System freezes, Ollama crashes

**Solution 1 - Use smaller models:**
```bash
ollama pull tinyllama    # Only 637MB
ollama pull gemma2:2b    # Only 1.6GB
```

**Solution 2 - Run models sequentially:**
Edit `multi_model_comparison.py`, comment out 2 models, run twice.

### Issue: Slow performance
**Solution:**
- Close other applications
- Use smaller models (tinyllama, gemma2:2b)
- Reduce n_episodes from 10 to 5
- Reduce attack_strengths to [0.0, 0.5, 0.9]

---

## 🎓 Research Paper Checklist

### Before Running Experiments
- [ ] Ollama service running (`ollama serve`)
- [ ] Models downloaded (phi3, llama3.2)
- [ ] Gemini API key verified
- [ ] Close unnecessary applications (free up RAM)

### Running Experiments
- [ ] Run quick test first (5 episodes)
- [ ] If successful, run thorough experiment (50 episodes)
- [ ] Monitor for errors in console
- [ ] Save console output to log file

### After Experiments
- [ ] Verify CSV created in `data/multi_model_comparison/`
- [ ] Check all 4 plots generated
- [ ] Review summary statistics
- [ ] Copy results to paper draft

### Writing Paper
- [ ] Include methodology section (describe benchmark)
- [ ] Report ASR and Utility for all models
- [ ] Include ASR comparison plot
- [ ] Include vulnerability ranking bar chart
- [ ] Discuss implications (which model is most robust?)
- [ ] Mention limitations (GridWorld simplification)

---

## 💡 Quick Start Commands

```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Download models (first time only)
ollama pull phi3
ollama pull llama3.2

# Terminal 3: Run experiment
cd c:\Users\Tanmay\Desktop\lam-injection
python experiments/multi_model_comparison.py

# View results
cd data\multi_model_comparison
dir
```

---

## 📧 Next Steps

1. **Download models** (~10 min)
2. **Run quick test** (~5 min)
3. **Review outputs** (~5 min)
4. **Run thorough experiment** (~1 hour)
5. **Write paper section** using plots and data

**Estimated time to publication-ready results:** 2-3 hours

---

## 🎉 Success Criteria

You'll know it worked when you see:
```
✓ EXPERIMENT COMPLETE!
Check outputs:
  • Results CSV: data/multi_model_comparison/multi_model_results_*.csv
  • Plots: data/multi_model_comparison/*.png
```

Good luck with your research paper! 🚀📊
