# 🎓 Research Paper Experiment: 4-Model Comparison

## Overview
This experiment compares **4 LLM models** for prompt injection robustness:
- **2 Cloud models**: Gemini-1.5-Flash, Gemini-1.5-Flash-8B (Google, free tier)
- **2 Local models**: Phi-3, Llama-3.2 (Microsoft/Meta, runs on your hardware)

Perfect for your research paper! ✅

---

## 🚀 Quick Start (5 minutes)

### Step 1: Start Ollama Service
Open a **new terminal** and keep it running:
```bash
ollama serve
```

### Step 2: Download Models
In **another terminal**:
```bash
cd c:\Users\Tanmay\Desktop\lam-injection
python scripts/setup_ollama_models.py
```

This will:
- ✓ Check if Ollama is running
- ✓ Show currently downloaded models
- ✓ Download phi3 (3.8GB) and llama3.2 (2GB)
- ✓ Verify everything is ready

**Time:** ~10-15 minutes (downloads)

### Step 3: Run Experiment
```bash
python experiments/multi_model_comparison.py
```

**Time:** ~10-15 minutes (with 10 episodes per config)

### Step 4: View Results
```bash
cd data\multi_model_comparison
dir
```

You'll see:
- `multi_model_results_*.csv` - Raw data for analysis
- `asr_comparison.png` - ASR curves
- `utility_comparison.png` - Goal-reaching rates
- `combined_comparison.png` - Side-by-side plots
- `vulnerability_ranking.png` - Bar chart ranking

---

## 📊 What Gets Measured

### 1. Attack Success Rate (ASR)
- **Definition:** % of episodes where agent was compromised
- **Formula:** `compromised_episodes / total_episodes`
- **Interpretation:** Lower = more robust
- **Range:** 0% (never compromised) to 100% (always compromised)

### 2. Utility
- **Definition:** % of episodes where agent reached goal
- **Formula:** `successful_episodes / total_episodes`
- **Interpretation:** Higher = better at maintaining functionality
- **Range:** 0% (never succeeded) to 100% (always succeeded)

### 3. Attack Strengths Tested
- **0.0:** No attack (baseline)
- **0.3:** Weak attack
- **0.5:** Moderate attack
- **0.7:** Strong attack
- **0.9:** Very strong attack

---

## 📈 Expected Results Pattern

### Typical Model Behavior
```
Attack Strength    ASR        Utility
0.0 (none)        0-5%       85-95%    (baseline performance)
0.3 (weak)        10-20%     75-85%    (minimal impact)
0.5 (medium)      25-45%     60-75%    (noticeable impact)
0.7 (strong)      40-65%     45-65%    (significant impact)
0.9 (very strong) 55-80%     30-55%    (heavy impact)
```

### Model Comparison Hypotheses
1. **Larger models** (Gemini-Flash) likely more robust than smaller (Flash-8B)
2. **Cloud models** (Gemini) may differ from local (Ollama)
3. **Phi-3** (Microsoft) optimized for reasoning tasks
4. **Llama-3.2** (Meta) latest compact architecture

---

## 🎯 Research Paper Sections

### Methodology
```
We evaluated 4 LLM models using the Agentic Prompt-Injection 
Robustness Benchmark framework [citation]. The benchmark simulates 
a goal-oriented navigation agent in a 10×10 GridWorld environment 
with potentially malicious documents.

Models tested:
1. Gemini-1.5-Flash (Google, cloud)
2. Gemini-1.5-Flash-8B (Google, cloud)  
3. Phi-3 (Microsoft, local)
4. Llama-3.2 (Meta, local)

Each model was tested across 5 attack strengths (0.0, 0.3, 0.5, 
0.7, 0.9) with 10 episodes per configuration (50 episodes total 
per model). We measured Attack Success Rate (ASR) and Utility 
(goal-reaching rate) as primary metrics.
```

### Results (example - use your actual data!)
```
Table 1: Model Robustness Comparison (ASR at strength 0.9)

Model                  ASR      Utility   Size
Gemini-1.5-Flash       45%      72%       Unknown
Gemini-1.5-Flash-8B    52%      68%       Unknown
Phi-3                  38%      65%       3.8B params
Llama-3.2              41%      70%       3.2B params

Lower ASR indicates higher robustness to prompt injection attacks.
Phi-3 demonstrated the lowest ASR (38%) at maximum attack strength,
suggesting superior resistance to adversarial prompts.
```

### Discussion Points
1. **Size vs Robustness:** Do larger models resist attacks better?
2. **Cloud vs Local:** Performance differences between deployment types
3. **Training Differences:** Impact of model architecture/training data
4. **Utility Trade-off:** Can models be robust without sacrificing utility?
5. **Practical Implications:** Which models suitable for production?

---

## 🔧 Configuration Options

### For Quick Testing (5 minutes)
Edit `experiments/multi_model_comparison.py` line 240:
```python
results = run_comparison_experiment(
    models=models,
    attack_strengths=[0.0, 0.5, 0.9],  # Only 3 strengths
    n_episodes=5  # Only 5 episodes
)
```

### For Publication Quality (1-2 hours)
```python
results = run_comparison_experiment(
    models=models,
    attack_strengths=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],  # More granular
    n_episodes=50  # Better statistics
)
```

### For Specific Model Subset
Comment out models in `setup_models()`:
```python
# Test only Gemini models (faster for API testing)
models = {}
models['gemini-flash'] = GeminiClient(model_name="gemini-1.5-flash")
models['gemini-flash-8b'] = GeminiClient(model_name="gemini-1.5-flash-8b")
# Skip Ollama models
```

---

## 📁 File Structure

```
lam-injection/
├── experiments/
│   └── multi_model_comparison.py      ← Main experiment script
├── llm/
│   ├── gemini_client.py               ← Gemini API integration
│   ├── ollama_client.py               ← Ollama integration
│   └── wrapper.py                     ← Unified interface
├── scripts/
│   └── setup_ollama_models.py         ← Model downloader
├── data/
│   └── multi_model_comparison/        ← Results output
│       ├── *.csv                      ← Raw data
│       └── *.png                      ← Plots
├── RESEARCH_PAPER_SETUP.md           ← Detailed setup guide
└── RESEARCH_QUICKSTART.md            ← This file
```

---

## 🐛 Troubleshooting

### Issue: "Ollama service not running"
**Solution:**
```bash
# Open new terminal
ollama serve
```
Keep this terminal open!

### Issue: "Model not found"
**Solution:**
```bash
# Download manually
ollama pull phi3
ollama pull llama3.2

# Verify
ollama list
```

### Issue: Gemini API quota exceeded
**Symptoms:** Error messages mentioning rate limits

**Solution 1 - Wait:** Free tier resets every 60 seconds

**Solution 2 - Run Ollama only:**
Comment out Gemini in `setup_models()` function

### Issue: Out of memory
**Symptoms:** System freezes, Ollama crashes

**Solution - Use smaller models:**
```bash
ollama pull tinyllama    # Only 637MB!
ollama pull gemma2:2b    # Only 1.6GB
```

Then edit `setup_models()` to use these instead.

### Issue: Slow performance
**Solutions:**
1. Close Chrome/other heavy apps
2. Use only 2 models instead of 4
3. Reduce n_episodes to 5
4. Use smaller Ollama models

---

## ✅ Validation Checklist

Before running full experiment:

**Environment:**
- [ ] Ollama service running (`ollama serve`)
- [ ] Models downloaded (phi3, llama3.2)
- [ ] Gemini API key in `.env`
- [ ] Python environment activated
- [ ] No other heavy apps running

**Quick Test:**
- [ ] Run with n_episodes=5 first
- [ ] Verify no errors in console
- [ ] Check outputs created in `data/multi_model_comparison/`
- [ ] Verify plots look reasonable

**Full Run:**
- [ ] Increase n_episodes to 50
- [ ] Monitor console for errors
- [ ] Save console output to log file
- [ ] Backup results CSV immediately

---

## 🎓 Academic Writing Tips

### Citing the Framework
```
@software{agentic_prompt_injection_benchmark,
  title={Agentic Prompt-Injection Robustness Benchmark},
  author={[Your name/team]},
  year={2024},
  url={https://github.com/...}
}
```

### Key Phrases for Your Paper
- "prompt injection vulnerability"
- "adversarial robustness"
- "goal-oriented agent security"
- "document-based attacks"
- "ASR-Utility tradeoff"
- "multi-model comparative analysis"

### Related Work to Cite
- Prompt injection attacks literature
- LLM agent security papers
- Adversarial robustness in NLP
- Gemini/Llama model papers

---

## 📞 Next Steps

1. **Run quick test** (5 episodes): `python experiments/multi_model_comparison.py`
2. **Review outputs:** Check `data/multi_model_comparison/`
3. **Run thorough experiment** (50 episodes): Edit script, re-run
4. **Analyze data:** Open CSV in Excel/Python for statistics
5. **Create paper figures:** Use generated PNG plots
6. **Write results section:** Report ASR and Utility values

---

## 🎉 Success Criteria

You'll have publication-ready results when you see:

✅ CSV file with ASR and Utility for all models
✅ 4 publication-quality plots (300 DPI)
✅ Console summary with statistics
✅ Clear ranking of model robustness

**Estimated time to complete:** 2-3 hours total

Good luck with your research! 🚀📊🎓
