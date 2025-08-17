# Stress-Testing `gpt-oss` 🚀🔥  

Welcome to my **science experiment / GPU torture chamber**.  
This repo documents the benchmarks, stress tests, and outright abuse I’ve put the `gpt-oss` model through — all in the name of *open-weight research*.  

---

## 🧪 Goal
- Explore the **limits of reasoning** and **context length** in `gpt-oss`  
- Compare performance against a baseline (Always1)  
- Later: run head-to-head with reasoning-oriented models  

---

## ⚙️ Setup
- **Main script:** `ai-bench.py`  
- **Task:** Menu navigation benchmark (pick the correct option based on a repeating sequence)  
- **Metrics tracked:**
  - Accuracy (%)  
  - Invalid outputs (#)  
  - Latency per query (ms)  
- **Baseline:** Always pick option 1 (our humble control group)  

---

## 📊 Results So Far
### Baseline sanity checks
- Small steps (difficulty ≤16): Accuracy holds steady around 60–80%  
- Medium steps (difficulty 32–512): Accuracy starts wobbling, still above random  
- Large steps (≥1024): Accuracy collapses into coin-flip territory  

### Extreme stress test  
