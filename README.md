Stress-Testing gpt-oss 🚀🔥

Welcome to my science experiment / GPU torture chamber.
This repo documents the benchmarks, stress tests, and outright abuse I’ve put the gpt-oss model through — all in the name of open-weight research.

🧪 Goal

Explore the limits of reasoning and context length in gpt-oss

Compare performance against a baseline (Always1)

Later: run head-to-head with reasoning-oriented models

⚙️ Setup

Main script: ai-bench.py

Task: Menu navigation benchmark (pick the correct option based on a repeating sequence)

Metrics tracked:

Accuracy (%)

Invalid outputs (#)

Latency per query (ms)

Baseline: Always pick option 1 (our humble control group)

📊 Results So Far
Baseline sanity checks

Small steps (difficulty ≤16): Accuracy holds steady around 60–80%

Medium steps (difficulty 32–512): Accuracy starts wobbling, still above random

Large steps (≥1024): Accuracy collapses into coin-flip territory

Extreme stress test
Model=gpt-oss | Steps=10 | Difficulty=32000
Accuracy = 40.0%
Invalid  = 0
Avg time = 160,682 ms (≈ 2.5 minutes per question)


💡 Fun fact: My GPU survived, but the fans started whispering "why have you forsaken us?"

📈 What This Shows

Reasoning vs Memory tradeoff:
gpt-oss can juggle small puzzles well, but falters once memory requirements go interstellar.

Performance scaling:
At low difficulty, quick and clever. At high difficulty, slower than dial-up internet.

Baseline check:
Sometimes “Always pick option 1” is just as good… which is equal parts hilarious and tragic.


📝 Final Thoughts

This repo is part benchmarking project, part comedy of errors.
It shows that open-weight models like gpt-oss are impressive — but also hilariously human when pushed beyond reason:

Smart until you crank the difficulty

Sometimes indistinguishable from guessing

Always entertaining

🤝 Pull requests welcome. If your GPU is braver than mine, please contribute to the suffering.

Want me to also draft a RESULTS.md where you can paste all the raw runs (like the logs you showed me) so the README stays clean and polished, while the messy experiment data lives separately?
