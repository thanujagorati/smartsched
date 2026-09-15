# SmartSched

An ML-based CPU scheduling advisor. Simulates FCFS, SJF, Round Robin, and
Priority scheduling algorithms, then uses machine learning to predict which
algorithm performs best for a given workload -- instead of brute-force
running all four every time.

## Status

🚧 Work in progress -- building in public, one small piece at a time.

- [x] FCFS simulator
- [x] SJF simulator
- [x] Priority scheduling simulator
- [x] Round Robin simulator
- [x] Metrics module (waiting time, turnaround time, response time)
- [x] Gantt chart data structure
- [x] Dataset generation (Poisson arrivals, bimodal burst times)
- [x] Random Forest classifier (best algorithm predictor)
- [x] Regression model (expected wait/turnaround time)
- [x] Heuristic baseline for comparison
- [x] SHAP explainability
- [x] Streamlit dashboard

## Why ML instead of just running all four simulations?

(To be written once the baseline comparison is built in Week 4 -- this will
become `docs/why_ml.md`.)

## Project structure

```
smartsched/
├── simulator/
│   ├── fcfs.py
│   ├── sjf.py            (coming soon)
│   ├── priority.py        (coming soon)
│   └── round_robin.py     (coming soon)
├── tests/
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
