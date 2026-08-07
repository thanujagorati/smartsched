"""
SmartSched Streamlit dashboard.

Ties together everything built so far: workload generation, all four
simulators, metrics, the ML classifier (with SHAP explanation), and
Gantt chart visualization -- into one interactive app.

Run with: streamlit run app.py  (from inside the dashboard/ folder)

NOTE: the trained model files (random_forest_model.joblib) are
excluded from git (too large for GitHub -- see .gitignore). If you're
running this on a fresh clone, first run:
    python ml/train_classifier.py
from inside the ml/ folder, to regenerate the model locally.
"""

import sys
import os
import statistics

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Make the simulator/ and ml/ modules importable from this dashboard/ folder
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "simulator"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ml"))

from workload_generator import generate_workload
from fcfs import fcfs
from sjf import sjf
from priority import priority_scheduling
from round_robin import round_robin
from metrics import compute_metrics
from gantt import to_gantt_segments
from dataset_builder import extract_features, compute_score
from train_classifier import FEATURE_COLUMNS
from explain import explain_features


MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "random_forest_model.joblib")
REGRESSION_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "regression_model.joblib")

# A few hand-crafted extreme workloads -- these are deliberately
# designed to highlight WHY each algorithm exists, rather than random
# generation (which mostly produces "typical" cases). Good for demos.
PRESET_WORKLOADS = {
    "Convoy effect (1 long process blocks many short)": [
        {"pid": "P1", "arrival_time": 0, "burst_time": 40, "estimated_burst_time": 38, "priority": 3},
        {"pid": "P2", "arrival_time": 1, "burst_time": 2, "estimated_burst_time": 2, "priority": 3},
        {"pid": "P3", "arrival_time": 2, "burst_time": 3, "estimated_burst_time": 3, "priority": 3},
        {"pid": "P4", "arrival_time": 3, "burst_time": 2, "estimated_burst_time": 2, "priority": 3},
        {"pid": "P5", "arrival_time": 4, "burst_time": 1, "estimated_burst_time": 1, "priority": 3},
    ],
    "All identical burst times (algorithms should tie)": [
        {"pid": f"P{i+1}", "arrival_time": i, "burst_time": 5, "estimated_burst_time": 5, "priority": 3}
        for i in range(6)
    ],
    "Priority favors short jobs (Priority ~ SJF)": [
        {"pid": "P1", "arrival_time": 0, "burst_time": 20, "estimated_burst_time": 20, "priority": 5},
        {"pid": "P2", "arrival_time": 0, "burst_time": 3, "estimated_burst_time": 3, "priority": 1},
        {"pid": "P3", "arrival_time": 0, "burst_time": 4, "estimated_burst_time": 4, "priority": 2},
        {"pid": "P4", "arrival_time": 0, "burst_time": 15, "estimated_burst_time": 15, "priority": 4},
    ],
    "All arrive at once, many short bursts (RR-friendly)": [
        {"pid": f"P{i+1}", "arrival_time": 0, "burst_time": 3 + (i % 3), "estimated_burst_time": 3 + (i % 3), "priority": 3}
        for i in range(8)
    ],
}


@st.cache_resource
def load_model():
    """Cached so the model only loads once per session, not on every rerun."""
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_regression_model():
    """Cached loader for the regression model + its feature column list."""
    if not os.path.exists(REGRESSION_MODEL_PATH):
        return None
    return joblib.load(REGRESSION_MODEL_PATH)


def run_all_algorithms(workload):
    """Run all four algorithms on the same workload, return results + segments."""
    avg_burst = statistics.mean(p["burst_time"] for p in workload)
    rr_quantum = max(1, round(avg_burst / 2))

    fcfs_result = fcfs(workload)
    sjf_result = sjf(workload, estimate_key="estimated_burst_time")
    priority_result = priority_scheduling(workload)
    rr_result, rr_segments = round_robin(workload, quantum=rr_quantum, context_switch_overhead=1)

    results = {
        "FCFS": fcfs_result,
        "SJF": sjf_result,
        "Priority": priority_result,
        "RoundRobin": rr_result,
    }
    segments = {
        "FCFS": to_gantt_segments(fcfs_result),
        "SJF": to_gantt_segments(sjf_result),
        "Priority": to_gantt_segments(priority_result),
        "RoundRobin": rr_segments,
    }
    return results, segments


def plot_gantt(algo_name, segments):
    """Render one algorithm's Gantt chart as a horizontal timeline."""
    fig, ax = plt.subplots(figsize=(9, 1.3))
    colors = plt.cm.tab10.colors
    pid_list = sorted({pid for pid, _, _ in segments})
    pid_color = {pid: colors[i % len(colors)] for i, pid in enumerate(pid_list)}

    for pid, start, end in segments:
        ax.barh(0, end - start, left=start, color=pid_color[pid], edgecolor="white")
        ax.text((start + end) / 2, 0, pid, ha="center", va="center", fontsize=8, color="white")

    ax.set_yticks([])
    ax.set_xlabel("Time")
    ax.set_title(algo_name, fontsize=11, loc="left")
    fig.tight_layout()
    return fig


st.set_page_config(page_title="SmartSched", layout="wide")
st.title("SmartSched: ML-Based CPU Scheduling Advisor")
st.caption("Compares FCFS, SJF, Priority, and Round Robin -- and predicts which one wins, before running any of them.")

# --- Sidebar: workload configuration ---
st.sidebar.header("Workload")
mode = st.sidebar.radio("Source", ["Generate random workload", "Preset extreme workload"])

if mode == "Generate random workload":
    num_processes = st.sidebar.slider("Number of processes", 3, 40, 10)
    arrival_rate = st.sidebar.slider("Arrival rate (higher = more bunched up)", 0.1, 1.0, 0.5)
    seed = st.sidebar.number_input("Random seed (for reproducibility)", value=42, step=1)
    workload = generate_workload(num_processes, arrival_rate=arrival_rate, seed=int(seed))
else:
    preset_name = st.sidebar.selectbox("Choose a preset", list(PRESET_WORKLOADS.keys()))
    workload = PRESET_WORKLOADS[preset_name]

# --- Show the workload itself ---
st.subheader("Workload")
st.dataframe(pd.DataFrame(workload), width='stretch', hide_index=True)

# --- Run simulations ---
results, segments = run_all_algorithms(workload)
metrics_by_algo = {name: compute_metrics(res) for name, res in results.items()}
metrics_df = pd.DataFrame(metrics_by_algo).T
actual_best = min(metrics_by_algo, key=lambda algo: compute_score(metrics_by_algo[algo]))

st.subheader("Metrics comparison (ground truth, via simulation)")
st.dataframe(metrics_df, width='stretch')
st.success(f"Best algorithm (by simulation): **{actual_best}**")

# --- ML prediction ---
st.subheader("ML Classifier Prediction")
model = load_model()

if model is None:
    st.error(
        "Model file not found. Run `python train_classifier.py` from the "
        "`ml/` folder first -- trained models are excluded from git "
        "(too large for GitHub) and must be regenerated locally."
    )
else:
    features = extract_features(workload)
    features_df = pd.DataFrame([features])[FEATURE_COLUMNS]
    predicted = model.predict(features_df)[0]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("ML predicted best algorithm", predicted)
    with col2:
        st.metric("Actual best (via simulation)", actual_best)

    if predicted == actual_best:
        st.success("Correct prediction!")
    else:
        st.warning(
            f"Mismatch -- the model predicted {predicted}, but {actual_best} "
            f"actually performed best on this specific workload."
        )

    # --- SHAP explanation: WHY did the model predict this? ---
    st.markdown("**Why this prediction? (SHAP feature impact)**")
    explanation = explain_features(model, features_df)
    shap_df = pd.DataFrame(
        explanation["feature_impact"], columns=["Feature", "Impact"]
    )
    shap_df["Direction"] = shap_df["Impact"].apply(
        lambda v: f"toward {predicted}" if v > 0 else f"away from {predicted}"
    )
    st.dataframe(shap_df, width='stretch', hide_index=True)
    st.caption(
        "Positive impact = pushed the model toward this prediction. "
        "Negative = pushed away from it. Ranked by strength of influence."
    )

    # --- Regression model: predicted wait time PER algorithm ---
    reg_bundle = load_regression_model()
    if reg_bundle is not None:
        st.markdown("**Predicted average waiting time per algorithm (regression model)**")
        reg_model = reg_bundle["model"]
        reg_feature_cols = reg_bundle["feature_columns"]

        algo_names = ["FCFS", "SJF", "Priority", "RoundRobin"]
        rows = []
        for algo in algo_names:
            row = {col: features[col] for col in FEATURE_COLUMNS}
            for a in algo_names:
                row[f"algo_{a}"] = (a == algo)
            rows.append(row)

        reg_input = pd.DataFrame(rows)[reg_feature_cols]
        predicted_times = reg_model.predict(reg_input)

        reg_result_df = pd.DataFrame({
            "Algorithm": algo_names,
            "Predicted avg waiting time": predicted_times.round(2),
            "Actual avg waiting time (simulated)": [
                metrics_by_algo[a]["avg_waiting_time"] for a in algo_names
            ],
        })
        st.dataframe(reg_result_df, width='stretch', hide_index=True)
        st.caption(
            "Compares the regressor's PREDICTED time (no simulation run) "
            "against the ACTUAL time from running the real simulator -- "
            "shows how close the regression model gets without brute-forcing it."
        )

# --- Gantt charts ---
st.subheader("Gantt Charts")
tabs = st.tabs(list(results.keys()))
for tab, algo_name in zip(tabs, results.keys()):
    with tab:
        fig = plot_gantt(algo_name, segments[algo_name])
        st.pyplot(fig)
