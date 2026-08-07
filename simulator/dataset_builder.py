"""
Dataset builder for the ML layer.

Generates many synthetic workloads, runs all four scheduling algorithms
on each, and records:
  - FEATURES: summary statistics describing the workload (things you
    could know BEFORE running any algorithm -- e.g. how many processes,
    how spread out their burst times are)
  - LABEL: which algorithm actually achieved the lowest average waiting
    time on that specific workload

This becomes the training data for:
  - a classifier that predicts the best algorithm from workload features
  - a regressor that predicts expected wait/turnaround time per algorithm

IMPORTANT (label leakage): features only ever come from the workload
itself, never from running the algorithms. If we fed in per-algorithm
results as "features," the model would just be memorizing answers
instead of learning to predict from workload shape alone.
"""

import statistics
import pandas as pd

from workload_generator import generate_workload
from fcfs import fcfs
from sjf import sjf
from priority import priority_scheduling
from round_robin import round_robin
from metrics import compute_metrics


def _pearson_correlation(x, y):
    """
    Simple Pearson correlation coefficient, no numpy dependency.
    Returns 0 if either list has zero variance (avoids division by zero).
    """
    n = len(x)
    mean_x, mean_y = statistics.mean(x), statistics.mean(y)

    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denom_x = sum((xi - mean_x) ** 2 for xi in x) ** 0.5
    denom_y = sum((yi - mean_y) ** 2 for yi in y) ** 0.5

    if denom_x == 0 or denom_y == 0:
        return 0.0
    return numerator / (denom_x * denom_y)


def extract_features(workload):
    """
    Compute summary statistics describing a workload -- these are
    the ONLY things the ML model gets to see as input.

    Args:
        workload: list of process dicts (from generate_workload)

    Returns:
        dict of feature_name -> value
    """
    burst_times = [p["burst_time"] for p in workload]
    arrival_times = sorted(p["arrival_time"] for p in workload)
    priorities = [p["priority"] for p in workload]

    # Gaps between consecutive arrivals -- describes how "bunched up"
    # vs "spread out" the workload's arrival pattern is
    gaps = [arrival_times[i] - arrival_times[i - 1]
            for i in range(1, len(arrival_times))]

    avg_burst = statistics.mean(burst_times)
    std_burst = statistics.stdev(burst_times) if len(burst_times) > 1 else 0

    # Priority-related features -- these were MISSING originally,
    # which meant the model had zero information to learn when
    # Priority scheduling actually wins. Without these, Priority's
    # outcome depends entirely on values the model never sees.
    priority_variance = statistics.variance(priorities) if len(priorities) > 1 else 0
    # Correlation between priority and burst_time: if negative, it
    # means short jobs tend to ALSO have good (low-number) priority --
    # in that case Priority scheduling behaves similarly to SJF. If
    # near zero, priorities are essentially random relative to burst
    # time, which is closer to what we're generating today.
    priority_burst_corr = (
        _pearson_correlation(priorities, burst_times)
        if len(priorities) > 1 else 0
    )

    return {
        "num_processes": len(workload),
        "avg_burst_time": round(avg_burst, 2),
        "std_burst_time": round(std_burst, 2),
        # Coefficient of variation: normalized spread, so a workload of
        # tiny processes and a workload of huge processes with similar
        # RELATIVE spread look comparable to the model
        "burst_time_cv": round(std_burst / avg_burst, 3) if avg_burst > 0 else 0,
        "min_burst_time": min(burst_times),
        "max_burst_time": max(burst_times),
        "avg_arrival_gap": round(statistics.mean(gaps), 2) if gaps else 0,
        "total_burst_time": sum(burst_times),
        "priority_variance": round(priority_variance, 2),
        "priority_burst_corr": round(priority_burst_corr, 3),
    }


def compute_score(metrics, starvation_weight=0.3, response_weight=0.5):
    """
    Combine multiple metrics into a single comparable score, instead
    of judging purely on average waiting time. Lower score = better.

    Why not just average waiting time alone? SJF is mathematically
    proven optimal for average waiting time, so using ONLY that metric
    makes SJF win almost every workload -- which makes the classifier's
    job trivial (and unrealistic; real scheduling decisions weigh more
    than one factor).

    response_weight matters a lot here: Round Robin structurally has
    HIGH average waiting time (a process's wait accumulates across
    every other process's turn, every round) -- that's expected
    scheduling theory, not a bug. RR's real advantage is fast
    RESPONSE time (how quickly a process gets its first slice of CPU),
    which matters for interactive/time-sharing systems. Without
    weighing response time, RR can structurally never win regardless
    of how the simulation is tuned.

    Args:
        metrics: dict from compute_metrics()
        starvation_weight: penalty for high worst-case waiting time
        response_weight: how much fast response time is rewarded

    Returns:
        float score (lower is better)
    """
    return (
        metrics["avg_waiting_time"]
        + starvation_weight * metrics["max_waiting_time"]
        + response_weight * metrics["avg_response_time"]
    )


def label_best_algorithm(workload, rr_overhead=1):
    """
    Run all four algorithms on the SAME workload and determine which
    one achieves the best (lowest) combined score.

    Args:
        workload: list of process dicts
        rr_overhead: context-switch cost charged per RR switch, so RR
                     isn't unrealistically free to preempt constantly

    Returns:
        (best_algorithm_name, metrics_by_algorithm dict)
    """
    fcfs_result = fcfs(workload)
    # SJF decides using 'estimated_burst_time' (noisy), not the true
    # burst_time -- this reflects that real schedulers don't have
    # perfect foresight, which is what made SJF win almost every time
    # in our first pass at this dataset.
    sjf_result = sjf(workload, estimate_key="estimated_burst_time")
    priority_result = priority_scheduling(workload)

    # A FIXED quantum unfairly penalizes RR on workloads whose typical
    # burst time doesn't match it -- e.g. quantum=4 is far too small
    # for a workload averaging burst_time=20, causing excessive
    # preemption/overhead, while it's far too large for tiny bursts,
    # causing RR to degrade into FCFS-like behavior. Real-world quantum
    # tuning is workload-aware, so we scale it relative to this
    # workload's average burst time instead of using one constant
    # for every dataset row.
    avg_burst = statistics.mean(p["burst_time"] for p in workload)
    rr_quantum = max(1, round(avg_burst / 2))

    rr_result, _ = round_robin(workload, quantum=rr_quantum,
                                context_switch_overhead=rr_overhead)

    metrics_by_algo = {
        "FCFS": compute_metrics(fcfs_result),
        "SJF": compute_metrics(sjf_result),
        "Priority": compute_metrics(priority_result),
        "RoundRobin": compute_metrics(rr_result),
    }

    best = min(metrics_by_algo, key=lambda algo: compute_score(metrics_by_algo[algo]))
    return best, metrics_by_algo


def build_dataset(num_samples=500, min_processes=5, max_processes=30,
                   arrival_rate_range=(0.2, 0.8), seed=None):
    """
    Generate a full training dataset by creating many random workloads
    and labeling each one.

    Args:
        num_samples: how many workload rows to generate
        min_processes, max_processes: range for workload size (varied,
            so the model sees both small and large workloads)
        arrival_rate_range: range for Poisson arrival_rate, also varied
            per sample for diversity
        seed: optional, for reproducible dataset generation

    Returns:
        pandas DataFrame: one row per workload, feature columns +
        'best_algorithm' label column
    """
    import random
    if seed is not None:
        random.seed(seed)

    rows = []
    for i in range(num_samples):
        num_processes = random.randint(min_processes, max_processes)
        arrival_rate = random.uniform(*arrival_rate_range)

        workload = generate_workload(num_processes, arrival_rate=arrival_rate)
        features = extract_features(workload)
        best_algo, metrics_by_algo = label_best_algorithm(workload)

        row = dict(features)
        row["best_algorithm"] = best_algo
        # Also keep each algorithm's actual avg waiting time -- useful
        # later for the regression model (predicting expected time per
        # algorithm, not just which one wins)
        for algo, m in metrics_by_algo.items():
            row[f"{algo}_avg_waiting_time"] = m["avg_waiting_time"]

        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Building a small sample dataset (50 rows) to verify everything works...")
    df = build_dataset(num_samples=50, seed=42)

    print(f"\nDataset shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nLabel distribution (which algorithm wins most often):")
    print(df["best_algorithm"].value_counts())
