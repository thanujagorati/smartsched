"""
Shared metrics for comparing CPU scheduling algorithm results.

Every algorithm in simulator/ (fcfs, sjf, priority, round_robin) produces
a list of per-process dicts containing at least: waiting_time and
turnaround_time. This module takes that output and computes the
summary statistics used to actually compare algorithms against each
other -- this is the "scoreboard" the ML layer will later learn to
predict from workload characteristics.
"""


def compute_metrics(schedule, total_time=None):
    """
    Compute summary metrics from a completed schedule.

    Args:
        schedule: list of dicts, each with at least 'waiting_time'
                  and 'turnaround_time' (the common output shape
                  across all four algorithms)
        total_time: optional -- the total elapsed simulation time
                    (last process's finish time). Needed for CPU
                    utilization and throughput. If not given, we
                    infer it from 'finish' times when present.

    Returns:
        dict with:
            'avg_waiting_time'
            'avg_turnaround_time'
            'max_waiting_time'   -- worst-case wait, useful for
                                     spotting starvation
            'throughput'         -- processes completed per unit time
    """
    n = len(schedule)
    if n == 0:
        raise ValueError("Cannot compute metrics on an empty schedule")

    avg_waiting = sum(p["waiting_time"] for p in schedule) / n
    avg_turnaround = sum(p["turnaround_time"] for p in schedule) / n
    max_waiting = max(p["waiting_time"] for p in schedule)

    if total_time is None:
        # Infer from 'finish' times if present (all four algorithms
        # include this field)
        finishes = [p["finish"] for p in schedule if "finish" in p]
        total_time = max(finishes) if finishes else None

    throughput = n / total_time if total_time else None

    return {
        "avg_waiting_time": round(avg_waiting, 2),
        "avg_turnaround_time": round(avg_turnaround, 2),
        "max_waiting_time": max_waiting,
        "throughput": round(throughput, 4) if throughput else None,
    }


def compare_algorithms(results_by_algorithm):
    """
    Compare metrics across multiple algorithms on the SAME workload.

    Args:
        results_by_algorithm: dict mapping algorithm name -> schedule
            e.g. {"FCFS": fcfs_result, "SJF": sjf_result, ...}

    Returns:
        dict mapping algorithm name -> metrics dict
    """
    return {
        name: compute_metrics(schedule)
        for name, schedule in results_by_algorithm.items()
    }


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))

    from fcfs import fcfs
    from sjf import sjf
    from priority import priority_scheduling

    # Same base workload run through three algorithms for a fair comparison.
    # Priority needs an extra 'priority' field, so we add one per process.
    base = [
        {"pid": "P1", "arrival_time": 0, "burst_time": 10},
        {"pid": "P2", "arrival_time": 1, "burst_time": 2},
        {"pid": "P3", "arrival_time": 2, "burst_time": 1},
    ]
    with_priority = [
        {**p, "priority": i + 1} for i, p in enumerate(base)
    ]

    results = {
        "FCFS": fcfs(base),
        "SJF": sjf(base),
        "Priority": priority_scheduling(with_priority),
    }

    comparison = compare_algorithms(results)
    for algo, metrics in comparison.items():
        print(f"{algo}: {metrics}")
