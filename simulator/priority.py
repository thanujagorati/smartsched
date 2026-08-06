"""
Priority CPU Scheduling Algorithm -- Non-preemptive.

Same shape as SJF, but instead of picking the shortest job among
available processes, we pick the one with the best priority.

Convention used here: LOWER priority number = HIGHER priority
(priority 1 runs before priority 5). This matches most textbooks
and real OS conventions (e.g. Linux nice values).

Like SJF, this can starve low-priority processes indefinitely if
higher-priority ones keep arriving. Real systems fix this with
"aging" -- gradually increasing the priority of a process the
longer it waits, guaranteeing it eventually runs.
"""


def priority_scheduling(processes):
    """
    Simulate non-preemptive priority scheduling.

    Args:
        processes: list of dicts, each with keys:
            'pid', 'arrival_time', 'burst_time', 'priority'
            (lower number = higher priority)

    Returns:
        list of dicts, each with keys:
            'pid', 'start', 'finish', 'waiting_time', 'turnaround_time'
    """
    remaining = sorted(processes, key=lambda p: p["arrival_time"])
    completed = []
    time = 0

    while remaining:
        available = [p for p in remaining if p["arrival_time"] <= time]

        if not available:
            time = min(p["arrival_time"] for p in remaining)
            continue

        # Pick best priority among available. Tie-break by arrival_time
        # so results stay deterministic (two processes with identical
        # priority fall back to FCFS ordering between themselves).
        next_process = min(
            available, key=lambda p: (p["priority"], p["arrival_time"])
        )

        start = time
        finish = start + next_process["burst_time"]

        completed.append({
            "pid": next_process["pid"],
            "start": start,
            "finish": finish,
            "waiting_time": start - next_process["arrival_time"],
            "response_time": start - next_process["arrival_time"],  # same as
            # waiting_time -- Priority here is non-preemptive, runs once
            "turnaround_time": finish - next_process["arrival_time"],
        })

        time = finish
        remaining.remove(next_process)

    return completed


if __name__ == "__main__":
    # P3 has the worst priority (5) but arrives early -- watch it
    # get pushed behind P2 even though P2 arrives later, because
    # P2 has better (lower) priority.
    sample_processes = [
        {"pid": "P1", "arrival_time": 0, "burst_time": 4, "priority": 2},
        {"pid": "P2", "arrival_time": 1, "burst_time": 3, "priority": 1},
        {"pid": "P3", "arrival_time": 2, "burst_time": 2, "priority": 5},
    ]

    result = priority_scheduling(sample_processes)
    for entry in result:
        print(entry)
