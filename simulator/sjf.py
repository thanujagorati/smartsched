"""
Shortest Job First (SJF) CPU Scheduling Algorithm -- Non-preemptive.

At each point the CPU is free, SJF looks at all processes that have
already arrived (and aren't done yet) and picks the one with the
smallest burst time. Unlike FCFS, order of arrival doesn't matter --
only "who is shortest among those currently waiting."

This is provably optimal for minimizing *average* waiting time among
non-preemptive algorithms, but it can starve long processes if short
ones keep arriving ahead of them.
"""


def sjf(processes, estimate_key="burst_time"):
    """
    Simulate non-preemptive SJF scheduling.

    Args:
        processes: list of dicts, each with keys:
            'pid', 'arrival_time', 'burst_time'
        estimate_key: which field SJF uses to DECIDE which process to
            run next. Defaults to 'burst_time' (perfect knowledge --
            classic textbook SJF). In practice, pass
            'estimated_burst_time' to simulate a real scheduler that
            only has a noisy estimate of how long a process will
            take -- the ACTUAL execution still uses the true
            'burst_time', only the decision-making is affected.
            This matters: SJF's optimality proof assumes perfect
            knowledge, which real OSes don't have.

    Returns:
        list of dicts, each with keys:
            'pid', 'start', 'finish', 'waiting_time', 'turnaround_time'
    """
    # Work on a copy so we don't mutate the caller's list
    remaining = sorted(processes, key=lambda p: p["arrival_time"])
    completed = []
    time = 0

    while remaining:
        # Find all processes that have arrived by the current time
        available = [p for p in remaining if p["arrival_time"] <= time]

        if not available:
            # No process has arrived yet -- CPU sits idle.
            # Jump time forward to the next arrival instead of
            # ticking one unit at a time (keeps this efficient).
            time = min(p["arrival_time"] for p in remaining)
            continue

        # Decide using estimate_key (perfect knowledge by default, or
        # a noisy estimate if estimate_key='estimated_burst_time').
        # Tie-break by arrival_time to keep results deterministic.
        next_process = min(available, key=lambda p: (p[estimate_key], p["arrival_time"]))

        start = time
        # ACTUAL execution always uses the true burst_time, regardless
        # of what estimate was used to pick this process -- a wrong
        # estimate doesn't change reality, just the scheduling decision.
        finish = start + next_process["burst_time"]

        completed.append({
            "pid": next_process["pid"],
            "start": start,
            "finish": finish,
            "waiting_time": start - next_process["arrival_time"],
            "turnaround_time": finish - next_process["arrival_time"],
        })

        time = finish
        remaining.remove(next_process)

    return completed


def _average_waiting_time(schedule):
    return sum(p["waiting_time"] for p in schedule) / len(schedule)


if __name__ == "__main__":
    # Case 1: same sample used for FCFS. SJF happens to match FCFS
    # here, because by the time the CPU frees up, arrival order and
    # burst-time order already agree.
    print("=== Case 1: SJF matches FCFS (coincidence, not a bug) ===")
    sample_processes = [
        {"pid": "P1", "arrival_time": 0, "burst_time": 5},
        {"pid": "P2", "arrival_time": 1, "burst_time": 3},
        {"pid": "P3", "arrival_time": 2, "burst_time": 8},
    ]
    result = sjf(sample_processes)
    for entry in result:
        print(entry)
    print(f"Average waiting time: {_average_waiting_time(result):.2f}\n")

    # Case 2: a long process arrives first, short ones arrive right
    # after. This is where SJF actually beats FCFS on average wait.
    print("=== Case 2: SJF vs FCFS diverge -- long job blocks short ones first ===")
    divergent_processes = [
        {"pid": "P1", "arrival_time": 0, "burst_time": 10},
        {"pid": "P2", "arrival_time": 1, "burst_time": 2},
        {"pid": "P3", "arrival_time": 2, "burst_time": 1},
    ]

    from fcfs import fcfs

    fcfs_result = fcfs(divergent_processes)
    sjf_result = sjf(divergent_processes)

    print("FCFS order:")
    for entry in fcfs_result:
        print(entry)
    print(f"FCFS average waiting time: {_average_waiting_time(fcfs_result):.2f}\n")

    print("SJF order:")
    for entry in sjf_result:
        print(entry)
    print(f"SJF average waiting time: {_average_waiting_time(sjf_result):.2f}")
    print("\nNote: P1 still runs first in both cases (nothing else has")
    print("arrived at t=0) -- SJF's benefit only kicks in for P2 and P3,")
    print("who get reordered by burst time once the CPU frees up at t=10.")
