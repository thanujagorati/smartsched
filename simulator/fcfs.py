"""
First Come First Served (FCFS) CPU Scheduling Algorithm.

FCFS is non-preemptive: processes are executed strictly in the order
they arrive, with no interruption once a process starts running.
It's the simplest scheduling algorithm and serves as the baseline
against which SJF, Priority, and Round Robin are compared.
"""


def fcfs(processes):
    """
    Simulate FCFS scheduling.

    Args:
        processes: list of dicts, each with keys:
            'pid'           -> process identifier
            'arrival_time'  -> time the process enters the ready queue
            'burst_time'    -> CPU time the process needs

    Returns:
        list of dicts, each with keys:
            'pid', 'start', 'finish', 'waiting_time', 'turnaround_time'
    """
    # Sort by arrival time -- this IS the definition of FCFS.
    # Without this, we're not simulating FCFS, just processing
    # list order, which may not reflect real arrival sequence.
    ordered = sorted(processes, key=lambda p: p["arrival_time"])

    time = 0
    schedule = []

    for p in ordered:
        # CPU can only start a process once it has arrived.
        # If the CPU finished the previous process before this one
        # arrived, the CPU sits idle -- max() handles that idle gap.
        start = max(time, p["arrival_time"])
        finish = start + p["burst_time"]

        schedule.append({
            "pid": p["pid"],
            "start": start,
            "finish": finish,
            "waiting_time": start - p["arrival_time"],
            "response_time": start - p["arrival_time"],  # same as waiting_time here --
            # FCFS only runs a process once, so "first got the CPU" and
            # "finished waiting" happen at the same instant
            "turnaround_time": finish - p["arrival_time"],
        })

        time = finish  # CPU becomes free at this point

    return schedule


if __name__ == "__main__":
    # Quick manual sanity check against a textbook-style example
    sample_processes = [
        {"pid": "P1", "arrival_time": 0, "burst_time": 5},
        {"pid": "P2", "arrival_time": 1, "burst_time": 3},
        {"pid": "P3", "arrival_time": 2, "burst_time": 8},
    ]

    result = fcfs(sample_processes)
    for entry in result:
        print(entry)
