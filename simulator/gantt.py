"""
Gantt chart data structure.

A Gantt chart for CPU scheduling is just a timeline: which process
ran during which time interval. Round Robin already tracks this
naturally (a process can run in several separate slices), but
FCFS/SJF/Priority only return one (start, finish) per process since
they're non-preemptive -- each process runs exactly once, start to
finish, with no interruption.

This module converts any algorithm's schedule output into a single,
consistent list of segments: (pid, start, end). This is the exact
shape a Streamlit/matplotlib Gantt chart plotting function will need
later, regardless of which algorithm produced it.
"""


def to_gantt_segments(schedule):
    """
    Convert a non-preemptive algorithm's output (FCFS, SJF, Priority)
    into Gantt chart segments.

    Args:
        schedule: list of dicts with 'pid', 'start', 'finish'
                  (the shape returned by fcfs/sjf/priority_scheduling)

    Returns:
        list of tuples: (pid, start, end), sorted by start time
    """
    segments = [(p["pid"], p["start"], p["finish"]) for p in schedule]
    return sorted(segments, key=lambda s: s[1])


def gantt_summary(segments):
    """
    Quick human-readable summary of a Gantt timeline, useful for
    sanity-checking output before plotting it.

    Args:
        segments: list of (pid, start, end) tuples

    Returns:
        str, e.g. "P1[0-5] -> P2[5-8] -> P3[8-16]"
    """
    return " -> ".join(f"{pid}[{start}-{end}]" for pid, start, end in segments)


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))

    from fcfs import fcfs
    from round_robin import round_robin

    sample = [
        {"pid": "P1", "arrival_time": 0, "burst_time": 5},
        {"pid": "P2", "arrival_time": 1, "burst_time": 3},
        {"pid": "P3", "arrival_time": 2, "burst_time": 8},
    ]

    # Non-preemptive algorithm -> convert to Gantt segments
    fcfs_result = fcfs(sample)
    fcfs_segments = to_gantt_segments(fcfs_result)
    print("FCFS Gantt:", gantt_summary(fcfs_segments))

    # Round Robin returns (final_stats, segments) directly -- segments
    # is already in Gantt-ready shape, no conversion needed like the
    # non-preemptive algorithms required above.
    rr_result, rr_segments = round_robin(sample, quantum=2)
    print("Round Robin Gantt:", gantt_summary(rr_segments))
