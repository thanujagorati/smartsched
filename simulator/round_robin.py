"""
Round Robin (RR) CPU Scheduling Algorithm -- Preemptive.

Unlike FCFS/SJF/Priority, RR is preemptive: each process gets a fixed
slice of CPU time called the "time quantum." If the process isn't
finished within that slice, it's paused and sent to the back of the
ready queue, and the next process in line gets its turn.

This is what real time-sharing OSes use to feel "responsive" -- no
single process can hog the CPU. The trade-off is in the quantum size:
- Too small  -> excessive context-switching overhead (wasted time
                just swapping processes in and out, no real work done)
- Too large  -> starts behaving like FCFS, losing RR's fairness benefit

Because a process can run multiple times before finishing, we track
partial execution using 'remaining_time', separate from the original
'burst_time'.
"""

from collections import deque


def round_robin(processes, quantum):
    """
    Simulate Round Robin scheduling.

    Args:
        processes: list of dicts, each with keys:
            'pid', 'arrival_time', 'burst_time'
        quantum: int, the fixed time slice given to each process turn

    Returns:
        list of dicts, each with keys:
            'pid', 'finish', 'waiting_time', 'turnaround_time'
        (note: no single 'start' time here, since a process can run
        in multiple separate slices -- see 'segments' for that detail)
    """
    # Sort by arrival so we admit processes into the queue in order
    remaining = sorted(processes, key=lambda p: p["arrival_time"])
    for p in remaining:
        p["remaining_time"] = p["burst_time"]

    queue = deque()
    time = 0
    finished = {}
    segments = []  # (pid, start, end) -- useful later for Gantt charts

    idx = 0  # pointer into `remaining`, tracks who has been admitted

    # Admit any processes that have arrived by time 0
    while idx < len(remaining) and remaining[idx]["arrival_time"] <= time:
        queue.append(remaining[idx])
        idx += 1

    while queue:
        current = queue.popleft()

        # If CPU had to jump ahead due to earlier idle time, current
        # process still can't start before it actually arrives
        start = max(time, current["arrival_time"])
        run_time = min(quantum, current["remaining_time"])
        end = start + run_time

        segments.append((current["pid"], start, end))
        current["remaining_time"] -= run_time
        time = end

        # Admit any newly-arrived processes BEFORE re-queueing the
        # current one -- this matters for correctness. A process that
        # arrives at the exact moment current's slice ends should get
        # queued ahead of current going back in.
        while idx < len(remaining) and remaining[idx]["arrival_time"] <= time:
            queue.append(remaining[idx])
            idx += 1

        if current["remaining_time"] > 0:
            queue.append(current)  # not done -- back of the line
        else:
            finished[current["pid"]] = {
                "pid": current["pid"],
                "finish": time,
                "turnaround_time": time - current["arrival_time"],
                "waiting_time": (time - current["arrival_time"]) - current["burst_time"],
            }

        # If queue is empty but there are still unadmitted processes,
        # jump forward to the next arrival (CPU idle gap)
        if not queue and idx < len(remaining):
            time = remaining[idx]["arrival_time"]
            while idx < len(remaining) and remaining[idx]["arrival_time"] <= time:
                queue.append(remaining[idx])
                idx += 1

    # Return both final per-process stats AND the raw segments list.
    # Segments are needed for Gantt charts (a process can appear
    # multiple times); final stats match the shape used by the other
    # three algorithms so compare_algorithms() works on RR too.
    final_stats = [finished[p["pid"]] for p in sorted(processes, key=lambda p: p["pid"])]
    return final_stats, segments


if __name__ == "__main__":
    sample_processes = [
        {"pid": "P1", "arrival_time": 0, "burst_time": 5},
        {"pid": "P2", "arrival_time": 1, "burst_time": 3},
        {"pid": "P3", "arrival_time": 2, "burst_time": 8},
    ]

    result, segments = round_robin(sample_processes, quantum=2)
    for entry in result:
        print(entry)
    print("Segments:", segments)
