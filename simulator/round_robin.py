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


def round_robin(processes, quantum, context_switch_overhead=0):
    """
    Simulate Round Robin scheduling.

    Args:
        processes: list of dicts, each with keys:
            'pid', 'arrival_time', 'burst_time'
        quantum: int, the fixed time slice given to each process turn
        context_switch_overhead: extra time cost charged EVERY time
            the CPU switches from one process to another. Defaults to
            0 (textbook-ideal RR with instant switching). Real OSes
            pay a real cost here -- saving/restoring registers, cache
            invalidation, etc. Setting this > 0 is what makes RR's
            fairness-vs-overhead trade-off show up in the metrics
            instead of being invisible.

    Returns:
        (final_stats, segments) tuple. final_stats is a list of dicts
        with 'pid', 'finish', 'waiting_time', 'turnaround_time'.
        segments is a list of (pid, start, end) tuples for Gantt charts.
    """
    # Sort by arrival so we admit processes into the queue in order
    remaining = sorted(processes, key=lambda p: p["arrival_time"])
    for p in remaining:
        p["remaining_time"] = p["burst_time"]

    queue = deque()
    time = 0
    finished = {}
    first_start = {}  # pid -> time of FIRST slice this process ever ran
    segments = []  # (pid, start, end) -- useful later for Gantt charts

    idx = 0  # pointer into `remaining`, tracks who has been admitted

    # Admit any processes that have arrived by time 0. If NO process
    # arrives at time 0 (e.g. the earliest arrival is t=5), the queue
    # stays empty here -- so we jump the clock forward to that first
    # arrival before entering the main loop. Without this, the
    # `while queue:` loop below would never even start.
    if remaining and remaining[0]["arrival_time"] > time:
        time = remaining[0]["arrival_time"]

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

        # Response time = when this process FIRST got the CPU, not
        # when it finally finished. Only recorded once, on its first run.
        if current["pid"] not in first_start:
            first_start[current["pid"]] = start

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
                "response_time": first_start[current["pid"]] - current["arrival_time"],
            }

        # Charge context-switch overhead if the CPU is about to hand
        # off to a different process (queue has someone else waiting,
        # or more processes are still due to arrive). No overhead is
        # charged after the very last process finishes -- there's
        # nothing left to switch TO.
        if context_switch_overhead > 0 and (queue or idx < len(remaining)):
            time += context_switch_overhead

        # If queue is empty but there are still unadmitted processes,
        # jump forward to the next arrival (CPU idle gap)
        if not queue and idx < len(remaining):
            time = max(time, remaining[idx]["arrival_time"])
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
