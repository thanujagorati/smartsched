"""
Synthetic workload generator for training data.

Real CPU workloads have two key statistical patterns this module
tries to mimic, rather than generating naive uniform-random data:

1. ARRIVALS follow a Poisson process: events happen randomly but at
   a known average RATE (lambda). The time GAP between consecutive
   arrivals follows an exponential distribution -- that's what we
   actually sample from and then accumulate into arrival times.

2. BURST TIMES are bimodal: most real workloads are a mix of short,
   I/O-bound tasks (quick) and longer CPU-bound tasks (slow), not a
   single bell curve centered on one "typical" value. We simulate
   this by randomly picking from two different normal distributions
   -- one for the "short" cluster, one for the "long" cluster.
"""

import random


def generate_arrival_times(num_processes, arrival_rate):
    """
    Generate arrival times using a Poisson process.

    Args:
        num_processes: how many processes to generate arrivals for
        arrival_rate: lambda, the average number of arrivals per
                      unit time (higher = processes arrive more
                      frequently, closer together)

    Returns:
        list of arrival times (ints, sorted ascending, starting near 0)
    """
    arrival_times = []
    current_time = 0.0

    for _ in range(num_processes):
        # Time until the NEXT arrival follows an exponential
        # distribution when arrivals overall form a Poisson process.
        # random.expovariate's argument is lambda directly.
        gap = random.expovariate(arrival_rate)
        current_time += gap
        arrival_times.append(round(current_time))

    return arrival_times


def generate_burst_times(num_processes, short_mean=3, short_std=1,
                          long_mean=12, long_std=3, short_ratio=0.7):
    """
    Generate burst times from a bimodal distribution: a mix of a
    "short task" cluster and a "long task" cluster.

    Args:
        num_processes: how many burst times to generate
        short_mean, short_std: mean/std-dev of the short-task cluster
        long_mean, long_std: mean/std-dev of the long-task cluster
        short_ratio: fraction of processes drawn from the short
                     cluster (0.7 = 70% short tasks, 30% long --
                     mimics typical real-world I/O-bound vs
                     CPU-bound task proportions)

    Returns:
        list of burst times (ints, minimum 1 -- a process can't take
        zero or negative CPU time)
    """
    burst_times = []

    for _ in range(num_processes):
        if random.random() < short_ratio:
            value = random.gauss(short_mean, short_std)
        else:
            value = random.gauss(long_mean, long_std)

        # Clamp to at least 1 -- gauss() can produce values near/below
        # zero, especially for the short cluster, which isn't valid
        # for a burst time.
        burst_times.append(max(1, round(value)))

    return burst_times


def generate_workload(num_processes, arrival_rate=0.5, priority_range=(1, 5),
                       estimation_error=0.25, seed=None):
    """
    Generate a complete synthetic workload: a list of process dicts
    ready to feed directly into any of the four scheduling algorithms.

    Args:
        num_processes: how many processes in this workload
        arrival_rate: lambda for the Poisson arrival process
        priority_range: (min, max) inclusive range for random priority
                         assignment (needed for priority_scheduling)
        estimation_error: how noisy 'estimated_burst_time' is relative
                           to the true burst_time (0.25 = roughly +/-25%
                           error on average). This models the real-world
                           fact that an OS scheduler doesn't actually
                           know a process's exact CPU time in advance --
                           SJF has to work off an ESTIMATE, not the
                           true value. Set to 0 for perfect-knowledge
                           SJF (useful for comparison/debugging).
        seed: optional random seed, for reproducible workloads
              (useful when debugging or writing tests)

    Returns:
        list of dicts with 'pid', 'arrival_time', 'burst_time',
        'estimated_burst_time', 'priority'
    """
    if seed is not None:
        random.seed(seed)

    arrival_times = generate_arrival_times(num_processes, arrival_rate)
    burst_times = generate_burst_times(num_processes)

    workload = []
    for i in range(num_processes):
        true_burst = burst_times[i]
        # Estimated burst time = true value plus proportional noise.
        # This is what a real scheduler would actually have access to
        # when making SJF decisions -- never the true value itself.
        noise_factor = random.gauss(1.0, estimation_error)
        estimated_burst = max(1, round(true_burst * noise_factor))

        workload.append({
            "pid": f"P{i + 1}",
            "arrival_time": arrival_times[i],
            "burst_time": true_burst,
            "estimated_burst_time": estimated_burst,
            "priority": random.randint(*priority_range),
        })

    return workload


if __name__ == "__main__":
    # Reproducible example -- same seed always gives same workload,
    # useful for comparing algorithm behavior on identical input.
    workload = generate_workload(num_processes=10, arrival_rate=0.4, seed=42)

    print("Generated workload:")
    for p in workload:
        print(p)

    burst_values = [p["burst_time"] for p in workload]
    print(f"\nBurst time range: {min(burst_values)} - {max(burst_values)}")
    print("(Should show a mix of short and long values, not all clustered together)")
