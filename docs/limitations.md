# Known Limitations

## Class imbalance in the training dataset

**What happened:** the initial dataset labeled "best algorithm" purely by
lowest average waiting time. Since SJF is mathematically proven optimal
for average waiting time (among non-preemptive algorithms), it won
~98% of generated workloads -- making the classification problem
trivial (a model could get ~98% accuracy by always predicting "SJF"
and learn nothing real).

**What I tried, in order:**
1. Gave SJF a noisy `estimated_burst_time` to decide with (instead of
   perfect knowledge of the true burst time) -- real schedulers don't
   know exact burst times in advance either.
2. Added `context_switch_overhead` to Round Robin, since RR was
   getting an unrealistic free pass on the cost of constant preemption.
3. Made RR's quantum adaptive to each workload's average burst time,
   instead of one fixed constant for every row.
4. Changed the "best algorithm" label from pure average waiting time
   to a weighted score: `avg_waiting_time + 0.3*max_waiting_time
   + 0.5*avg_response_time` -- adding a starvation penalty and
   rewarding fast response time (RR's real strength, not something
   average waiting time alone captures).

**Result after all four fixes:** SJF ~92%, FCFS ~7%, Priority ~1%,
RoundRobin <0.1%. Meaningfully better than the original ~98%/2% split,
but still heavily imbalanced.

**Why I stopped tuning further:** at some point, continuing to adjust
the simulation specifically to produce more balanced labels stops being
"more realistic modeling" and starts being reverse-engineering the
data to fit a story. SJF genuinely is close to optimal for
minimizing average wait -- that's real scheduling theory, not an
artifact of my simulation. Round Robin's actual advantage is
qualitative (fairness, low response time for interactive workloads),
which doesn't necessarily translate into "wins" under any single
scalar score.

**How this is actually handled:** at model-training time (Week 3),
using `class_weight='balanced'` in scikit-learn's RandomForestClassifier,
so the model is penalized more for misclassifying minority classes
instead of just learning to always predict the majority class.

## Round Robin's fixed vs adaptive quantum

Quantum is set adaptively per workload (`avg_burst_time / 2`) rather
than a single fixed value, since a bad quantum-to-burst-time ratio
unfairly penalizes RR on workloads it wasn't tuned for. A cleaner
future improvement: try multiple quantum values per workload and
report RR's best result, rather than one heuristic guess.
