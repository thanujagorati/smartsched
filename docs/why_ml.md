# Why ML Instead of Just a Fixed Rule?

## The comparison

A simple heuristic baseline was built to test whether machine learning
is actually justified here, rather than assuming it:

**Heuristic rule:** if `burst_time_cv > 0.5` (burst times are quite
uneven), predict SJF. Otherwise, predict FCFS. No learning involved --
just a hand-written if/else based on domain knowledge (SJF helps most
when burst times vary a lot).

| Approach            | Overall accuracy |
|----------------------|------------------|
| Heuristic baseline    | 92.4%            |
| Random Forest (ML)    | 93.4%            |

## The honest takeaway

On raw accuracy, the gap is small -- about 1 percentage point. If
accuracy were the only thing that mattered, the heuristic would look
"good enough."

**But the heuristic can structurally never predict Priority or
RoundRobin** -- it was only ever written to choose between SJF and
FCFS. Its recall for those two classes is mathematically guaranteed
to be exactly 0%, no matter how the data looks.

The Random Forest, despite still performing weakly on those same rare
classes (Priority f1=0.02, RoundRobin f1=0.12), at least has SOME
ability to recognize when they're appropriate -- something no fixed
rule can do without a human manually writing a new branch for every
new pattern discovered in the data.

**So the real justification for ML here isn't "it's more accurate on
paper today."** It's that:
1. ML can represent patterns a hardcoded rule structurally cannot,
   without a human needing to notice and hand-code every case
2. As the feature set and dataset grow (see `priority_burst_corr` in
   `limitations.md` -- adding it measurably improved Priority
   predictions), the ML model can absorb that improvement automatically.
   The heuristic would need to be manually rewritten every time.
3. The regression model (predicting expected wait time per algorithm,
   not just picking a winner) has no heuristic equivalent at all --
   there's no simple rule that outputs a continuous, accurate time
   estimate the way a trained regressor does (R²=0.956, MAE=3.51 time
   units on held-out data).

This is presented honestly rather than oversold: ML's advantage here
is capacity and flexibility, not a dramatic accuracy jump on this
particular dataset.
