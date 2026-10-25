---
name: performance
description: "Measure, profile, and fix real bottlenecks; prove gains with before-and-after benchmarks."
aliases: perf, profiling, benchmarking
license: MIT
source: oh-my-hermes (MIT) — adapted
---

# Performance

Performance work starts with a measurement and ends with a measurement. The
middle is one narrowly scoped change at a time. If you cannot state the metric,
the workload, and the command that produces the number, you are not ready to
optimize.

## Use this skill when

- A command, endpoint, page, job, or test suite is measurably slow.
- Memory, CPU, disk, network, startup time, or bundle size is growing.
- A change claims to improve performance and needs before-and-after evidence.
- You need to choose where to optimize rather than guess.

## Do not use this skill when

- The task is correctness or code quality with no measured cost. Use review.
- The only goal is to make code look faster. Do not optimize unmeasured code.
- The problem is an outage and the priority is restoring service. Stabilize
  first, then measure with production evidence.

## Workflow

### 1. Define the metric and the budget

Name the user-visible symptom and one primary metric:

- latency: p50, p95, p99, or worst case at a stated load;
- throughput: requests or jobs per second at a stated concurrency;
- resources: peak RSS, CPU time, disk I/O, network bytes;
- startup or build: wall-clock time to a defined milestone;
- frontend: bundle size, long tasks, layout shift, time to interactive.

Record the target and the acceptable regression, for example "p95 under 300 ms
at 50 concurrent users, no more than 5% slower than baseline". A number without
a workload is not a benchmark.

### 2. Capture a baseline

Use the same command, data, machine, and environment for every run. Warm up
caches, run multiple iterations, and report the distribution rather than one
lucky sample.

```sh
hyperfine --warmup 3 --runs 10 './bench.sh'
/usr/bin/time -v ./command
```

For a service, record the exact request mix, payload sizes, concurrency, and
database state. For a frontend, record the route, viewport, network profile, and
build mode. Save raw output so the comparison is reproducible.

### 3. Profile before editing

Find where time or memory actually goes:

```sh
python -m cProfile -o out.prof script.py
python -c "import pstats; pstats.Stats('out.prof').sort_stats('cumtime').print_stats(20)"
py-spy record -o profile.svg -- python script.py
perf stat -r 5 ./command
node --cpu-prof app.js
```

For databases, read the plan before adding an index:

```sh
psql -c "EXPLAIN (ANALYZE, BUFFERS) SELECT ..."
```

For I/O or memory, check whether the bottleneck is syscalls, allocation churn,
lock contention, garbage collection, serialization, or the network. A flame
graph or profile is evidence; a hunch is a hypothesis.

### 4. Pick one hot path

Rank candidates by measured cost and user impact. Choose the smallest change
with a plausible large effect. State the hypothesis: "most time is spent
re-parsing the same config for every request". If the profile does not support
it, choose a different path.

Do not bundle several optimizations into one change. If the result moves, you
will not know which change caused it, and you cannot revert safely.

### 5. Change, re-measure, decide

After each change:

1. rerun the exact baseline command under the same conditions;
2. compare the distribution, not a single run;
3. check correctness and resource side effects;
4. keep the change only if the gain is real and within the regression budget;
5. otherwise revert and re-rank.

Report a compact table:

```
Metric        Baseline    After      Delta
p95 latency   412 ms      288 ms     -30%
peak RSS      1.8 GB      1.1 GB     -39%
Command       ./bench.sh --users 50 (10 runs, same host)
```

### 6. Add a guard

Turn the result into a regression check where practical: a benchmark in CI with
a tolerance, a budget for bundle size, a query-count assertion, or a test that
fails on accidental N+1 behavior. A benchmark that nobody runs will drift.

## Terminal workflow

- Keep a scratch directory with the exact commands and raw outputs.
- Use `time` for a quick sanity check and `hyperfine` for comparisons.
- Pin CPU frequency or note turbo/thermal effects when results are noisy.
- Close background load; record the host, OS, language runtime, and dependency
  versions.
- Run the old and new versions back-to-back, not hours apart.
- If results vary more than the claimed gain, the benchmark is not decisive.

## Common fixes, in rough order of value

- remove repeated work: cache with a clear invalidation rule, batch queries,
  avoid N+1 access;
- reduce data: select only needed columns, stream instead of buffering, compress;
- move work off the hot path: background jobs, lazy loading, precomputation;
- fix algorithmic complexity before micro-tuning constants;
- reduce contention: smaller critical sections, pooled connections, backoff;
- avoid premature caches that add invalidation bugs and memory growth.

## Completion checklist

- [ ] Metric, workload, environment, and budget stated.
- [ ] Baseline captured with a repeatable command and multiple runs.
- [ ] Profile identifies the hot path; not guessed.
- [ ] One reversible change per measurement.
- [ ] Before-and-after numbers use identical conditions.
- [ ] Correctness and resource tradeoffs checked.
- [ ] Regression guard added or explicitly deferred.
- [ ] Claimed gain is larger than measurement noise.

## Failure modes

- Optimizing the function that looks expensive instead of the one measured.
- Comparing a single run before and after on a noisy laptop.
- Adding a cache without invalidation, masking correctness bugs.
- Reporting instruction-level tweaks while the real cost is a network round trip.
- Changing the benchmark workload between runs.
- Treating a restart, cache flush, or larger machine as a code fix.
- Keeping a change that did not move the metric because it "should" help.
- Quoting a percentage gain without the command or environment.
