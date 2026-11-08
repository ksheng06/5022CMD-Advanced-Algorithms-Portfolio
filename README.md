# 5022CMD Advanced Algorithms - Portfolio (Questions 1-3)

Python source for the 5022CMD portfolio. Each program is self-contained and uses
only the Python standard library.

## Programs

| File | Purpose |
|------|---------|
| `question1_hashing.py` | Generic open-addressing hash table with linear probing, tombstone deletion and resizing; a pharmacy inventory CLI built on it; and a timed search experiment comparing the hash table with a one-dimensional list. |
| `question2_graph.py` | Generic unweighted directed graph (adjacency lists), a `Course` entity, a course-prerequisite service and a menu-driven CLI (mandatory + advanced features). |
| `question3_concurrency.py` | Direct recursive Fibonacci (no memoisation), operation counting, and a timed comparison of three-thread versus sequential execution for Fibonacci(35), (38), (40) over 10 rounds. |

## Requirements

* Python 3.9 or newer (developed with 3.12 type-hint style using `from __future__ import annotations`).
* **No packages to install.** No `pip install` step. No NetworkX, NumPy, or other libraries.
* Check your version: `python --version` (use `python3` on macOS/Linux if needed).

## How to run

```
python question1_hashing.py                 # interactive pharmacy menu
python question1_hashing.py --demo          # deterministic collision / tombstone / resize demo
python question1_hashing.py --experiment    # hash table vs list timing, then exit

python question2_graph.py                   # interactive course menu
python question2_graph.py --demo            # deterministic scripted demonstration

python question3_concurrency.py             # full assessed experiment (can take many minutes)
python question3_concurrency.py --self-test # correctness checks only
```

Syntax check without running: `python -m py_compile question1_hashing.py question2_graph.py question3_concurrency.py`

## Reproducing the experiments

**Question 1** (defaults: sizes 1000/5000/10000, 200 queries per timed batch, 20 timed batches, 2 warm-up batches, seed 5022):

```
python question1_hashing.py --experiment
python question1_hashing.py --experiment --sizes 2000 20000 --queries 300 --repeats 30 --warmup 3 --seed 7
```

Reduce `--sizes` or `--repeats` if the run is too slow on your machine.
Keep the seed and settings you used - they are recorded in the CSV.

**Question 3** (defaults: inputs 35 38 40, 10 rounds, rounds alternate which mode runs first, 2 s pause between runs):

```
python question3_concurrency.py
python question3_concurrency.py --order block --pause-seconds 5
python question3_concurrency.py --rounds 2 --inputs 20 22 24 --pause-seconds 0   # smoke test only
```

For cleaner measurements: plug in the power supply, close heavy applications, and do not use
the computer while it runs. Use the default inputs (35 38 40) and 10 rounds for assessed evidence.

## Result files (written to `results/`)

| File | Created by | Content |
|------|-----------|---------|
| `question1_search_results.csv` | Q1 experiment | Raw per-trial nanosecond timings (`record_type=raw`) and summary statistics (`record_type=summary`), plus hash-table capacity, load factor and mean probe counts. |
| `question3_threaded_results.csv` | Q3 | Per-round threaded durations (ns), computed Fibonacci values, verification flag. |
| `question3_sequential_results.csv` | Q3 | Same for the sequential runs. |
| `question3_comparison_summary.csv` | Q3 | Mean/median/stdev/min/max, speedup ratio, percentage change, design settings, interpreter/OS facts read at run time. |

The CSV files contain only data produced by your own runs. Generated CSVs are your evidence:
commit them, but do not edit the numbers by hand.
