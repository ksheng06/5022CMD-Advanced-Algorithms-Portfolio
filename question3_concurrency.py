#!/usr/bin/env python3
"""5022CMD Advanced Algorithms - Question 3: Concurrency, threading and Fibonacci.

Definition used throughout
--------------------------
    Fib(0) = 0,  Fib(1) = 1,  Fib(n) = Fib(n-1) + Fib(n-2)  for n >= 2

Recurrence (counting model defined in `fibonacci_counted`)
----------------------------------------------------------
Each call performs a constant number c of primitive operations (one comparison,
and for n >= 2 two subtractions, one addition, one return) plus two recursive calls:

    T(0) = T(1) = c0               (base case: comparison + return)
    T(n) = T(n-1) + T(n-2) + c     (n >= 2)

T(n) grows like Fib(n), which is Theta(phi^n) with phi = (1 + sqrt 5) / 2 ~ 1.618.
Since phi^n < 2^n, O(2^n) is a valid (looser) upper bound. The number of calls is
exactly 2*Fib(n+1) - 1. Space: the recursion depth is n, so O(n) stack frames.
Exact operation counts depend on which operations are counted; the growth rate
does not.

Experiment design
-----------------
* Threaded round: three threading.Thread objects (one per input) are created
  BEFORE timing. The timer starts immediately before the first start() and stops
  immediately after the last join() returns. This is the operational form of
  "end time of the last-finishing thread minus start time of the first-started thread".
* Sequential round: the same worker function runs the same three inputs one after
  another between the same two timer reads.
* Results are written into a pre-allocated list, one slot per thread, so no two
  threads write to the same location: no lock is needed. Worker exceptions are
  captured per slot and re-raised after joining.
* Verification and printing happen OUTSIDE the timed section.
* Timer: time.perf_counter_ns().
* Rounds alternate which mode runs first (threaded-first on odd rounds, sequential-first
  on even rounds) to reduce order bias; `--order block` runs all threaded rounds first.
  A pause between timed runs reduces (does not remove) heat/load carry-over.

CPython note: in a standard GIL-enabled CPython build only one thread executes
Python bytecode at a time, so this CPU-bound work is concurrent but not parallel.
The program records the interpreter version and GIL status for your evidence.

Run:
    python question3_concurrency.py                      # full assessed experiment
    python question3_concurrency.py --rounds 2 --inputs 20 22 24   # quick smoke test
    python question3_concurrency.py --self-test          # correctness checks only
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

RESULTS_DIR = Path(__file__).resolve().parent / "results"
ASSESSED_INPUTS: Tuple[int, int, int] = (35, 38, 40)
DEFAULT_ROUNDS = 10
DEFAULT_PAUSE_SECONDS = 2.0
COUNT_TABLE_MAX_N = 20           # instrumented counting is only practical for small n
MAX_COUNTED_N = 30

# Known reference values, used ONLY to verify correctness (never timed).
KNOWN_FIBONACCI: Dict[int, int] = {
    0: 0, 1: 1, 2: 1, 3: 2, 4: 3, 5: 5, 6: 8, 7: 13, 8: 21, 9: 34, 10: 55,
    30: 832040, 35: 9227465, 38: 39088169, 40: 102334155,
}


# =========================================================================== #
# A. FIBONACCI (direct recursion - deliberately NOT memoised)
# =========================================================================== #
def validate_fibonacci_input(n: object) -> int:
    """Return n if it is a non-negative int (bool rejected); else raise."""
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError(f"n must be an integer, got {type(n).__name__}")
    if n < 0:
        raise ValueError("n must be non-negative")
    return n


def fibonacci_recursive(n: int) -> int:
    """Recursive core: no validation and no caching, so the cost is exponential.

    Base cases: Fib(0) = 0, Fib(1) = 1. Callers must pass a valid n.
    """
    if n < 2:
        return n
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)


def fibonacci(n: int) -> int:
    """Validated public entry point (validates once, then calls the core)."""
    return fibonacci_recursive(validate_fibonacci_input(n))


@dataclass
class OperationCounts:
    """Primitive-operation tallies under the model documented in fibonacci_counted."""

    calls: int = 0
    comparisons: int = 0
    subtractions: int = 0
    additions: int = 0
    returns: int = 0

    @property
    def total(self) -> int:
        return self.calls + self.comparisons + self.subtractions + self.additions + self.returns


def fibonacci_counted(n: int, counts: OperationCounts) -> int:
    """Same recursion as fibonacci_recursive, with operation counting.

    Counting model (one unit each): function call, comparison `n < 2`,
    subtraction (n-1, n-2), addition of the two results, return.
    Only for small n (the instrumentation slows the function noticeably).
    """
    counts.calls += 1
    counts.comparisons += 1
    if n < 2:
        counts.returns += 1
        return n
    counts.subtractions += 2
    result = fibonacci_counted(n - 1, counts) + fibonacci_counted(n - 2, counts)
    counts.additions += 1
    counts.returns += 1
    return result


def fibonacci_iterative_reference(n: int) -> int:
    """Independent O(n) reference used ONLY to verify results and counts.

    It is never used inside any timed section or in place of the assessed function.
    """
    previous, current = 0, 1
    for _ in range(n):
        previous, current = current, previous + current
    return previous


def format_operation_table(max_n: int = COUNT_TABLE_MAX_N) -> str:
    """Derived table: calls, ops, 2*Fib(n+1)-1 check and growth ratio (not timings)."""
    if not 1 <= max_n <= MAX_COUNTED_N:
        raise ValueError(f"max_n must be between 1 and {MAX_COUNTED_N}")
    header = (f"{'n':>3} {'Fib(n)':>8} {'calls':>10} {'2*Fib(n+1)-1':>13} "
              f"{'total ops':>11} {'calls(n)/calls(n-1)':>20} {'2^n':>10}")
    lines = [header, "-" * len(header)]
    previous_calls = 0
    for n in range(max_n + 1):
        counts = OperationCounts()
        value = fibonacci_counted(n, counts)
        ratio = f"{counts.calls / previous_calls:.4f}" if previous_calls else "-"
        lines.append(f"{n:>3} {value:>8} {counts.calls:>10} {2 * fibonacci_iterative_reference(n + 1) - 1:>13} "
                     f"{counts.total:>11} {ratio:>20} {2 ** n:>10}")
        previous_calls = counts.calls
    lines.append("Ratio tends towards phi = 1.6180339887... (counting model: call, compare, "
                 "subtract, add, return).")
    return "\n".join(lines)


def run_self_tests() -> None:
    """Correctness checks; raises AssertionError-style RuntimeError on failure."""
    problems: List[str] = []
    for n, expected in KNOWN_FIBONACCI.items():
        if n <= 30 and fibonacci(n) != expected:
            problems.append(f"fibonacci({n}) != {expected}")
        if fibonacci_iterative_reference(n) != expected:
            problems.append(f"reference({n}) != {expected}")
    for n in range(0, 21):
        counts = OperationCounts()
        if fibonacci_counted(n, counts) != fibonacci_iterative_reference(n):
            problems.append(f"counted value wrong for n={n}")
        if counts.calls != 2 * fibonacci_iterative_reference(n + 1) - 1:
            problems.append(f"call count formula fails for n={n}")
    for bad in (-1, 2.5, "7", None, True):
        try:
            fibonacci(bad)          # type: ignore[arg-type]
            problems.append(f"fibonacci({bad!r}) should have been rejected")
        except (TypeError, ValueError):
            pass
    if problems:
        raise RuntimeError("self-test failures: " + "; ".join(problems))
    print("Self-tests passed (values, call-count formula, input validation).")


# =========================================================================== #
# B/C. EXPERIMENT MACHINERY
# =========================================================================== #
class WorkerError(RuntimeError):
    """A Fibonacci worker raised, or a result failed verification."""


@dataclass(frozen=True)
class RoundOutcome:
    elapsed_ns: int
    values: Tuple[Optional[int], ...]
    errors: Tuple[Optional[BaseException], ...]


def fibonacci_worker(slot: int, n: int, results: List[Optional[int]],
                     errors: List[Optional[BaseException]]) -> None:
    """Compute Fib(n) into results[slot]; capture any exception in errors[slot].

    Used by BOTH the threaded and the sequential experiment, so the work and the
    error handling are identical. Each call touches only its own slot.
    """
    try:
        results[slot] = fibonacci_recursive(n)
    except BaseException as exc:            # noqa: BLE001 - reported after join
        errors[slot] = exc


def run_threaded_round(inputs: Sequence[int]) -> RoundOutcome:
    """One threaded round: one thread per input."""
    results: List[Optional[int]] = [None] * len(inputs)       # pre-allocated
    errors: List[Optional[BaseException]] = [None] * len(inputs)
    threads = [threading.Thread(target=fibonacci_worker, args=(i, n, results, errors),
                                name=f"fib-{n}") for i, n in enumerate(inputs)]
    start = time.perf_counter_ns()          # immediately before first start()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    end = time.perf_counter_ns()            # after the final join() returns
    return RoundOutcome(end - start, tuple(results), tuple(errors))


def run_sequential_round(inputs: Sequence[int]) -> RoundOutcome:
    """One sequential round: same worker, same inputs, no threads."""
    results: List[Optional[int]] = [None] * len(inputs)
    errors: List[Optional[BaseException]] = [None] * len(inputs)
    start = time.perf_counter_ns()
    for i, n in enumerate(inputs):
        fibonacci_worker(i, n, results, errors)
    end = time.perf_counter_ns()
    return RoundOutcome(end - start, tuple(results), tuple(errors))


def verify_outcome(inputs: Sequence[int], outcome: RoundOutcome, expected: Sequence[int]) -> None:
    """Raise WorkerError on worker exceptions or wrong values (after timing)."""
    failures = [f"Fib({n}) raised {err!r}" for n, err in zip(inputs, outcome.errors) if err is not None]
    if failures:
        raise WorkerError("; ".join(failures))
    for n, got, want in zip(inputs, outcome.values, expected):
        if got != want:
            raise WorkerError(f"Fib({n}) returned {got}, expected {want}")


@dataclass
class RoundRecord:
    mode: str                     # "threaded" | "sequential"
    round_number: int
    run_position: int             # order in which this timed run executed overall
    elapsed_ns: int
    values: Tuple[int, ...]


@dataclass
class ExperimentData:
    inputs: Tuple[int, ...]
    rounds: int
    order: str
    pause_seconds: float
    records: List[RoundRecord] = field(default_factory=list)

    def elapsed(self, mode: str) -> List[int]:
        return [r.elapsed_ns for r in sorted(self.records, key=lambda r: r.round_number) if r.mode == mode]


def run_experiment(inputs: Sequence[int], rounds: int, order: str, pause_seconds: float) -> ExperimentData:
    """Execute all timed runs; verify every result; keep every raw duration."""
    expected = [fibonacci_iterative_reference(n) for n in inputs]
    for n, value in zip(inputs, expected):
        if n in KNOWN_FIBONACCI and KNOWN_FIBONACCI[n] != value:
            raise WorkerError(f"reference value for Fib({n}) disagrees with known table")
    data = ExperimentData(tuple(inputs), rounds, order, pause_seconds)
    runners = {"threaded": run_threaded_round, "sequential": run_sequential_round}

    schedule: List[Tuple[str, int]] = []
    if order == "block":
        schedule = [("threaded", r) for r in range(1, rounds + 1)] + \
                   [("sequential", r) for r in range(1, rounds + 1)]
    else:   # alternate
        for r in range(1, rounds + 1):
            pair = ["threaded", "sequential"] if r % 2 == 1 else ["sequential", "threaded"]
            schedule += [(mode, r) for mode in pair]

    for position, (mode, round_number) in enumerate(schedule, start=1):
        outcome = runners[mode](inputs)                       # timed inside
        verify_outcome(inputs, outcome, expected)             # not timed
        data.records.append(RoundRecord(mode, round_number, position, outcome.elapsed_ns,
                                        tuple(int(v) for v in outcome.values if v is not None)))
        print(f"  run {position:>2}/{len(schedule)}  {mode:<10} round {round_number:>2}: "
              f"{outcome.elapsed_ns:>16,} ns  (verified)", flush=True)
        if pause_seconds > 0 and position < len(schedule):
            time.sleep(pause_seconds)                         # outside timing
    return data


# =========================================================================== #
# D. STATISTICS, REPORTING, CSV
# =========================================================================== #
@dataclass(frozen=True)
class Summary:
    mean: float
    median: float
    stdev: float        # sample standard deviation (n-1); 0.0 if fewer than 2 values
    minimum: int
    maximum: int


def summarise(values: Sequence[int]) -> Summary:
    if not values:
        raise ValueError("no values to summarise")
    return Summary(statistics.fmean(values), statistics.median(values),
                   statistics.stdev(values) if len(values) > 1 else 0.0,
                   min(values), max(values))


@dataclass(frozen=True)
class Comparison:
    speedup_ratio: float            # sequential mean / threaded mean
    percentage_change: float        # (threaded - sequential) / sequential * 100


def compare(threaded: Summary, sequential: Summary) -> Comparison:
    return Comparison(sequential.mean / threaded.mean,
                      (threaded.mean - sequential.mean) / sequential.mean * 100.0)


def gil_status() -> str:
    check = getattr(sys, "_is_gil_enabled", None)       # present in CPython 3.13+
    if check is None:
        return "not reported by this interpreter (sys._is_gil_enabled unavailable)"
    return "enabled" if check() else "disabled (free-threaded build)"


def environment_info() -> Dict[str, str]:
    """Facts read from the running interpreter/OS (nothing assumed)."""
    return {"python_version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "gil_status": gil_status(),
            "platform": platform.platform(),
            "logical_cpu_count": str(os.cpu_count())}


def print_round_table(label: str, values: Sequence[int], summary: Summary) -> None:
    print(f"\n{label}")
    print(f"  {'Round':>5}  {'Time (ns)':>20}")
    for i, ns in enumerate(values, start=1):
        print(f"  {i:>5}  {ns:>20,}")
    print(f"  Average : {summary.mean:>20,.1f} ns")
    print(f"  Median  : {summary.median:>20,.1f} ns")
    print(f"  Std dev : {summary.stdev:>20,.1f} ns (sample)")
    print(f"  Min/Max : {summary.minimum:,} / {summary.maximum:,} ns")


def describe_direction(change: float) -> str:
    if change > 0:
        return "threaded runs took LONGER than sequential runs on average"
    if change < 0:
        return "threaded runs took LESS time than sequential runs on average"
    return "threaded and sequential averages were equal"


def write_mode_csv(path: Path, data: ExperimentData, mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fib_columns = [f"fib_{n}" for n in data.inputs]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["round", "run_position", "mode", "elapsed_ns", *fib_columns, "verified"])
        for r in sorted((x for x in data.records if x.mode == mode), key=lambda x: x.round_number):
            writer.writerow([r.round_number, r.run_position, r.mode, r.elapsed_ns, *r.values, True])


def write_summary_csv(path: Path, data: ExperimentData, threaded: Summary,
                      sequential: Summary, comparison: Comparison) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Tuple[str, str, object]] = []
    for label, s in (("threaded", threaded), ("sequential", sequential)):
        rows += [(label, "rounds", data.rounds), (label, "mean_ns", f"{s.mean:.3f}"),
                 (label, "median_ns", f"{s.median:.3f}"), (label, "sample_stdev_ns", f"{s.stdev:.3f}"),
                 (label, "min_ns", s.minimum), (label, "max_ns", s.maximum)]
    rows += [("comparison", "speedup_ratio_sequential_mean_over_threaded_mean", f"{comparison.speedup_ratio:.6f}"),
             ("comparison", "percentage_change_threaded_vs_sequential", f"{comparison.percentage_change:.6f}"),
             ("comparison", "interpretation", describe_direction(comparison.percentage_change)),
             ("design", "inputs", " ".join(map(str, data.inputs))),
             ("design", "order", data.order), ("design", "pause_seconds", data.pause_seconds)]
    rows += [("environment", k, v) for k, v in environment_info().items()]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["section", "metric", "value"])
        writer.writerows(rows)


def report(data: ExperimentData, output_dir: Path) -> None:
    threaded_ns, sequential_ns = data.elapsed("threaded"), data.elapsed("sequential")
    t_sum, s_sum = summarise(threaded_ns), summarise(sequential_ns)
    comparison = compare(t_sum, s_sum)
    print_round_table(f"MULTITHREADED ({len(data.inputs)} threads per round, inputs {list(data.inputs)})",
                      threaded_ns, t_sum)
    print_round_table("SEQUENTIAL (same function, same inputs, no threads)", sequential_ns, s_sum)
    print("\nCOMPARISON")
    print(f"  Speedup ratio (sequential average / threaded average) : {comparison.speedup_ratio:.4f}")
    print(f"  Percentage change ((threaded - sequential)/sequential * 100) : {comparison.percentage_change:+.2f}%")
    print("  Key: positive change = threading took longer; negative change = threading took less time.")
    print(f"  Measured direction: {describe_direction(comparison.percentage_change)}.")
    write_mode_csv(output_dir / "question3_threaded_results.csv", data, "threaded")
    write_mode_csv(output_dir / "question3_sequential_results.csv", data, "sequential")
    write_summary_csv(output_dir / "question3_comparison_summary.csv", data, t_sum, s_sum, comparison)
    print(f"\nCSV files written to: {output_dir}")


# =========================================================================== #
# Entry point
# =========================================================================== #
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="5022CMD Q3 - Fibonacci threads vs sequential")
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS, help="measured rounds (default 10)")
    parser.add_argument("--inputs", type=int, nargs=3, default=list(ASSESSED_INPUTS),
                        metavar=("N1", "N2", "N3"), help="three Fibonacci inputs (default 35 38 40)")
    parser.add_argument("--order", choices=("alternate", "block"), default="alternate",
                        help="alternate which mode runs first each round, or run all threaded then all sequential")
    parser.add_argument("--pause-seconds", type=float, default=DEFAULT_PAUSE_SECONDS,
                        help="pause between timed runs (outside timing)")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR, help="folder for CSV files")
    parser.add_argument("--self-test", action="store_true", help="run correctness checks and exit")
    parser.add_argument("--skip-count-table", action="store_true", help="skip the operation-count table")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        for n in args.inputs:
            validate_fibonacci_input(n)
        if args.rounds < 1 or args.pause_seconds < 0:
            raise ValueError("rounds must be >= 1 and pause must be >= 0")
    except (TypeError, ValueError) as error:
        print(f"Invalid arguments: {error}", file=sys.stderr)
        return 2

    run_self_tests()
    if args.self_test:
        return 0
    if not args.skip_count_table:
        print("\nOPERATION-COUNT TABLE (derived by instrumented recursion, not timing)")
        print(format_operation_table())

    print("\nENVIRONMENT (read from the running interpreter)")
    for key, value in environment_info().items():
        print(f"  {key}: {value}")
    print(f"\nEXPERIMENT: inputs={list(args.inputs)}, rounds={args.rounds}, order={args.order}, "
          f"pause={args.pause_seconds}s. Close other programs and keep the machine on mains power.")
    print("Large inputs can take several minutes; progress is printed after each timed run.\n")
    data = run_experiment(args.inputs, args.rounds, args.order, args.pause_seconds)
    report(data, args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
