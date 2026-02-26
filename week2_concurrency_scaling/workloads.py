# workloads.py
import time

# -----------------------------
# CPU-bound workload
# -----------------------------
def cpu_task(n: int) -> int:
    """Pure Python CPU work."""
    total = 0
    for i in range(n):
        total += (i * i) % 97
    return total


# -----------------------------
# IO-bound workload
# -----------------------------
def io_task(delay: float) -> None:
    """Simulates I/O wait."""
    time.sleep(delay)


# -----------------------------
# Helper for executors
# -----------------------------
def apply_star(args):
    """
    Unpacks (fn, fn_args) and calls fn(*fn_args).

    Must live in an importable module (NOT __main__) so
    ProcessPool on Windows (spawn) can pickle it reliably,
    including when running under cProfile.
    """
    fn, fn_args = args
    return fn(*fn_args)