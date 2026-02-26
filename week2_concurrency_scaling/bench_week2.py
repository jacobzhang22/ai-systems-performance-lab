# bench_week2.py
import asyncio
import time
import csv
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# ----- workloads -----
def cpu_task(n: int) -> int:
    x = 0
    for i in range(n):
        x += (i * i) % 97
    return x

def io_task_blocking(delay: float) -> int:
    time.sleep(delay)
    return 1

async def io_task_async(delay: float) -> int:
    await asyncio.sleep(delay)
    return 1

# ----- picklable helper for ProcessPool on Windows -----
def apply_star(item):
    """item is (fn, args_tuple) -> returns fn(*args_tuple)"""
    fn, args = item
    return fn(*args)

# ----- helpers -----
def bench_serial(fn, args_list) -> float:
    t0 = time.perf_counter()
    for args in args_list:
        fn(*args)
    return time.perf_counter() - t0

def bench_pool(executor_cls, fn, args_list, max_workers: int) -> float:
    # pack work to avoid lambdas (Windows needs picklable callables)
    packed = [(fn, a) for a in args_list]
    t0 = time.perf_counter()
    with executor_cls(max_workers=max_workers) as ex:
        list(ex.map(apply_star, packed))
    return time.perf_counter() - t0

async def bench_async_io(delay: float, num_tasks: int, limit: int) -> float:
    sem = asyncio.Semaphore(limit)

    async def wrapped():
        async with sem:
            return await io_task_async(delay)

    t0 = time.perf_counter()
    await asyncio.gather(*(wrapped() for _ in range(num_tasks)))
    return time.perf_counter() - t0

def run_suite_cpu(num_tasks=2000, cpu_n=200_000, workers=(1,2,4,8)):
    args = [(cpu_n,) for _ in range(num_tasks)]
    t_serial = bench_serial(cpu_task, args)

    rows = [("cpu", "serial", 0, t_serial, 1.0)]
    for w in workers:
        t_threads = bench_pool(ThreadPoolExecutor, cpu_task, args, w)
        t_procs   = bench_pool(ProcessPoolExecutor, cpu_task, args, w)
        rows.append(("cpu", "threads", w, t_threads, t_serial / t_threads))
        rows.append(("cpu", "procs",   w, t_procs,   t_serial / t_procs))
    return rows

def run_suite_io(num_tasks=10_000, delay=0.005, workers=(1,2,4,8,16,32), async_limits=(100, 500, 1000, 5000)):
    args = [(delay,) for _ in range(num_tasks)]
    t_serial = bench_serial(io_task_blocking, args)

    rows = [("io", "serial", 0, t_serial, 1.0)]
    for w in workers:
        t_threads = bench_pool(ThreadPoolExecutor, io_task_blocking, args, w)
        t_procs   = bench_pool(ProcessPoolExecutor, io_task_blocking, args, w)
        rows.append(("io", "threads", w, t_threads, t_serial / t_threads))
        rows.append(("io", "procs",   w, t_procs,   t_serial / t_procs))

    for lim in async_limits:
        t_async = asyncio.run(bench_async_io(delay, num_tasks, lim))
        rows.append(("io", "async", lim, t_async, t_serial / t_async))

    return rows

def save_csv(rows, path="results.csv"):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["workload", "model", "param", "seconds", "speedup_vs_serial"])
        w.writerows(rows)

def plot(rows, out="scaling.png"):
    import matplotlib.pyplot as plt

    for workload in ("cpu", "io"):
        subset = [r for r in rows if r[0] == workload]
        plt.figure()

        for model in sorted(set(r[1] for r in subset)):
            pts = [(r[2], r[4]) for r in subset if r[1] == model and r[2] != 0]
            if not pts:
                continue
            pts.sort(key=lambda x: x[0])
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            plt.plot(xs, ys, marker="o", label=model)

        plt.title(f"{workload.upper()} scaling (speedup vs serial)")
        plt.xlabel("workers (threads/procs) or async limit")
        plt.ylabel("speedup")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{workload}_{out}")

if __name__ == "__main__":
    workers_cpu = (1, 2, 4, 8)
    workers_io  = (1, 2, 4, 8, 16)

    rows = []
    rows += run_suite_cpu(num_tasks=300, cpu_n=800_000, workers=workers_cpu)
    rows += run_suite_io(num_tasks=2000, delay=0.001, workers=workers_io, async_limits=(100, 500, 1000))

    save_csv(rows)
    plot(rows)
    print("Wrote results.csv, cpu_scaling.png, io_scaling.png")