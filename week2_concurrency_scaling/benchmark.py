# benchmark.py
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from workloads import cpu_task, io_task, apply_star


def bench_map(executor_cls, fn, args_list, max_workers: int) -> float:
    t0 = time.perf_counter()
    with executor_cls(max_workers=max_workers) as ex:
        # materialize results to ensure all work completes
        list(ex.map(fn, args_list))
    return time.perf_counter() - t0


def bench_serial(fn, args_list) -> float:
    t0 = time.perf_counter()
    for args in args_list:
        fn(*args)
    return time.perf_counter() - t0


def run_suite(name: str, fn, args_list, workers_list):
    print(f"\n=== {name} ===")
    t1 = bench_serial(fn, args_list)
    print(f"serial: {t1:.4f}s  (speedup 1.00x)")

    # pack work items as (fn, args) so apply_star can call fn(*args)
    packed = [(fn, a) for a in args_list]

    for w in workers_list:
        tt = bench_map(ThreadPoolExecutor, apply_star, packed, max_workers=w)
        tp = bench_map(ProcessPoolExecutor, apply_star, packed, max_workers=w)
        print(f"threads w={w}: {tt:.4f}s  (speedup {t1/tt:.2f}x)")
        print(f"procs   w={w}: {tp:.4f}s  (speedup {t1/tp:.2f}x)")


# if __name__ == "__main__":
#     WORKERS = [1, 2, 4, 8, 16, 32, 60]
#     NUM_TASKS = 2000
#     CPU_N = 100_000
#     IO_DELAY = 0.005

#     cpu_args = [(CPU_N,) for _ in range(NUM_TASKS)]
#     io_args = [(IO_DELAY,) for _ in range(NUM_TASKS)]

#     run_suite("CPU-bound (cpu_task)", cpu_task, cpu_args, WORKERS)
#     run_suite("IO-bound (io_task)", io_task, io_args, WORKERS)

if __name__ == "__main__":
    NUM_TASKS = 2000
    CPU_N = 100_000

    cpu_args = [(CPU_N,) for _ in range(NUM_TASKS)]
    packed = [(cpu_task, a) for a in cpu_args]

    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=4) as ex:
        list(ex.map(apply_star, packed))