# Week 2 – Concurrency & Scaling in Python

## Overview

In this experiment, I compared how different concurrency models in Python scale under two types of workloads:

- **CPU-bound task** (pure Python loop)
- **I/O-bound task** (`sleep()`-based delay)

Each workload was executed using:

- Serial baseline
- Multithreading (`ThreadPoolExecutor`)
- Multiprocessing (`ProcessPoolExecutor`)
- Async (`asyncio`) — for the I/O-bound case

I measured wall-clock runtime across varying numbers of workers to observe scaling behavior. All experiments were conducted in Python, which uses a **Global Interpreter Lock (GIL)**.

---

## Observations and Results

### CPU-Bound Task

For the CPU-bound workload (pure Python computation):

- **Multithreading did not improve runtime and scaled poorly.**
  - Due to the GIL, only one thread can execute Python bytecode at a time.
  - Increasing the number of threads introduced context-switching overhead without enabling true parallelism.
  - As a result, runtime remained roughly the same or even worsened as worker count increased.

- **Multiprocessing improved runtime and scaled well.**
  - Each process has its own Python interpreter instance and its own GIL.
  - The operating system schedules processes across multiple CPU cores, enabling true parallel execution.
  - Performance scaled until approaching the number of available CPU cores, after which diminishing returns would be expected due to scheduling overhead and resource contention.

**Conclusion:**  
For CPU-bound pure Python workloads, multiprocessing is necessary to achieve parallel speedup. Threads are ineffective due to the GIL.

---

### I/O-Bound Task

For the I/O-bound workload (simulated using `sleep()`):

- **Multithreading scaled well.**
  - Blocking I/O operations release the GIL.
  - When one thread blocks on I/O, another thread can run.
  - This allows overlapping of waiting times, leading to near-linear speedup as worker count increases (up to overhead limits).
  - Threads are relatively lightweight compared to processes, making them efficient for this case.

- **Multiprocessing provided little advantage and often performed worse than threads.**
  - There is no GIL bottleneck for blocking I/O to solve.
  - Process startup, inter-process communication (IPC), and serialization overhead outweighed the benefits.
  - As a result, multiprocessing was less efficient for this workload.

- **Async performed the best at high concurrency.**
  - Async uses a single-threaded event loop with cooperative scheduling.
  - There is no OS-level thread context switching.
  - Coroutines resume only when their I/O is ready.
  - This reduces scheduling overhead and allows efficient handling of large numbers of concurrent I/O tasks.

**Conclusion:**  
For I/O-bound workloads, threads scale well and are often sufficient. Async scales even better at high concurrency due to lower scheduling overhead. Multiprocessing is generally unnecessary for I/O-bound tasks.

---

## Key Takeaways

1. The GIL prevents Python threads from achieving parallelism for CPU-bound pure Python code.
2. Multiprocessing enables true parallel execution by using separate interpreter instances.
3. Blocking I/O releases the GIL, allowing threads to overlap waiting time effectively.
4. Async uses cooperative scheduling and avoids OS-level thread overhead, making it highly efficient for large-scale I/O concurrency.
5. Scaling is not linear indefinitely — overhead, scheduling costs, and system limits eventually dominate.

This experiment reinforced that choosing the correct concurrency model depends entirely on whether the workload is CPU-bound or I/O-bound.
