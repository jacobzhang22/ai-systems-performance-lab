# Week 4 — GPU Utilization Basics

## Overview

In this experiment, I trained a simple neural network on a **Tesla T4 GPU** using a synthetic dataset with intentionally slow CPU-side preprocessing. The goal was to study **GPU underutilization**, identify where time was actually being spent in the training loop, and measure how changes to the data pipeline affected throughput.

I compared two dataloader configurations:

- **`num_workers=0`** (single-process loading)
- **`num_workers=4`** (multiprocess loading)

For each configuration, I measured:

- Data loading / batch preparation time
- GPU compute time
- Full step time
- Steps/second
- Samples/second

I also used **`torch.profiler`** to compare CPU and CUDA activity and confirm whether the bottleneck was in the model computation or in the input pipeline.

The main goal of the experiment was to understand a core ML systems idea:

> Low GPU utilization does not necessarily mean the GPU is weak — it often means the GPU is waiting.

---

## Experimental Setup

- GPU: **Tesla T4**
- Model: small feedforward neural network
- Parameters: **2,101,249**
- Batch size: **64**
- Dataset: synthetic data with intentionally slow preprocessing using a sleep in `__getitem__`
- Benchmarked:
  - `num_workers=0`
  - `num_workers=4`
- Tools used:
  - manual timing with synchronized CUDA timing
  - `nvidia-smi`
  - `torch.profiler`

For each step, I separated timing into:

- **Data time**: time spent waiting for the next batch
- **Compute time**: time spent executing the training step on the GPU
- **Full step time**: total end-to-end step duration

This let me distinguish between **CPU-side pipeline cost** and **actual GPU math**.

---

## Benchmark Results

| Config          | Avg Data Time (ms) | Avg Compute Time (ms) | Avg Full Step Time (ms) | Steps/sec | Samples/sec |
| --------------- | -----------------: | --------------------: | ----------------------: | --------: | ----------: |
| `num_workers=0` |             646.29 |                  2.22 |                  648.56 |      1.54 |       98.68 |
| `num_workers=4` |             154.04 |                  2.29 |                  156.38 |      6.39 |      409.24 |

---

## Key Observations

### 1. The baseline training loop was overwhelmingly data-bound

With `num_workers=0`, almost the entire step was spent waiting for the next batch:

- **Avg data time:** `646.29 ms`
- **Avg compute time:** `2.22 ms`
- **Avg full step time:** `648.56 ms`

This means the GPU compute portion was only a tiny fraction of total step time. The model itself was not the bottleneck; the bottleneck was the CPU-side dataloader and preprocessing pipeline.

In other words, the GPU was not slow — it was mostly idle because the next batch was not ready.

---

### 2. Increasing dataloader workers significantly improved throughput

Changing from `num_workers=0` to `num_workers=4` reduced average data time from:

- `646.29 ms` → `154.04 ms`

This reduced full step time from:

- `648.56 ms` → `156.38 ms`

And throughput increased from:

- `1.54 steps/sec` → `6.39 steps/sec`

This is roughly a **4.1x throughput improvement**, achieved without changing the model or the GPU compute at all.

This result is important because it shows that the fastest optimization was **not** to change the model or buy a larger GPU. The highest-leverage fix was to improve the bottlenecked stage in the pipeline.

---

### 3. GPU compute time stayed almost unchanged

Across both runs, compute time stayed essentially constant:

- baseline: `2.22 ms`
- optimized: `2.29 ms`

That tells us the model math did not become faster. Instead, the optimization improved how quickly batches were prepared and delivered to the GPU.

This is a key systems lesson:

> Optimizing throughput does not always mean making compute faster. Often it means reducing the time compute spends waiting.

---

### 4. The system remained data-bound even after optimization

Even after increasing worker count, the step was still dominated by data time:

- **Data:** `154.04 ms`
- **Compute:** `2.29 ms`

So while the optimization improved throughput substantially, the workload was still clearly **CPU/data limited** rather than compute limited.

The GPU had more work than before, but it was still underutilized overall because batch preparation still took much longer than the training step itself.

This means there is still room for additional optimization in the input pipeline before the GPU becomes the primary bottleneck.

---

### 5. `nvidia-smi` showing ~0% utilization was expected

While running the benchmark, `nvidia-smi` often showed GPU utilization near `0%`, even though the GPU was definitely being used.

This makes sense because the GPU compute region was extremely short relative to wall-clock step time:

- compute time was only around **2.2 ms**
- total step time was between **156 ms** and **649 ms**

So the GPU was only active for a very small slice of the total runtime. Since `nvidia-smi` samples periodically, it can easily miss these short bursts of compute and report near-zero utilization.

This was a useful reminder that coarse monitoring tools can be misleading when kernels are very short and the pipeline is dominated by CPU-side waiting.

---

## Profiler Evidence

The profiler strongly confirmed the manual benchmark results.

### With `num_workers=0`

The dominant CPU-time operator was:

- `enumerate(DataLoader)#_SingleProcessDataLoaderIter...`
- **CPU total:** `6.479 s`
- **Avg per call:** `647.892 ms`

This matched the manual benchmark almost exactly.

At the same time:

- **Self CUDA time total:** `8.653 ms`

So over 10 profiled steps, the profiler showed that the CUDA work was tiny compared to the CPU-side dataloader stall.

---

### With `num_workers=4`

The dominant CPU-time operator became:

- `enumerate(DataLoader)#_MultiProcessingDataLoaderIter...`
- **CPU total:** `1.258 s`
- **Avg per call:** `125.836 ms`

Again, this aligned closely with the benchmarked reduction in data time.

Importantly:

- **Self CUDA time total remained:** `8.653 ms`

So the GPU work stayed nearly identical across runs, while the CPU-side waiting time dropped sharply.

This is strong evidence that the observed speedup came from improving the data pipeline rather than changing compute behavior.

---

## Core Takeaways

- GPU underutilization is often caused by **CPU-side bottlenecks**, not weak GPU hardware.
- In the baseline run, nearly all step time was spent waiting for the dataloader.
- Increasing `num_workers` reduced batch preparation time by about **4x**.
- Throughput improved from **1.54** to **6.39 steps/sec** without materially changing GPU compute time.
- The bottleneck remained in the input pipeline even after optimization.
- `torch.profiler` confirmed that the dominant cost was the dataloader iterator, not CUDA kernels.
- Short GPU kernels can make `nvidia-smi` appear misleadingly idle even when the GPU is active.

---

## Key Insight

A training loop can run on a GPU and still spend almost all of its time waiting on the CPU.

In this experiment, the GPU computation itself only took about **2.2 ms per step**, while data loading took anywhere from **154 ms to 646 ms** depending on dataloader configuration. The main performance problem was not GPU execution speed — it was the system’s inability to keep the GPU fed with work.

This is the core systems takeaway from the experiment:

> GPU performance depends not just on compute speed, but on whether the surrounding pipeline can supply work fast enough to keep the device busy.
