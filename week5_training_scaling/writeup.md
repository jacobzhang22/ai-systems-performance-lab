# Week 5 — Training Performance Fundamentals

## Overview

In this experiment, I studied how **training performance scales with batch size** when training a small transformer model on a **Tesla T4 GPU**. The goal was to understand a core ML systems tradeoff:

- Increasing batch size can improve **GPU utilization and throughput**
- However, batch size also increases **activation memory usage**
- Eventually the GPU becomes **compute-saturated**, after which larger batches provide no throughput improvement but continue increasing memory usage

To analyze this behavior, I trained a small transformer model on a **synthetic dataset** and varied the batch size across several orders of magnitude. For each configuration, I measured:

- Average training step time
- Training throughput (samples/sec)
- Peak GPU memory usage

The experiment was designed to capture three important system behaviors:

1. GPU underutilization at small batch sizes
2. Throughput saturation once compute resources are fully utilized
3. Linear activation memory growth that eventually causes **Out-Of-Memory (OOM)** errors

---

## Experimental Setup

Hardware:

- GPU: **NVIDIA Tesla T4 (16 GB VRAM)**

Model:

- Small transformer encoder
- Hidden dimension: **256**
- Attention heads: **4**
- Transformer layers: **2**
- Feedforward dimension: **512**
- Sequence length: **128**

Training configuration:

- Optimizer: **AdamW**
- Loss: **CrossEntropyLoss**
- Synthetic token dataset
- Each experiment measured:
  - **20 timed training steps**
  - **5 warmup steps** (excluded from measurement)

Batch sizes tested:

```
8, 16, 32, 64, 128, 256, 512, 1024
```

The experiment continued until the first configuration that produced a **CUDA Out Of Memory error**, which occurred at **batch size 1152**.

---

## Benchmark Results

| Batch Size | Avg Step Time (s) | Throughput (samples/sec) | Peak GPU Memory (MB) |
| ---------: | ----------------: | -----------------------: | -------------------: |
|          8 |           0.00744 |                  1075.08 |               169.70 |
|         16 |           0.01304 |                  1227.14 |               278.54 |
|         32 |           0.02419 |                  1323.05 |               488.08 |
|         64 |           0.04597 |                  1392.18 |               916.89 |
|        128 |           0.08841 |                  1447.85 |              1775.18 |
|        256 |           0.17603 |                  1454.26 |              3493.55 |
|        512 |           0.35505 |                  1442.06 |              6920.05 |
|       1024 |           0.70938 |                  1443.51 |             13781.16 |

First failing configuration:

```
Batch size = 1152 → CUDA Out Of Memory
```

---

## Key Observations

### 1. Small batch sizes underutilize the GPU

At smaller batch sizes, throughput increases significantly:

- 8 → **1075 samples/sec**
- 16 → **1227 samples/sec**
- 32 → **1323 samples/sec**
- 64 → **1392 samples/sec**
- 128 → **1447 samples/sec**

This improvement occurs because larger batches allow the GPU to execute **larger matrix operations** and distribute work across more **Streaming Multiprocessors (SMs)**. Small batches do not provide enough parallel work to fully utilize the GPU, so many compute units remain idle.

Increasing batch size therefore improves **kernel efficiency, memory access patterns, and SM utilization**, resulting in higher throughput.

---

### 2. Throughput saturates once GPU compute is fully utilized

After batch size 128–256, throughput stops improving:

- 128 → **1447 samples/sec**
- 256 → **1454 samples/sec**
- 512 → **1442 samples/sec**
- 1024 → **1443 samples/sec**

Despite doubling the batch size, throughput remains almost constant. This indicates that the GPU's compute resources are now **fully saturated**. At this point, increasing batch size only increases the amount of work per step rather than improving parallelism.

This behavior is also visible in the **step time scaling**:

| Batch | Step Time |
| ----: | --------: |
|   256 |  ~0.176 s |
|   512 |  ~0.355 s |
|  1024 |  ~0.709 s |

Step time roughly doubles when batch size doubles. This indicates that the GPU is now executing **fully utilized compute workloads**, and the system has transitioned from an **underutilized regime** to a **compute-saturated regime**.

---

### 3. Activation memory grows linearly with batch size

Peak GPU memory usage increases almost linearly:

| Batch |   Memory |
| ----: | -------: |
|     8 |   169 MB |
|    16 |   278 MB |
|    32 |   488 MB |
|    64 |   916 MB |
|   128 |  1775 MB |
|   256 |  3493 MB |
|   512 |  6920 MB |
|  1024 | 13781 MB |

This occurs because **activation memory scales with batch size**.

During training, the forward pass stores intermediate activations so that gradients can be computed during the backward pass. These activations dominate the variable portion of training memory and scale approximately as:

```
Activation Memory ∝ batch_size × sequence_length × hidden_dimension × number_of_layers
```

In this experiment, the model architecture remained fixed, so the only variable factor was **batch size**, which explains the nearly linear growth in memory usage.

---

### 4. Out-of-memory occurs after compute saturation

Even though throughput stopped improving after batch sizes around **256**, increasing batch size continued increasing memory usage. Eventually the GPU ran out of memory:

```
Batch size 1024 → 13.8 GB used
Batch size 1152 → CUDA Out Of Memory
```

This illustrates an important systems tradeoff: **compute efficiency and memory capacity are separate constraints**.

Once the GPU reaches compute saturation, increasing batch size no longer improves throughput but still consumes additional memory. Eventually the required activation memory exceeds the available VRAM and training fails.

---

### 5. Optimal batch size occurs before memory limits

In this experiment, throughput peaked around **batch size 256**. Beyond that point:

- throughput remained constant
- step latency increased
- memory usage continued increasing rapidly

This means batch sizes larger than 256 offered **no performance benefit** but significantly increased memory pressure and the risk of OOM errors.

For a single-GPU training setup, choosing a batch size near the **throughput saturation point** provides the best balance between efficiency and memory safety.

---

## Core Takeaways

- Small batch sizes underutilize GPUs because the workload is too small to fully occupy all streaming multiprocessors.
- Increasing batch size improves throughput until the GPU’s compute resources become saturated.
- After saturation, step time increases proportionally with batch size while throughput remains constant.
- Activation memory grows roughly **linearly with batch size** because forward-pass activations must be stored for backpropagation.
- Once activation memory exceeds available GPU memory, training fails with a **CUDA Out Of Memory error**.
- The optimal batch size for a system is typically near the **compute saturation point**, not the maximum batch size that fits in memory.

---

## Key Insight

Training performance is determined by the interaction between **compute utilization and memory capacity**.

In this experiment, increasing batch size improved throughput until the GPU’s compute units were fully utilized. After that point, larger batches only increased activation memory usage and step latency without improving throughput. Eventually, the memory required to store activations exceeded the available GPU memory and the training run failed with an OOM error.

This highlights a fundamental principle of ML systems engineering:

> Increasing batch size improves GPU utilization until compute becomes saturated, after which memory becomes the dominant limiting resource.
