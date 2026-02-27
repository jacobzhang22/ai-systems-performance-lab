# Week 3 — Latency, Throughput, and Tail Behavior

## Overview

In this experiment, I implemented a CPU-bound FastAPI server with configurable service time (~50ms per request) and optional jitter (0–20ms). I evaluated how latency and throughput behave under varying client-side concurrency using a synthetic load generator.

With:

- **1 worker** and ~50ms service time → theoretical capacity ≈ 20 requests/second
- **4 workers** → theoretical capacity ≈ 80 requests/second

I varied client concurrency while keeping total request count constant and measured:

- Throughput
- P50 latency
- Tail latency (P99 / max)
- Rejection rate

The goal was to understand the tradeoff between latency and throughput, observe queueing behavior under load, and study tail amplification effects.

---

## Experimental Setup

- CPU-bound endpoint performing ~50ms of work
- Configurable jitter (0–20ms)
- Synthetic load generator with configurable concurrency (`c`)
- Fixed total request count per run
- Experiments with:
  - Increasing concurrency
  - Jitter vs no jitter
  - With and without server-side backpressure (in-flight limits)

---

## Key Observations

### 1. Throughput vs Concurrency

Throughput increased as concurrency increased — until the system reached service capacity.

After saturation:

- Throughput plateaued
- Latency increased sharply

This demonstrates the **utilization cliff**:

As utilization approaches 100%, response time grows nonlinearly.

Even though compute time remained constant (~50ms), observed latency grew dramatically due to queueing delay.

---

### 2. Queueing and Tail Amplification

When the server accepted all incoming requests, excess traffic accumulated in a queue.

As concurrency increased:

- P99 latency degraded first at moderate load.
- Near saturation, the entire latency distribution shifted right.
- Waiting time became significantly larger than service time.

At high utilization: Total latency = Service time + Waiting time

Under heavy load, **waiting time dominated service time**.

---

### 3. Variance Amplifies Tail Under Load

Introducing small jitter (±20ms) had minimal effect at low concurrency.

However, near saturation:

- The entire latency distribution shifted.
- Tail behavior worsened.
- Variance compounded queueing delay.

Small differences in service time caused downstream requests to wait longer, amplifying latency due to the queue.

This demonstrates that even minor variance can meaningfully impact tail behavior when utilization is high.

---

### 4. Backpressure via In-Flight Limits

I introduced a server-side semaphore to limit the number of concurrent in-flight requests.

Requests exceeding the limit were rejected immediately rather than queued.

This change:

- Prevented unbounded queue growth
- Stabilized tail latency
- Preserved predictable latency behavior

Successful throughput remained bounded by service capacity, while rejection rate increased under overload.

This illustrates the fundamental tradeoff:

- Accept everything → high tail latency, unstable system
- Reject early → stable latency, reduced availability

---

## Core Takeaways

- Throughput increases with concurrency until saturation.
- Beyond saturation, latency grows rapidly while throughput flattens.
- Tail latency degrades before averages.
- Waiting time dominates compute time near high utilization.
- Small variance amplifies tail latency under load.
- Backpressure protects latency at the cost of availability.

---

## Key Insight

A system with 50ms of compute time can show 1200ms latency because most of that time is spent waiting in a queue, not executing work.

When utilization approaches 100%, queueing delay dominates service time.
