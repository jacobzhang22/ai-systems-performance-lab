# Week 1 – CPU Bottleneck Analysis

## Hypothesis

The naive prime-counting implementation is CPU-bound due to nested iteration and repeated modulo operations.  
Because the algorithm checks divisibility from `2` to `num` for each candidate number, I expected quadratic scaling behavior (O(n²)) and predicted that the modulo operation inside the inner loop would dominate runtime.

---

## Measurement Strategy

I benchmarked the baseline implementation using `time.perf_counter()` and averaged execution time across three runs per input size.

Input sizes doubled across:

{500, 1000, 2000, 4000}

Empirical results showed approximately 4× runtime increase when doubling input size, consistent with quadratic scaling behavior.

This confirmed the expected O(n²) time complexity in practice.

---

## Profiling

To identify the bottleneck:

### Function-level profiling

```bash
python -m cProfile -s tottime benchmark.py
```

This showed that nearly all execution time was spent inside `count_primes`.

### Line-level profiling

```bash
kernprof -l workloads.py
python -m line_profiler workloads.py.lprof
```

Line-level results showed:

- ~60% of total runtime spent in the modulo operation:
  `if num % i == 0`
- ~39% spent in inner loop iteration overhead

This confirmed the performance bottleneck was repeated divisibility checks inside the inner loop.

---

## Optimization

The optimization reduces divisor checks from:

```python
for i in range(2, num):
```

to:

```python
for i in range(2, int(num**0.5) + 1):
```

This leverages the mathematical property that factors occur in symmetric pairs.  
If a number has a divisor larger than √n, the complementary divisor must be smaller than √n, meaning checks beyond √n are redundant.

This reduces per-number work from O(n) to O(√n), changing total complexity from:

O(n²) → O(n√n)

---

## Results

Across n ∈ {500, 1000, 2000, 4000}:

- Speedup ranged from **3.7× to 20.2×**
- Largest observed speedup at n=4000: **20.2× faster**

Observed scaling also shifted:

- Baseline: ~4× increase when doubling input (quadratic)
- Optimized: ~2.8× increase when doubling input (≈ n^(3/2))

This confirms the theoretical complexity reduction translated into measurable real-world performance improvement.

---

## Conclusion

This experiment demonstrates a structured performance engineering workflow:

1. Form hypothesis about bottleneck
2. Measure scaling behavior
3. Profile at function and line level
4. Apply mathematically grounded optimization
5. Quantify improvement

The workload was CPU-bound, dominated by repeated arithmetic operations in Python, and significantly improved by reducing unnecessary work through algorithmic optimization.
