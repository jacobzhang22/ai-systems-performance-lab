# benchmark.py

import time
from workloads import count_primes
from workloads import count_primes_sqrt

def benchmark(func, n: int, runs: int = 3):
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        func(n)
        end = time.perf_counter()
        times.append(end - start)

    avg_time = sum(times) / len(times)
    #print("test ", times)
    return avg_time


if __name__ == "__main__":
    sizes = [500, 1000, 2000, 4000]
    differences = []
    #sizes = [4000]
    for size in sizes:
        avg = benchmark(count_primes, size)
        print(f"Input: {size:5d} | Avg Time: {avg:.4f} seconds")
        avg_sqrt = benchmark(count_primes_sqrt, size)
        print(f"Input: {size:5d} | Avg Time (sqrt): {avg_sqrt:.4f} seconds")
        differences.append(avg / avg_sqrt)
    assert count_primes(4000) == count_primes_sqrt(4000)
    print(f"Differences: {differences}")
    


"""
notes
i expect the new complexity to be O(n * sqrt(n)) instead of O(n^2). this means instead of seeing a 4x increase in time 
when we double, we should see when we double the size of the input a 2*root(2) increase in time instead of a 4x increase.

"""