# workloads.py
from line_profiler import profile

@profile
def count_primes(n: int) -> int:
    count = 0
    for num in range(2, n):
        is_prime = True
        for i in range(2, num):
        
            if num % i == 0:
                is_prime = False
                break
        if is_prime:
            count += 1
    return count


def count_primes_sqrt(n: int) -> int:
    """
    Same logic but only checks divisors up to sqrt(num).
    """
    count = 0
    for num in range(2, n):
        is_prime = True
        limit = int(num ** 0.5) + 1
        for i in range(2, limit):
            if num % i == 0:
                is_prime = False
                break
        if is_prime:
            count += 1
    return count

if __name__ == "__main__":
    count_primes(4000)
    #count_primes_sqrt(4000)