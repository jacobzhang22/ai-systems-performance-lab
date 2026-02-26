# async_demo.py
import asyncio
import time

async def io_task(delay: float):
    await asyncio.sleep(delay)
    return 1

async def run(delay=0.005, n=10_000, limit=1_000):
    sem = asyncio.Semaphore(limit)

    async def wrapped():
        async with sem:
            return await io_task(delay)

    t0 = time.perf_counter()
    results = await asyncio.gather(*(wrapped() for _ in range(n)))
    dt = time.perf_counter() - t0
    print(f"async n={n} limit={limit} delay={delay}: {dt:.4f}s, results={sum(results)}")

if __name__ == "__main__":
    asyncio.run(run())