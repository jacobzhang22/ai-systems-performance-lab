# load_test.py
import asyncio
import time
import statistics
import aiohttp

async def one_request(session, url):
    t0 = time.perf_counter()
    async with session.get(url) as resp:
        try:
            data = await resp.json()
        except:
            data = {"ok": False, "rejected": True}
    latency = (time.perf_counter() - t0) * 1000.0
    return latency, data

async def run(url: str, total_requests: int, concurrency: int):
    sem = asyncio.Semaphore(concurrency)
    latencies = []
    rejected = 0

    async with aiohttp.ClientSession() as session:
        async def worker():
            nonlocal rejected
            async with sem:
                lat, data = await one_request(session, url)
                latencies.append(lat)
                if not data.get("ok", True):
                    rejected += 1

        tasks = [asyncio.create_task(worker()) for _ in range(total_requests)]
        await asyncio.gather(*tasks)

    return latencies, rejected

if __name__ == "__main__":
    import argparse, json
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--c", type=int, default=10)
    args = p.parse_args()

    t0 = time.perf_counter()
    lats, rejected = asyncio.run(run(args.url, args.n, args.c))
    wall = time.perf_counter() - t0

    print(f"n={args.n} c={args.c} wall={wall:.3f}s throughput={args.n/wall:.1f} req/s")
    print(f"mean={statistics.mean(lats):.1f}ms  p50={statistics.median(lats):.1f}ms")
    print(f"max={max(lats):.1f}ms")
    print(f"rejected={rejected} ({100*rejected/args.n:.1f}%)")

    # dump raw
    print(json.dumps({"n": args.n, "c": args.c, "wall_s": wall, "latencies_ms": lats}))