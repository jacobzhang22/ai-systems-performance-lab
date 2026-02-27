# server.py
import time
import random
from fastapi import FastAPI

app = FastAPI()

def burn_cpu(ms: float) -> None:
    end = time.perf_counter() + ms / 1000.0
    x = 0
    while time.perf_counter() < end:
        x += 1  # prevent optimization

@app.get("/work")
def work(ms: float = 50.0, jitter: float = 0.0):
    extra = random.random() * jitter if jitter > 0 else 0.0
    burn_cpu(ms + extra)
    return {"ok": True, "work_ms": ms + extra}

# ---- profiling hook (ONLY used when you run this file directly) ----
if __name__ == "__main__":
    from line_profiler import LineProfiler

    lp = LineProfiler()
    lp_wrapper = lp(work)

    for _ in range(2000):
        lp_wrapper(ms=50.0, jitter=0.0)

    lp.print_stats()