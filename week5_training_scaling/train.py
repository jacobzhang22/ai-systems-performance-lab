import csv
import time
import torch
import torch.nn as nn

from model import TinyTransformerLM


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VOCAB_SIZE = 5000
SEQ_LEN = 128
WARMUP_STEPS = 5
MEASURE_STEPS = 20
BATCH_SIZES = [8, 16, 32, 64, 128, 256, 512, 1024, 1152, 1280]



def make_batch(batch_size: int):
    x = torch.randint(0, VOCAB_SIZE, (batch_size, SEQ_LEN), device=DEVICE)
    y = torch.randint(0, VOCAB_SIZE, (batch_size, SEQ_LEN), device=DEVICE)
    return x, y


def run_experiment(batch_size: int):
    model = TinyTransformerLM(
        vocab_size=VOCAB_SIZE,
        d_model=256,
        nhead=4,
        num_layers=2,
        dim_feedforward=512,
        max_seq_len=SEQ_LEN,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    for _ in range(WARMUP_STEPS):
        x, y = make_batch(batch_size)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
        loss.backward()
        optimizer.step()

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    step_times = []

    for _ in range(MEASURE_STEPS):
        x, y = make_batch(batch_size)

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
        loss.backward()
        optimizer.step()

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

        step_times.append(end - start)

    avg_step_time = sum(step_times) / len(step_times)
    throughput = batch_size / avg_step_time

    peak_mem_mb = 0.0
    if DEVICE == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    return {
        "batch_size": batch_size,
        "avg_step_time_s": avg_step_time,
        "throughput_samples_per_s": throughput,
        "peak_memory_mb": peak_mem_mb,
    }


def main():
    results = []

    print(f"Running on: {DEVICE}")
    for batch_size in BATCH_SIZES:
        try:
            result = run_experiment(batch_size)
            results.append(result)
            print(result)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"OOM at batch size {batch_size}")
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
                break
            raise

    with open("results.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "batch_size",
                "avg_step_time_s",
                "throughput_samples_per_s",
                "peak_memory_mb",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print("Saved results to results.csv")


if __name__ == "__main__":
    main()
