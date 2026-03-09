import time
import torch
from train import loader, model, optimizer, device, train_step

def sync_if_cuda():
    if device.startswith("cuda"):
        torch.cuda.synchronize()

def main():
    print("=== Week 4 GPU Utilization Benchmark ===")
    print(f"Device: {device}")
    if device.startswith("cuda"):
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Loader settings: batch_size={loader.batch_size}, num_workers={loader.num_workers}")
    print("Tip: In another terminal run: watch -n 0.5 nvidia-smi")
    print()

    warmup_steps = 10
    timed_steps = 50

    print(f"Warmup: {warmup_steps} steps (not timed)")
    it = iter(loader)
    for i in range(warmup_steps):
        x, y = next(it)
        loss_val = train_step(x, y)
        if (i + 1) % 5 == 0:
            print(f"  warmup step {i+1}/{warmup_steps} | loss={loss_val:.4f}")
    sync_if_cuda()
    print("Warmup done.\n")

    print(f"Timed run: {timed_steps} steps")

    data_times = []
    compute_times = []
    full_step_times = []
    losses = []

    it = iter(loader)
    sync_if_cuda()
    run_start = time.perf_counter()

    for step in range(timed_steps):
        # full step starts here
        full_s0 = time.perf_counter()

        # measure dataloader / CPU-side wait
        data_s0 = time.perf_counter()
        x, y = next(it)
        data_s1 = time.perf_counter()
        data_time = data_s1 - data_s0

        # measure train_step compute region
        sync_if_cuda()
        compute_s0 = time.perf_counter()

        loss_val = train_step(x, y)

        sync_if_cuda()
        compute_s1 = time.perf_counter()
        compute_time = compute_s1 - compute_s0

        # full step ends here
        full_s1 = time.perf_counter()
        full_step_time = full_s1 - full_s0

        data_times.append(data_time)
        compute_times.append(compute_time)
        full_step_times.append(full_step_time)
        losses.append(loss_val)

        if (step + 1) % 10 == 0:
            avg_data_last10 = sum(data_times[-10:]) / 10
            avg_compute_last10 = sum(compute_times[-10:]) / 10
            avg_full_last10 = sum(full_step_times[-10:]) / 10
            print(
                f"  step {step+1:>3}/{timed_steps} | "
                f"loss={loss_val:.4f} | "
                f"data={avg_data_last10*1000:.2f} ms | "
                f"compute={avg_compute_last10*1000:.2f} ms | "
                f"full={avg_full_last10*1000:.2f} ms"
            )

    sync_if_cuda()
    run_end = time.perf_counter()

    total = run_end - run_start

    avg_data = sum(data_times) / len(data_times)
    avg_compute = sum(compute_times) / len(compute_times)
    avg_full = sum(full_step_times) / len(full_step_times)

    steps_per_sec = timed_steps / total
    samples_per_sec = steps_per_sec * loader.batch_size

    other_time = avg_full - avg_compute
    cpu_overhead_excl_dataloader = avg_full - avg_data - avg_compute

    print("\n=== Results ===")
    print(f"Total timed duration:          {total:.4f} s")
    print(f"Avg data time:                {avg_data*1000:.2f} ms")
    print(f"Avg compute time:             {avg_compute*1000:.2f} ms")
    print(f"Avg full step time:           {avg_full*1000:.2f} ms")
    print(f"Avg non-compute time:         {other_time*1000:.2f} ms")
    print(f"Avg CPU overhead excl. data:  {cpu_overhead_excl_dataloader*1000:.2f} ms")
    print(f"Steps/sec:                    {steps_per_sec:.2f}")
    print(f"Samples/sec:                  {samples_per_sec:.2f}")
    print(f"Final loss:                   {losses[-1]:.4f}")

if __name__ == "__main__":
    main()