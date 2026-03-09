import torch
from torch.profiler import profile, ProfilerActivity
from train import loader, model, optimizer, device, train_step

def sync_if_cuda():
    if device.startswith("cuda"):
        torch.cuda.synchronize()

def main():
    print("=== Week 4 Profiler Run ===")
    print(f"Device: {device}")
    print(f"Loader settings: batch_size={loader.batch_size}, num_workers={loader.num_workers}")
    print()

    it = iter(loader)

    # warmup a few steps first
    warmup_steps = 5
    for _ in range(warmup_steps):
        x, y = next(it)
        train_step(x, y)
    sync_if_cuda()

    # profile a small number of steps
    profile_steps = 10

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False
    ) as prof:
        for step in range(profile_steps):
            x, y = next(it)
            train_step(x, y)

    sync_if_cuda()

    print("\n=== Top operators by CUDA time ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

    print("\n=== Top operators by CPU time ===")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))

if __name__ == "__main__":
    main()