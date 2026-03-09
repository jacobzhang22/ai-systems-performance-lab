import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import SlowDataset

# ---- Device ----
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Data ----
dataset = SlowDataset(size=10_000, sleep_s=0.01)

loader = DataLoader(
    dataset,
    batch_size=64,
    num_workers=0,       # vary this later (0 -> 4)
    pin_memory=True,     # helps host->device copies when using CUDA
    drop_last=True       # keeps batch sizes consistent
)

# ---- Model ----
model = nn.Sequential(
    nn.Linear(1024, 2048),
    nn.ReLU(),
    nn.Linear(2048, 1)
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train_step(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    Runs one training step and returns loss value (float).
    """
    model.train()

    # Move to device (non_blocking works only if pin_memory=True AND tensor is in pinned memory)
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)

    pred = model(x)
    loss = ((pred - y.unsqueeze(1)) ** 2).mean()

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    return float(loss.detach().item())