import time
import torch
from torch.utils.data import Dataset

class SlowDataset(Dataset):
    def __init__(self, size: int = 10_000, sleep_s: float = 0.01):
        self.size = size
        self.sleep_s = sleep_s

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Simulate CPU preprocessing / tokenization / augmentation
        time.sleep(self.sleep_s)

        x = torch.randn(1024)
        y = torch.sum(x)
        return x, y