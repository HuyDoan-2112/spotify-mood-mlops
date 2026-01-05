import torch.nn as nn

def build_mlp(input_dim: int, num_classes: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.25),
        nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.20),
        nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.15),
        nn.Linear(32, num_classes),
    )