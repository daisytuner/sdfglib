import torch
import torch.nn as nn
from python import optimize_model

class IdentityNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x

model = IdentityNet()
example_input = torch.randn(2, 1)
optimize_model(model, example_input)
