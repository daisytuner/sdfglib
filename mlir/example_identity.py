import torch
import torch.nn as nn

from docc.ai import import_from_pytorch


class IdentityNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x


model = IdentityNet()
example_input = torch.randn(2, 1)
result = import_from_pytorch(model, example_input)
print(result)
