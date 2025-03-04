"""
Example inference script using a QLISA-wrapped model.
This demonstrates that on-the-fly dequantization is applied during forward passes.
"""

import torch
import torch.nn as nn
from qlisa.qlisa_diffusion import QLISADiffusion

class SimpleModel(nn.Module):
    def __init__(self, in_features=10, out_features=2):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)

def main():
    model = SimpleModel()
    qlisa_obj = QLISADiffusion(model)

    # In inference, we simply forward pass.
    x = torch.randn(4, 10)
    y = model(x)
    print("Inference output:", y)

if __name__ == "__main__":
    main()
