import torch
import torch.nn as nn
from qlisa.qlisa_diffusion import QLISADiffusion

class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.linear1 = nn.Linear(10, 5)
        self.linear2 = nn.Linear(5, 2)

    def forward(self, x):
        x = self.linear1(x)
        return self.linear2(x)

def test_qlisa_activation():
    model = DummyModel()
    qlisa_obj = QLISADiffusion(model)
    # Check that the first and last parameters are active.
    params = list(model.parameters())
    assert params[0].requires_grad == True
    assert params[-1].requires_grad == True
    # Check that not all intermediate parameters are active.
    mid_active = any(p.requires_grad for p in params[1:-1])
    assert mid_active

def test_forward_pass():
    model = DummyModel()
    qlisa_obj = QLISADiffusion(model)
    x = torch.randn(4, 10)
    # Run a forward pass; the monkey-patched forward should be used.
    y = model(x)
    assert y.shape == (4, 2)
