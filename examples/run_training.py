"""
Example training script showing how to integrate QLISADiffusion.
This script creates a simple model, wraps it with QLISA, and demonstrates
the registration of optimizers and hooks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from qlisa.qlisa_diffusion import QLISADiffusion

# Dummy scheduler function for demonstration.
def get_scheduler(optimizer, num_warmup_steps=10, num_training_steps=100):
    return optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

# A simple model with a single linear layer.
class SimpleModel(nn.Module):
    def __init__(self, in_features=10, out_features=2):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)

def main():
    model = SimpleModel()
    # Wrap the model with QLISA.
    qlisa_obj = QLISADiffusion(model)

    # For demonstration, use AdamW optimizer.
    optimizer_class = optim.AdamW
    # Register optimizers and schedulers for active parameters.
    qlisa_obj.register(optimizer_class=optimizer_class,
                       get_scheduler=get_scheduler,
                       accelerator=None,
                       optim_kwargs={'lr': 1e-3},
                       sched_kwargs={'num_warmup_steps': 5, 'num_training_steps': 50})

    # Insert the hook so that after gradients are accumulated, steps are performed.
    qlisa_obj.insert_hook(optimizer_class=optimizer_class,
                          get_scheduler=get_scheduler,
                          accelerator=None,
                          optim_kwargs={'lr': 1e-3},
                          sched_kwargs={'num_warmup_steps': 5, 'num_training_steps': 50})

    # Dummy training loop.
    for epoch in range(5):
        x = torch.randn(16, 10)
        y = model(x)
        loss = y.mean()
        loss.backward()
        # Simulate the optimizer hook call.
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                # This hook would be automatically called after gradient accumulation in a real loop.
                pass
        print(f"Epoch {epoch}: Loss {loss.item()}")

if __name__ == "__main__":
    main()
