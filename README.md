```markdown
# QLISA: Quantized Layerwise Importance Sampled Adaptation

This repository implements **QLISA**, a method that combines 4‑bit weight quantization (NF4 with double quantization) with layerwise importance sampling to efficiently fine‑tune large diffusion models.

## Features
- **NF4 Quantization:** Convert full‑precision weights to a simulated 4‑bit representation.
- **Double Quantization:** Quantize scaling factors to reduce memory overhead.
- **Dynamic Dequantization:** Dequantize weights on-the‑fly to high precision (BFloat16) during forward passes.
- **Layerwise Importance Sampling:** Freeze most layers and randomly activate a subset (with the first and last always active) to mimic LoRA’s behavior.
- **Optimizer Hook Integration:** Per‑parameter optimizer and scheduler hooks for efficient updates.

## Repository Structure
```

qlisa/
├── README.md
├── LICENSE
├── setup.py
├── requirements.txt
├── .gitignore
├── qlisa/
│   ├──  **init** .py
│   ├── qlisa_diffusion.py
│   ├── quantization.py
│   └── utils.py
├── examples/
│   ├── run_training.py
│   └── run_inference.py
└── tests/
├──  **init** .py
├── test_quantization.py
└── test_qlisa.py

```

## Installation

Clone the repository and install the package:
```bash
git clone https://github.com/yourusername/qlisa.git
cd qlisa
pip install -r requirements.txt
pip install -e .
```

## Usage

A minimal example:

```python
from qlisa.qlisa_diffusion import QLISADiffusion
import torch
import torch.nn as nn

# Define a simple linear model for demonstration.
class SimpleModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)

model = SimpleModel(10, 2)
qlisa_obj = QLISADiffusion(model)
# Now integrate qlisa_obj.register() and qlisa_obj.insert_hook() into your training loop.
```

## Running Examples

* **Training Example:**
  ```bash
  python examples/run_training.py
  ```
* **Inference Example:**
  ```bash
  python examples/run_inference.py
  ```

## Running Tests

Use pytest to run the tests:

```bash
pytest tests/
```
