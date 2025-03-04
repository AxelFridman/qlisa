import torch
from qlisa.quantization import quantize_weight, dequantize_weight

def test_quantization_dequantization():
    weight = torch.randn(16, 16)
    weight_q, scale_fp32, scale_q = quantize_weight(weight)
    dequant_weight = dequantize_weight(weight_q, scale_fp32, scale_q)
    # The dequantized weight should be close to the original scaled weight.
    # We use a loose tolerance as this is a simulation.
    assert torch.allclose(weight, dequant_weight.float(), atol=1e-1)
