import torch

def quantize_weight(weight):
    """
    Simulate 4-bit NF4 quantization:
      - Compute scaling factor (using std of weights).
      - Normalize, clip to [-1, 1] and map to 16 levels.
      - Double quantize the scaling factor to 8 bits.
    :param weight: FP32 tensor.
    :return: (weight_q, scale_fp32, scale_q)
    """
    # Compute scaling factor
    scale_fp32 = weight.std()
    w_normalized = weight / (scale_fp32 + 1e-8)
    w_clipped = torch.clamp(w_normalized, -1, 1)
    # Map from [-1,1] to 16 levels (0 to 15)
    w_mapped = ((w_clipped + 1) / 2 * 15).round().to(torch.int8)
    weight_q = w_mapped

    # Simulate 8-bit quantization for the scale.
    scale_q = torch.clamp((scale_fp32 * 127).round(), -128, 127).to(torch.int8)
    return weight_q, scale_fp32, scale_q

def dequantize_weight(weight_q, scale_fp32, scale_q):
    """
    Dequantize the weight on-the-fly:
      Reverse the mapping from the 4-bit quantized representation to a float.
    :param weight_q: The quantized weight tensor (int8).
    :param scale_fp32: The full precision scaling factor.
    :param scale_q: The 8-bit quantized scale (not used in this simple simulation).
    :return: Dequantized weight tensor in BFloat16.
    """
    w_float = (weight_q.float() / 15 * 2 - 1) * scale_fp32
    return w_float.to(torch.bfloat16)
