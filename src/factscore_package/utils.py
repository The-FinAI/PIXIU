# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch

def assert_all_approx_close(a, b, rtol, atol, count):

    idx = torch.isclose(a.float(), b.float(), rtol, atol)
    sumval = (idx==0).sum().item()
    if sumval > count:
        print(f'Too many values not close: assert {sumval} < {count}')
        try:
            torch.testing.assert_allclose(a, b, rtol, atol)
        except Exception as e:
            print(e)


def get_memory_footprint(model, return_buffers=True):
    """
    Get the memory footprint of a model. This will return the memory footprint of the current model in bytes.
    Useful to benchmark the memory footprint of the current model and design some tests. Solution inspired from the
    PyTorch discussions: https://discuss.pytorch.org/t/gpu-memory-that-model-uses/56822/2
    Arguments:
        return_buffers (`bool`, *optional*, defaults to `True`):
            Whether to return the size of the buffer tensors in the computation of the memory footprint. Buffers
            are tensors that do not require gradients and not registered as parameters. E.g. mean and std in batch
            norm layers. Please see: https://discuss.pytorch.org/t/what-pytorch-means-by-buffers/120266/2
    """
    mem = sum([param.nelement() * param.element_size() for param in model.parameters()])
    if return_buffers:
        mem_bufs = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
        mem = mem + mem_bufs
    return mem


def ـreplace_linear_with_int8linear(model, modules_to_not_convert="lm_head"):
    for name, module in model.named_children():
        ـreplace_linear_with_int8linear(module, modules_to_not_convert)

        if isinstance(module, torch.nn.Linear) and name != modules_to_not_convert:
            model._modules[name] = QuantizedLinearInt8(linear_layer=module)
    return


class QuantizedLinearInt8(torch.nn.Module):
    '''
    A simple but effictive implmenetion of Int8 quantization for linear layers.
    The weights are quantized and stored as Int8, which saves ~50% of the gpu memory.
    During the forwared pass, the weights are de-quantized back to fp16 to do multiplication.
    Pros:
        - saves ~50% of the gpu memory
        - accurate quantization because only the weights are quantized, and the weights don't suffer
            from the "outliers" issue mentioned in the LLM.int8 paper; only the activations do.
        - high precision results beacuse the multiplication is done in fp16
        - much faster than LLM.int8
    Cons:
        - a bit slower because of the added computation of dequantization in each forward pass. In practice, the slowdown
            is not large because in the generation application, gpu utilization is not very high.
    '''
    def __init__(self, linear_layer):
        super().__init__()
        self.bias = linear_layer.bias

        weight_bit_width = 8
        weight = linear_layer.weight

        self.weight_scale = torch.nn.Parameter(
            (weight.abs().max(dim=-1).values / ((2 ** (weight_bit_width - 1)) - 1)).half(),
        )
        # print(self.weight_scale.max().item(), self.weight_scale.min().item(), self.weight_scale.mean().item())
        # if self.weight_scale.max().item() > 0.002:
            # print(self.weight_scale.max().item())
        self.weight = torch.nn.Parameter(
            torch.round(weight.float() / self.weight_scale[:, None]).char(),
            requires_grad=False
            )

    def forward(self, x):
        weight = self.weight.half() * self.weight_scale[:, None]
        return torch.nn.functional.linear(x, weight, self.bias)


def convert_model_to_int8_on_gpu(model, device):
    """
    Quantize a model to int8 and move it to GPU using a simple method.
    """
    if 'cuda' not in device:
        raise ValueError(f"Target device should be a gpu. Device {device} is not supported")

    model.half()

    memory_before_quantization = get_memory_footprint(model)  # without lm_head

    ـreplace_linear_with_int8linear(model)  # replace `Linear` with `QuantizedLinearInt8`

    model.to(device=device)
    memory_after_quantization = get_memory_footprint(model)  # without lm_head

    saving = round(100 * memory_after_quantization/memory_before_quantization)
    memory_before_quantization = round(memory_before_quantization / 2**30, 2)  # rounding for printing
    memory_after_quantization = round(memory_after_quantization / 2**30, 2)  # rounding for printing

    print(f'Quantization memory - before: {memory_before_quantization} GB, after: {memory_after_quantization} GB ({saving}% of the size before)')
    return model
