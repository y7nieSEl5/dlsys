"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        if kernel_size <= 0 or stride <= 0:
            raise ValueError("kernel_size and stride must be positive")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        # The primitive convolution stores kernels as (K, K, Cin, Cout),
        # matching its NHWC input convention.  Initialize with the same
        # uniform bound used by torch's Conv2d for a comparable scale.
        bound = 1 / np.sqrt(in_channels * kernel_size * kernel_size)
        self.weight = Parameter(
            init.rand(
                kernel_size,
                kernel_size,
                in_channels,
                out_channels,
                low=-bound,
                high=bound,
                device=device,
                dtype=dtype,
            )
        )
        self.bias = (
            Parameter(
                init.rand(
                    out_channels,
                    low=-bound,
                    high=bound,
                    device=device,
                    dtype=dtype,
                )
            )
            if bias
            else None
        )
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # Convert NCHW -> NHWC for ops.conv.  Tensor.transpose swaps two
        # axes at a time, so two swaps are needed for a full permutation.
        x_nhwc = x.transpose((1, 2)).transpose((2, 3))
        out = ops.conv(
            x_nhwc,
            self.weight,
            stride=self.stride,
            padding=self.kernel_size // 2,
        )
        if self.bias is not None:
            bias = ops.reshape(self.bias, (1, 1, 1, self.out_channels))
            out = out + ops.broadcast_to(bias, out.shape)

        # Convert NHWC -> NCHW before returning to the module API.
        return out.transpose((1, 3)).transpose((2, 3))
        ### END YOUR SOLUTION
