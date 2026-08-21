import math
from .init_basic import *
from typing import Any


def xavier_uniform(fan_in: int, fan_out: int, gain: float = 1.0, **kwargs: Any) -> "Tensor":
    ### BEGIN YOUR SOLUTION
    bound = gain * math.sqrt(6.0 / (fan_in + fan_out))
    return rand(fan_in, fan_out, low = -bound, high = bound, **kwargs)
    ### END YOUR SOLUTION


def xavier_normal(fan_in: int, fan_out: int, gain: float = 1.0, **kwargs: Any) -> "Tensor":
    ### BEGIN YOUR SOLUTION
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    return randn(fan_in, fan_out, mean = 0.0, std = std, **kwargs)
    ### END YOUR SOLUTION

def kaiming_uniform(
    fan_in: int,
    fan_out: int,
    nonlinearity: str = "relu",
    shape: tuple | None = None,
    **kwargs: Any,
) -> "Tensor":
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    if shape is not None:
        fan_in = shape[0]
        fan_out = shape[1]
    if nonlinearity == "relu":
        gain = math.sqrt(2.0)
    bound = gain * math.sqrt(3.0 / fan_in)
    return rand(fan_in, fan_out, low = -bound, high = bound, **kwargs)
    ### END YOUR SOLUTION



def kaiming_normal(fan_in: int, fan_out: int, nonlinearity: str = "relu", **kwargs: Any) -> "Tensor":
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    if nonlinearity == "relu":
        gain = math.sqrt(2.0)
    std = gain * math.sqrt(1.0 / fan_in)
    return randn(fan_in, fan_out, mean = 0.0, std = std, **kwargs)
    ### END YOUR SOLUTION
