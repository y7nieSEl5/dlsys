from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        Z_max = array_api.max(Z, axis=1, keepdims=True)
        Z_stable = Z - Z_max
        logsumexp = array_api.log(array_api.sum(array_api.exp(Z_stable), axis=1, keepdims=True))
        return Z_stable - logsumexp
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0] # (N, C)
        Z_data = Z.realize_cached_data() # (N, C)
        Z_max = array_api.max(Z_data, axis=1, keepdims=True) # (N, 1)
        Z_stable = Z_data - Z_max # (N, C)
        logsumexp = array_api.log(array_api.sum(array_api.exp(Z_stable), axis=1, keepdims=True)) # (N, 1)
        softmax = array_api.exp(Z_stable - logsumexp)   # (N, C)

        sum_grad = summation(out_grad, axes=(1,)) # (1, C) -> (N,)

        sum_grad = sum_grad.reshape((-1, 1)) # (N, 1)

        return out_grad - multiply(sum_grad, Tensor(softmax)) # (N, C) - (N, 1) * (N, C)
        ### END YOUR SOLUTION


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)