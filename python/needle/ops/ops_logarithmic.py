from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api

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
        if axes is not None and not isinstance(axes, tuple):
            axes = (axes,)
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        Z_max = array_api.max(Z, axis=self.axes, keepdims=True)
        Z_stable = Z - Z_max
        logsumexp = array_api.log(array_api.sum(array_api.exp(Z_stable), axis=self.axes, keepdims=True))
        return (Z_max + logsumexp).squeeze()
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0].realize_cached_data() # (N, C)
        max_Z = array_api.max(Z, axis=self.axes, keepdims=True) # (N, 1)
        exp = array_api.exp(Z - array_api.broadcast_to(max_Z, Z.shape)) # (N, C)
        sum_exp = array_api.sum(exp, axis=self.axes, keepdims=True) # (N, 1)
        softmax = Tensor(exp / array_api.broadcast_to(sum_exp, exp.shape)) # (N, C)

        if self.axes is not None:
            shape = list(Z.shape)
            for axis in self.axes:
                shape[axis] = 1
            out_grad = out_grad.reshape(shape)
        return out_grad.broadcast_to(Z.shape) * softmax # (N, C)
        ### END YOUR SOLUTION


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)
