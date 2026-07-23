"""The module.
"""
from typing import Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> list[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> list["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self) -> None:
        self.training = True

    def parameters(self) -> list[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> list["Module"]:
        return _child_modules(self.__dict__)

    def eval(self) -> None:
        self.training = False
        for m in self._children():
            m.training = False

    def train(self) -> None:
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features))
        self.bias = Parameter(ops.reshape(init.kaiming_uniform(out_features, 1), (1, self.out_features))) if bias else None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        out = ops.matmul(X, self.weight)
        if self.bias is not None:
            out = out + ops.broadcast_to(self.bias, out.shape)
        return out
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.reshape(X, (X.shape[0], -1))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        N = logits.shape[0] 
        log_probs = ops.logsoftmax(logits)
        one_hot = init.one_hot(logits.shape[1], y, device=logits.device, dtype=logits.dtype)
        loss = -ops.summation(log_probs * one_hot) / N
        return loss
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype)) # (D,)
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype)) # (D,)
        self.running_mean = init.zeros(dim, device=device, dtype=dtype) # (D,)
        self.running_var = init.ones(dim, device=device, dtype=dtype) # (D,)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            mean = ops.summation(x, axes=(0,)) / x.shape[0] # (D,)
            mean_broadcasted = ops.broadcast_to(ops.reshape(mean, (1, -1)), x.shape) # (N, D)
            var = ops.summation((x - mean_broadcasted) * (x - mean_broadcasted), axes=(0,)) / x.shape[0] # (D,)
            var_broadcasted = ops.broadcast_to(ops.reshape(var, (1, -1)), x.shape) # (N, D)
            x_hat = (x - mean_broadcasted) / ops.power_scalar(var_broadcasted + self.eps, 0.5) # (N, D)
            out = ops.broadcast_to(ops.reshape(self.weight, (1, -1)), x.shape) * x_hat + ops.broadcast_to(ops.reshape(self.bias, (1, -1)), x.shape) # (N, D)

            self.running_mean.data = self.momentum * mean + (1 - self.momentum) * self.running_mean.data
            self.running_var.data = self.momentum * var + (1 - self.momentum) * self.running_var.data
        else:
            running_mean_broadcasted = ops.broadcast_to(ops.reshape(self.running_mean, (1, -1)), x.shape) # (N, D)
            running_var_broadcasted = ops.broadcast_to(ops.reshape(self.running_var, (1, -1)), x.shape) # (N, D)
            x_hat = (x - running_mean_broadcasted) / ops.power_scalar(running_var_broadcasted + self.eps, 0.5) # (N, D)
            out = ops.broadcast_to(ops.reshape(self.weight, (1, -1)), x.shape) * x_hat + ops.broadcast_to(ops.reshape(self.bias, (1, -1)), x.shape) # (N, D)
        return out
        ### END YOUR SOLUTION



class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype)) # (1, D)
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype)) # (1, D)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        mean = ops.summation(x, axes=(1,)) / self.dim # (N,)
        mean = ops.broadcast_to(ops.reshape(mean, (-1, 1)), x.shape) # (N, D)
        var = ops.summation((x - mean) * (x - mean), axes=(1,)) / self.dim # (N,)
        var = ops.broadcast_to(ops.reshape(var, (-1, 1)), x.shape) # (N, D)
        x_hat = (x - mean) / ops.power_scalar(var + self.eps, 0.5) # (N, D)
        out = ops.broadcast_to(ops.reshape(self.weight, (1, -1)), x.shape) * x_hat + ops.broadcast_to(ops.reshape(self.bias, (1, -1)), x.shape) # (N, D)
        return out
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            mask = init.randb(*x.shape, p=1-self.p, device=x.device, dtype=x.dtype) # (N, D)
            out = x * mask / (1 - self.p) # (N, D)
        else:
            out = x
        return out
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x + self.fn(x)
        ### END YOUR SOLUTION
