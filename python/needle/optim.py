"""Optimization module"""
import needle as ndl
import numpy as np

from . import data


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for p in self.params:
            if p.grad is None:
                continue
            if p not in self.u:
                self.u[p] = ndl.init.zeros(
                    *p.shape,
                    device=p.device,
                    dtype=p.dtype,
                )
            grad_eff = p.grad + self.weight_decay * p.data
            self.u[p] = self.momentum * self.u[p] + (1 - self.momentum) * grad_eff
            p.data -= self.lr * self.u[p]
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        Note: This does not need to be implemented for HW2 and can be skipped.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for p in self.params:
            if p.grad is None:
                continue
            if p not in self.m:
                self.m[p] = ndl.init.zeros(
                    *p.shape,
                    device=p.device,
                    dtype=p.dtype,
                )
                self.v[p] = ndl.init.zeros(
                    *p.shape,
                    device=p.device,
                    dtype=p.dtype,
                )

            grad = p.grad.detach()
            data = p.data.detach()

            if self.weight_decay != 0:
                grad = grad + self.weight_decay * data

            self.m[p].data = self.beta1 * self.m[p].data + (1 - self.beta1) * grad.data
            self.v[p].data = self.beta2 * self.v[p].data + (1 - self.beta2) * (grad.data ** 2)

            m_hat = self.m[p].data / (1 - self.beta1 ** self.t)
            v_hat = self.v[p].data / (1 - self.beta2 ** self.t)

            new_data = data - self.lr * m_hat / (ndl.ops.power_scalar(v_hat, 0.5) + self.eps)
            p.data = new_data
        ### END YOUR SOLUTION
