"""Optimization module"""
import needle as ndl
import numpy as np

from python.needle import data


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
                self.u[p] = ndl.init.zeros(*p.shape)
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
                self.m[p] = ndl.init.zeros(*p.shape)
                self.v[p] = ndl.init.zeros(*p.shape)

            grad = p.grad.numpy()
            data = p.data.numpy()
            m = self.m[p].numpy()
            v = self.v[p].numpy()

            grad_eff = grad + self.weight_decay * data

            m_new = self.beta1 * m + (1 - self.beta1) * grad_eff
            v_new = self.beta2 * v + (1 - self.beta2) * (grad_eff ** 2)

            m_hat = m_new / (1 - self.beta1 ** self.t)
            v_hat = v_new / (1 - self.beta2 ** self.t)

            data_new = data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

            self.m[p] = ndl.Tensor(m_new, device=p.device, dtype=p.dtype)
            self.v[p] = ndl.Tensor(v_new, device=p.device, dtype=p.dtype)
            p.data = ndl.Tensor(data_new, device=p.device, dtype=p.dtype)
        ### END YOUR SOLUTION
