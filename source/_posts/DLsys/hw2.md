---
title: DLsys-hw2
date: 2024-06-27 10:32:56
tags:
categories:
- [学习笔记, DLsys]
description: DLsys第2次作业
---

# 作业二

## Question2

### BatchNorm1d

Batch Normalization (BatchNorm) 在训练和推理过程中使用的统计量不同。
- 在训练过程中，BatchNorm使用当前批次的均值和方差来标准化数据，
- 在推理过程中，它使用训练过程中累积的全局均值和方差。

具体来说：

在训练过程中，模型不断更新权重和偏置，数据的分布可能会发生变化。为了使模型更快地收敛和更稳定，BatchNorm在训练过程中对每一个批次的数据计算其均值和方差，然后用这些值来标准化当前批次的数据。

在推理过程中，模型的权重和偏置已经训练好了，数据的分布也应该稳定下来。此时，为了确保模型的一致性和稳定性，BatchNorm使用在训练过程中累积的全局均值和方差来标准化数据。

```python
class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device))
        self.bias = Parameter(init.zeros(dim, device=device))
        self.running_mean = Parameter(init.zeros(dim, device=device))
        self.running_var = Parameter(init.ones(dim, device=device))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size, features = x.shape[0], x.shape[1]
        broadcast_weight = ops.broadcast_to(ops.reshape(self.weight, (1, -1)), x.shape)
        broadcast_bias = ops.broadcast_to(ops.reshape(self.bias, (1, -1)), x.shape)

        if self.training:
            mean_x = ops.divide_scalar(ops.summation(x, axes=0), batch_size)
            broadcast_mean = ops.broadcast_to(ops.reshape(mean_x, (1,-1)), x.shape)

            numerator = x - broadcast_mean

            var_x = ops.power_scalar(numerator, 2)
            var_x = ops.summation(ops.divide_scalar(var_x, batch_size), axes=0) # 这里先累加和先处以batch_size是一样的
            broadcast_var = ops.broadcast_to(ops.reshape(var_x, (1,-1)), x.shape)

            denominator = ops.power_scalar(broadcast_var + self.eps, 0.5)

            frac = numerator / denominator

            y = ops.multiply(broadcast_weight, frac) + broadcast_bias

            # update running estimates
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean_x
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var_x
        else:
            broadcast_rm = ops.broadcast_to(ops.reshape(self.running_mean, (1, -1)), x.shape)
            broadcast_rv = ops.broadcast_to(ops.reshape(self.running_var, (1, -1)), x.shape)

            numerator = x - broadcast_rm

            denominator = ops.power_scalar(broadcast_rv + self.eps, 0.5)

            frac = numerator / denominator

            y = ops.multiply(broadcast_weight, frac) + broadcast_bias

        return y
        ### END YOUR SOLUTION
```

## Question3

### SGD
```python
needle.optim.SGD(params, lr=0.01, momentum=0.0, weight_decay=0.0)
```
开始前先做一些说明：
- `params`: 类型为`list`，每一个元素的元素类型是`Parameter`，也即是`Tensor`。
- `self.u`: 类型为`dict`，每个参数遇到一个对应的**动量值**，参数本身是个键。
- `self.momentum`: 这个其实不是动量，而是动量因子，决定了之前动量和当前梯度在更新中的权重。
- `weight_decay`: 这个其实是损失函数的L2正则项系数的两倍。

实现如下：

```python
class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for param in self.params:
            if param.grad == None:
              continue
            grad_data = ndl.Tensor(param.grad.data + self.weight_decay * param.data, dtype=param.dtype)
            if param not in self.u:
              self.u[param] = 0
            self.u[param] = self.momentum * self.u[param] + (1 - self.momentum) * grad_data 
            param.data = param.data - self.lr * self.u[param]
        ### END YOUR SOLUTION
```

### Adam

这个优化方法没有细看，照着公式将算法实现了一遍，最后一个测例进行内存检查时一直通过不了，感觉这门课最大的问题就在这里了，无法调试，无能为力了

```python
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
        for param in self.params:
            if param.grad is None:
              continue
            grad_data = ndl.Tensor(param.grad.data + self.weight_decay * param.data, dtype=param.dtype)
            if param not in self.m:
                self.m[param] = 0
            if param not in self.v:
                self.v[param] = 0

            self.m[param] = self.beta1 * self.m[param] + (1 - self.beta1) * grad_data
            self.v[param] = self.beta2 * self.v[param] + (1 - self.beta2) * (grad_data ** 2)
            param.data -= ndl.Tensor(self.lr * (self.m[param] / (1 - self.beta1**self.t)) / ((self.v[param] / (1 - self.beta2**self.t))**0.5 + self.eps)).data
        ### END YOUR SOLUTION
```







