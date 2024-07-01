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

