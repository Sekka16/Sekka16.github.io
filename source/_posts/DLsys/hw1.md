---
title: DLsys-hw1
date: 2024-04-17 22:26:56
tags:
categories:
- [学习笔记, DLsys]
description: DLsys第1次作业
---

# 作业1

## 1 forward computation

题目要求在`ops_mathematic.py`实现以下功能，注意到导入的`array_api`其实是`numpy`，所以实现起来就很简单了

- `PowerScalar`: raise input to an integer (scalar) power
- `EWiseDiv`: true division of the inputs, element-wise (2 inputs)
- `DivScalar`: true division of the input by a scalar, element-wise (1 input, scalar - number)
- `MatMul`: matrix multiplication of the inputs (2 inputs)
- `Summation`: sum of array elements over given axes (1 input, axes - tuple)
- `BroadcastTo`: broadcast an array to a new shape (1 input, shape - tuple)
- `Reshape`: gives a new shape to an array without changing its data (1 input, shape - tuple)
- `Negate`: numerical negative, element-wise (1 input)
- `Transpose`: reverses the order of two axes (axis1, axis2), defaults to the last two axes (1 input, axes - tuple)

唯一值得注意的是最后一个`Transpose`的实现，它的实现涉及到`axes`参数，在`numpy`中对`axes`参数说明如下：

>axes: tuple or list of ints, optional
>If specified, it must be a tuple or list which contains a permutation of [0,1,…,N-1] where N is the number of axes of a. The i’th axis of the returned array will correspond to the axis numbered axes[i] of the input. If not specified, defaults to range(a.ndim)[::-1], which reverses the order of the axes.

简而言之，axes[i] = n 中代表结果的第i维度是原张量数组中的第n维，举例如下：

axes = [1, 0, 2]

i = 0, axes[0] = 1, 则结果的第0维是原来的第1维
i = 1, axes[1] = 0, 则结果的第1维其实是的第0维

```python
a = np.ones((1, 2, 3))
np.transpose(a, (1, 0, 2)).shape
(2, 1, 3)
```

**而正因为这些对本次实验造成了一定的困扰**，本次实验实现的`transpose`仅仅能实现两个维度的交换，但是却又要用到`numpy.tranpose()`，如前文所言两者对于`axes`参数的定义其实是不同的，因此我们需要将`axes`做一定的转化。

```python
class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        dim = list(range(len(a.shape)))
        if self.axes is None:
            dim[-2], dim[-1] = dim[-1], dim[-2]
        else:
            dim[self.axes[0]], dim[self.axes[1]] = dim[self.axes[1]], dim[self.axes[0]]
        return array_api.transpose(a, dim)
        ### END YOUR SOLUTION
```

## backward passes

同样是在`ops_mathematic.py`中实现反向的梯度计算，但是这次实现的部分功能就很抽象了

### transpose

假设`x`是一个`n * 1`的行向量，对于`y = x.T`这样的表达式，`y`关于`x`的导数是什么？换句话说，一个`n * 1`的行向量对于与一个`1 * n`的列向量求导的结果是什么？

答案是一个`n * n`的矩阵也可以是`n * 1 * 1 * n`，称为**雅可比矩阵**，在`y = x.T`的前提下是一个主对角线全为1的单位阵。如果`x`和`y`是矩阵也没有关系，可以将他们进行向量化之后在进行相同的操作。

