---
title: Needle框架解读
date: 2024-05-09 22:46:20
tags: 
- DLsys
categories:
description:
---

# Needle框架解读

## NDArray后端

### 1 项目文件结构

Python:

- needle/backend_ndarray/ndarray.py
- needle/backend_ndarray/ndarray_backend_numpy.py

C++/CUDA

- src/ndarray_backend_cpu.cc
- src/ndarray_backend_cuda.cu

### 2 关键数据结构

- NDArray: the container to hold device specific ndarray
- BackendDevice: backend device
  - mod holds the module implementation that implements all functions
  - checkout ndarray_backend_numpy.py for a python-side reference.

### 3 追踪加法执行过程

在执行张量加法的过程中发生了什么？

```python
x = nd.NDArray([1, 2, 3], device=nd.cuda())
y = x + 1
```

路径如下：

- `NDArray.__add__`

在`__add__`中跳转到`ewise_or_scalar`方法中，确定执行张量加法还是标量加法

并且传入了`self.device.ewise_add`, `self.device.scalar_add`参数

```python
def __add__(self, other):
    return self.ewise_or_scalar(
        other, self.device.ewise_add, self.device.scalar_add
    )
```

- `NDArray.ewise_or_scalar`

根据传入参数`other`的类型选择要执行的函数

```python
def ewise_or_scalar(self, other, ewise_func, scalar_func):
        """Run either an elementwise or scalar version of a function,
        depending on whether "other" is an NDArray or scalar
        """
        out = NDArray.make(self.shape, device=self.device)
        if isinstance(other, NDArray):
            assert self.shape == other.shape, "operation needs two equal-sized arrays"
            ewise_func(self.compact()._handle, other.compact()._handle, out._handle)
        else:
            scalar_func(self.compact()._handle, other, out._handle)
        return out
```

- `ndarray_backend_cpu.cc:ScalarAdd`

首先看文件中的一段代码：

```c++
PYBIND11_MODULE(ndarray_backend_cpu, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cpu;

  m.attr("__device_name__") = "cpu";
  m.attr("__tile_size__") = TILE;

  py::class_<AlignedArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def("ptr", &AlignedArray::ptr_as_int)
      .def_readonly("size", &AlignedArray::size);

  // return numpy array (with copying for simplicity, otherwise garbage
  // collection is a pain)
  m.def("to_numpy", [](const AlignedArray& a, std::vector<size_t> shape,
                       std::vector<size_t> strides, size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });
    return py::array_t<scalar_t>(shape, numpy_strides, a.ptr + offset);
  });

  // convert from numpy (with copying)
  m.def("from_numpy", [](py::array_t<scalar_t> a, AlignedArray* out) {
    std::memcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE);
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);
}
```

这段代码是使用 Pybind11 库将 C++ 的函数和类绑定到 Python 中。主要实现了以下功能：

1. 将 C++ 中的 `AlignedArray` 类绑定为 Python 中的 `Array` 类，使其可以在 Python 中使用。这个类似于在 Python 中创建了一个新的类，可以使用 C++ 中的功能来操作它。
2. 定义了两个函数 `to_numpy` 和 `from_numpy`，用于将 C++ 的 `AlignedArray` 对象转换为 NumPy 数组，并将 NumPy 数组转换为 `AlignedArray` 对象。这使得可以在 C++ 和 Python 之间方便地进行数据传递和转换。
3. 定义了其他一些函数，例如 `fill`、`compact`、`ewise_setitem` 等，这些函数提供了一些常用的数组操作功能，可以在 Python 中直接调用。

**这段代码的最终目的是创建一个名为 `ndarray_backend_cpu` 的 Python 模块，其中包含了一些与数组操作相关的 C++ 函数和类的绑定，使得可以在 Python 中使用这些功能。**

然后是朴实无华的`ScalarAdd`实现

```c++
void ScalarAdd(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of corresponding entry in a plus the scalar val.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + val;
  }
}
```

### 4 NDArray的device属性

介绍完上述内容后，还有一个小小的问题，以下代码的调用过程是什么？

```python
self.device.scalar_add
```

`NDArray.device`：注意这是个属性，访问的时候使用`self.device`即可

```python
@property
def device(self):
    return self._device
```

那么`device` 是如何与不同的后端进行绑定的呢？我们继续看`NDArray`的初始化过程如下

```python
    def __init__(self, other, device=None):
        """Create by copying another NDArray, or from numpy"""
        if isinstance(other, NDArray):
            # create a copy of existing NDArray
            if device is None:
                device = other.device
            self._init(other.to(device) + 0.0)  # this creates a copy
        elif isinstance(other, np.ndarray):
            # create copy from numpy array
            device = device if device is not None else default_device()
            array = self.make(other.shape, device=device)
            array.device.from_numpy(np.ascontiguousarray(other), array._handle)
            self._init(array)
        else:
            # see if we can create a numpy array from input
            array = NDArray(np.array(other), device=device)
            self._init(array)

    def _init(self, other):
        self._shape = other._shape
        self._strides = other._strides
        self._offset = other._offset
        self._device = other._device
        self._handle = other._handle
```

以`cpu`环境为例，这里的`ndarray_backend_cpu`即前文中`PYBIND11_MODULE(ndarray_backend_cpu, m){...}`生成的python模块`ndarray_backend_cpu`，这个模块作为`mod`参数创建一个`BackendDevice`的实例，该实例包含`ndarray_backend_cpu`的各种方法，其中也包括`scalar_add`。

```python
class BackendDevice:
    """A backend device, wrapps the implementation module."""

    def __init__(self, name, mod):
        self.name = name
        self.mod = mod
    ...

def cpu():
    """Return cpu device"""
    return BackendDevice("cpu", ndarray_backend_cpu)
```

### 5 跨步计算的转换

我们可以利用步长和偏移量来执行零拷贝的转换和切片操作。

- 广播（Broadcast）：插入步长为 0 的步幅（strides） 

- 转置（Transpose）：交换步幅 

- 切片（Slice）：修改偏移量和形状 

然而，对于大多数计算，我们会首先调用 `array.compact()` 方法，**以获取一个连续且对齐的内存**，然后再运行计算。
