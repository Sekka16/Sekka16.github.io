---
title: MLC Lesson6 矩阵乘法样例分析
date: 2024-03-31 19:09:35
tags:
description: TVM矩阵乘法课程样例，与CPU不同，GPU上涉及线程块和线程，访存也有所优化，包括local memory、shared memory、global memory的数据传递等。本文为笔记中独立出来的样例部分，详细展示了样例的每一步变换。
---

# MLC Lesson8 GPU 矩阵乘法样例分析

## 矩阵乘法的样例

```python
import ipdb

def cache_read_and_coop_fetch(sch, block, nthread, read_idx, read_loc):
    read_cache = sch.cache_read(block=block, read_buffer_index=read_idx, storage_scope="shared")
    sch.compute_at(block=read_cache, loop=read_loc)
    # vectorized cooperative fetch
    inner0, inner1 = sch.get_loops(block=read_cache)[-2:]
    inner = sch.fuse(inner0, inner1)
    _, tx, vec = sch.split(loop=inner, factors=[None, nthread, 4])
    sch.vectorize(vec)
    sch.bind(tx, "threadIdx.x")

def blocking_with_shared(
    sch,
    tile_local_y,
    tile_local_x,
    tile_block_y,
    tile_block_x,
    tile_k):
    block_C = sch.get_block("C")
    C_local = sch.cache_write(block_C, 0, "local")

    i, j, k = sch.get_loops(block=block_C)

    i0, i1, i2 = sch.split(loop=i, factors=[None, tile_block_y, tile_local_y])
    j0, j1, j2 = sch.split(loop=j, factors=[None, tile_block_x, tile_local_x])
    k0, k1 = sch.split(loop=k, factors=[None, tile_k])

    sch.reorder(i0, j0, i1, j1, k0, k1, i2, j2)
    ipdb.set_trace()
    sch.reverse_compute_at(C_local, j1)

    sch.bind(i0, "blockIdx.y")
    sch.bind(j0, "blockIdx.x")

    tx = sch.fuse(i1, j1)
    sch.bind(tx, "threadIdx.x")
    nthread = tile_block_y * tile_block_x
    cache_read_and_coop_fetch(sch, block_C, nthread, 0, k0)
    cache_read_and_coop_fetch(sch, block_C, nthread, 1, k0)
    sch.decompose_reduction(block_C, k0)

    return sch

sch = tvm.tir.Schedule(MyModuleMatmul)
sch = blocking_with_shared(sch, 8, 8, 8, 8, 8)
sch.mod.show()
```

未做任何变换前的IRModule：

```python
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((1024, 1024), "float32"), B: T.Buffer((1024, 1024), "float32"), C: T.Buffer((1024, 1024), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i, j, k in T.grid(1024, 1024, 1024):
            with T.block("C"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                T.reads(A[vi, vk], B[vk, vj])
                T.writes(C[vi, vj])
                with T.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
```

这里的`cache_write`创建的类型是local，每个线程私有的内存空间，这样做的原因可以不用频繁地访问global memory？

```
    block_C = sch.get_block("C")
    C_local = sch.cache_write(block_C, 0, "local")
```

```python
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((1024, 1024), "float32"), B: T.Buffer((1024, 1024), "float32"), C: T.Buffer((1024, 1024), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
##################################################################################
        C_local = T.alloc_buffer((1024, 1024), scope="local")
##################################################################################
        for i, j, k in T.grid(1024, 1024, 1024):
            with T.block("C"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                T.reads(A[vi, vk], B[vk, vj])
                T.writes(C_local[vi, vj])
                with T.init():
                    C_local[vi, vj] = T.float32(0)
                C_local[vi, vj] = C_local[vi, vj] + A[vi, vk] * B[vk, vj]
        for ax0, ax1 in T.grid(1024, 1024):
            with T.block("C_local"):
                v0, v1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(C_local[v0, v1])
                T.writes(C[v0, v1])
                C[v0, v1] = C_local[v0, v1]
```

坐标轴分割：

```
    i, j, k = sch.get_loops(block=block_C)

    i0, i1, i2 = sch.split(loop=i, factors=[None, tile_block_y, tile_local_y])
    j0, j1, j2 = sch.split(loop=j, factors=[None, tile_block_x, tile_local_x])
    k0, k1 = sch.split(loop=k, factors=[None, tile_k])
```

```python
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((1024, 1024), "float32"), B: T.Buffer((1024, 1024), "float32"), C: T.Buffer((1024, 1024), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        C_local = T.alloc_buffer((1024, 1024), scope="local")
        for i_0, i_1, i_2, j_0, j_1, j_2, k_0, k_1 in T.grid(16, 8, 8, 16, 8, 8, 128, 8):
            with T.block("C"):
                vi = T.axis.spatial(1024, i_0 * 64 + i_1 * 8 + i_2)
                vj = T.axis.spatial(1024, j_0 * 64 + j_1 * 8 + j_2)
                vk = T.axis.reduce(1024, k_0 * 8 + k_1)
                T.reads(A[vi, vk], B[vk, vj])
                T.writes(C_local[vi, vj])
                with T.init():
                    C_local[vi, vj] = T.float32(0)
                C_local[vi, vj] = C_local[vi, vj] + A[vi, vk] * B[vk, vj]
        for ax0, ax1 in T.grid(1024, 1024):
            with T.block("C_local"):
                v0, v1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(C_local[v0, v1])
                T.writes(C[v0, v1])
                C[v0, v1] = C_local[v0, v1]
```

坐标轴重新排序后：

```python
sch.reorder(i0, j0, i1, j1, k0, k1, i2, j2)
```

```python
@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((1024, 1024), "float32"), B: T.Buffer((1024, 1024), "float32"), C: T.Buffer((1024, 1024), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        C_local = T.alloc_buffer((1024, 1024), scope="local")
        for i_0, j_0, i_1, j_1, k_0, k_1, i_2, j_2 in T.grid(16, 16, 8, 8, 128, 8, 8, 8):
            with T.block("C"):
                vi = T.axis.spatial(1024, i_0 * 64 + i_1 * 8 + i_2)
                vj = T.axis.spatial(1024, j_0 * 64 + j_1 * 8 + j_2)
                vk = T.axis.reduce(1024, k_0 * 8 + k_1)
                T.reads(A[vi, vk], B[vk, vj])
                T.writes(C_local[vi, vj])
                with T.init():
                    C_local[vi, vj] = T.float32(0)
                C_local[vi, vj] = C_local[vi, vj] + A[vi, vk] * B[vk, vj]
        for ax0, ax1 in T.grid(1024, 1024):
            with T.block("C_local"):
                v0, v1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(C_local[v0, v1])
                T.writes(C[v0, v1])
                C[v0, v1] = C_local[v0, v1]
```

将写回的操作挂到计算的操作的循环内：

```python
sch.reverse_compute_at(C_local, j1)
```

```python
@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((1024, 1024), "float32"), B: T.Buffer((1024, 1024), "float32"), C: T.Buffer((1024, 1024), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        C_local = T.alloc_buffer((1024, 1024), scope="local")
        for i_0, j_0, i_1, j_1 in T.grid(16, 16, 8, 8):
            for k_0, k_1, i_2, j_2 in T.grid(128, 8, 8, 8):
                with T.block("C"):
                    vi = T.axis.spatial(1024, i_0 * 64 + i_1 * 8 + i_2)
                    vj = T.axis.spatial(1024, j_0 * 64 + j_1 * 8 + j_2)
                    vk = T.axis.reduce(1024, k_0 * 8 + k_1)
                    T.reads(A[vi, vk], B[vk, vj])
                    T.writes(C_local[vi, vj])
                    with T.init():
                        C_local[vi, vj] = T.float32(0)
                    C_local[vi, vj] = C_local[vi, vj] + A[vi, vk] * B[vk, vj]
##################################################################################
            for ax0, ax1 in T.grid(8, 8):
                with T.block("C_local"):
                    v0 = T.axis.spatial(1024, i_0 * 64 + i_1 * 8 + ax0)
                    v1 = T.axis.spatial(1024, j_0 * 64 + j_1 * 8 + ax1)
                    T.reads(C_local[v0, v1])
                    T.writes(C[v0, v1])
                    C[v0, v1] = C_local[v0, v1]
##################################################################################
```

绑定坐标轴到线程块的id，这里的id是二维坐标包含`blockIdx.y`和`blockIdx.x`：

```python
    sch.bind(i0, "blockIdx.y")
    sch.bind(j0, "blockIdx.x")
```

```python
@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((1024, 1024), "float32"), B: T.Buffer((1024, 1024), "float32"), C: T.Buffer((1024, 1024), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        C_local = T.alloc_buffer((1024, 1024), scope="local")
        for i_0 in T.thread_binding(16, thread="blockIdx.y"):
            for j_0 in T.thread_binding(16, thread="blockIdx.x"):
                for i_1, j_1 in T.grid(8, 8):
                    for k_0, k_1, i_2, j_2 in T.grid(128, 8, 8, 8):
                        with T.block("C"):
                            vi = T.axis.spatial(1024, i_0 * 64 + i_1 * 8 + i_2)
                            vj = T.axis.spatial(1024, j_0 * 64 + j_1 * 8 + j_2)
                            vk = T.axis.reduce(1024, k_0 * 8 + k_1)
                            T.reads(A[vi, vk], B[vk, vj])
                            T.writes(C_local[vi, vj])
                            with T.init():
                                C_local[vi, vj] = T.float32(0)
                            C_local[vi, vj] = C_local[vi, vj] + A[vi, vk] * B[vk, vj]
                    for ax0, ax1 in T.grid(8, 8):
                        with T.block("C_local"):
                            v0 = T.axis.spatial(1024, i_0 * 64 + i_1 * 8 + ax0)
                            v1 = T.axis.spatial(1024, j_0 * 64 + j_1 * 8 + ax1)
                            T.reads(C_local[v0, v1])
                            T.writes(C[v0, v1])
                            C[v0, v1] = C_local[v0, v1]
```

坐标轴融合：

```python
tx = sch.fuse(i1, j1)
```

```python
@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((1024, 1024), "float32"), B: T.Buffer((1024, 1024), "float32"), C: T.Buffer((1024, 1024), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        C_local = T.alloc_buffer((1024, 1024), scope="local")
        for i_0 in T.thread_binding(16, thread="blockIdx.y"):
            for j_0 in T.thread_binding(16, thread="blockIdx.x"):
                for i_1_j_1_fused in range(64):
                    for k_0, k_1, i_2, j_2 in T.grid(128, 8, 8, 8):
                        with T.block("C"):
                            vi = T.axis.spatial(1024, i_0 * 64 + i_1_j_1_fused // 8 * 8 + i_2)
                            vj = T.axis.spatial(1024, j_0 * 64 + i_1_j_1_fused % 8 * 8 + j_2)
                            vk = T.axis.reduce(1024, k_0 * 8 + k_1)
                            T.reads(A[vi, vk], B[vk, vj])
                            T.writes(C_local[vi, vj])
                            with T.init():
                                C_local[vi, vj] = T.float32(0)
                            C_local[vi, vj] = C_local[vi, vj] + A[vi, vk] * B[vk, vj]
                    for ax0, ax1 in T.grid(8, 8):
                        with T.block("C_local"):
                            v0 = T.axis.spatial(1024, i_0 * 64 + i_1_j_1_fused // 8 * 8 + ax0)
                            v1 = T.axis.spatial(1024, j_0 * 64 + i_1_j_1_fused % 8 * 8 + ax1)
                            T.reads(C_local[v0, v1])
                            T.writes(C[v0, v1])
                            C[v0, v1] = C_local[v0, v1]
```

坐标轴融合之后再绑定线程号`threadIdx.x`

```python
sch.bind(tx, "threadIdx.x")
```

```python
@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((1024, 1024), "float32"), B: T.Buffer((1024, 1024), "float32"), C: T.Buffer((1024, 1024), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        C_local = T.alloc_buffer((1024, 1024), scope="local")
        for i_0 in T.thread_binding(16, thread="blockIdx.y"):
            for j_0 in T.thread_binding(16, thread="blockIdx.x"):
                for i_1_j_1_fused in T.thread_binding(64, thread="threadIdx.x"):
                    for k_0, k_1, i_2, j_2 in T.grid(128, 8, 8, 8):
                        with T.block("C"):
                            vi = T.axis.spatial(1024, i_0 * 64 + i_1_j_1_fused // 8 * 8 + i_2)
                            vj = T.axis.spatial(1024, j_0 * 64 + i_1_j_1_fused % 8 * 8 + j_2)
                            vk = T.axis.reduce(1024, k_0 * 8 + k_1)
                            T.reads(A[vi, vk], B[vk, vj])
                            T.writes(C_local[vi, vj])
                            with T.init():
                                C_local[vi, vj] = T.float32(0)
                            C_local[vi, vj] = C_local[vi, vj] + A[vi, vk] * B[vk, vj]
                    for ax0, ax1 in T.grid(8, 8):
                        with T.block("C_local"):
                            v0 = T.axis.spatial(1024, i_0 * 64 + i_1_j_1_fused // 8 * 8 + ax0)
                            v1 = T.axis.spatial(1024, j_0 * 64 + i_1_j_1_fused % 8 * 8 + ax1)
                            T.reads(C_local[v0, v1])
                            T.writes(C[v0, v1])
                            C[v0, v1] = C_local[v0, v1]
```

接下来是`cache_read_and_coop_fetch(sch, block, nthread, read_idx, read_loc)`函数中的实现:

参数说明：

- `sch`: `tvm.tir.Schedule`对象，表示待调度的计算图的调度器。
- `block`: 表示待缓存读取的计算块（block），通常是某个计算块的索引或标识。
- `nthread`: 表示协作获取的线程数，用于确定如何分割计算块以进行协作获取。
- `read_idx`: 表示缓存读取操作的索引，用于区分不同的缓存读取操作。（The index of the buffer in block’s read region.）
- `read_loc`: 表示在哪个循环层级进行计算，通常是某个循环的索引或标识。

首先在调度器 `sch` 上创建一个缓存读取操作，并将其结果存储到共享内存中：

```python
read_cache = sch.cache_read(block=block, read_buffer_index=read_idx, storage_scope="shared")
```

其中`read_cache`的类型是一个block

```python
type(read_cache)
<class 'tvm.tir.schedule.schedule.BlockRV'>
```

```python
@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((1024, 1024), "float32"), B: T.Buffer((1024, 1024), "float32"), C: T.Buffer((1024, 1024), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        C_local = T.alloc_buffer((1024, 1024), scope="local")
##################################################################################
        A_shared = T.alloc_buffer((1024, 1024), scope="shared")
        for ax0, ax1 in T.grid(1024, 1024):
            with T.block("A_shared"):
                v0, v1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(A[v0, v1])
                T.writes(A_shared[v0, v1])
                A_shared[v0, v1] = A[v0, v1]
##################################################################################
        for i_0 in T.thread_binding(16, thread="blockIdx.y"):
            for j_0 in T.thread_binding(16, thread="blockIdx.x"):
                for i_1_j_1_fused in T.thread_binding(64, thread="threadIdx.x"):
                    for k_0, k_1, i_2, j_2 in T.grid(128, 8, 8, 8):
                        with T.block("C"):
                            vi = T.axis.spatial(1024, i_0 * 64 + i_1_j_1_fused // 8 * 8 + i_2)
                            vj = T.axis.spatial(1024, j_0 * 64 + i_1_j_1_fused % 8 * 8 + j_2)
                            vk = T.axis.reduce(1024, k_0 * 8 + k_1)
                            T.reads(A_shared[vi, vk], B[vk, vj])
                            T.writes(C_local[vi, vj])
                            with T.init():
                                C_local[vi, vj] = T.float32(0)
                            C_local[vi, vj] = C_local[vi, vj] + A_shared[vi, vk] * B[vk, vj]
                    for ax0, ax1 in T.grid(8, 8):
                        with T.block("C_local"):
                            v0 = T.axis.spatial(1024, i_0 * 64 + i_1_j_1_fused // 8 * 8 + ax0)
                            v1 = T.axis.spatial(1024, j_0 * 64 + i_1_j_1_fused % 8 * 8 + ax1)
                            T.reads(C_local[v0, v1])
                            T.writes(C[v0, v1])
                            C[v0, v1] = C_local[v0, v1]
```

将创建出来的共享内存块的读写移动到循环`read_loc`上

```python
sch.compute_at(block=read_cache, loop=read_loc)
```

```python
@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((1024, 1024), "float32"), B: T.Buffer((1024, 1024), "float32"), C: T.Buffer((1024, 1024), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        C_local = T.alloc_buffer((1024, 1024), scope="local")
        A_shared = T.alloc_buffer((1024, 1024), scope="shared")
        for i_0 in T.thread_binding(16, thread="blockIdx.y"):
            for j_0 in T.thread_binding(16, thread="blockIdx.x"):
                for i_1_j_1_fused in T.thread_binding(64, thread="threadIdx.x"):
                    for k_0 in range(128):
##################################################################################
                        for ax0, ax1 in T.grid(64, 8):
                            with T.block("A_shared"):
                                v0 = T.axis.spatial(1024, i_0 * 64 + ax0)
                                v1 = T.axis.spatial(1024, k_0 * 8 + ax1)
                                T.reads(A[v0, v1])
                                T.writes(A_shared[v0, v1])
                                A_shared[v0, v1] = A[v0, v1]
##################################################################################
                        for k_1, i_2, j_2 in T.grid(8, 8, 8):
                            with T.block("C"):
                                vi = T.axis.spatial(1024, i_0 * 64 + i_1_j_1_fused // 8 * 8 + i_2)
                                vj = T.axis.spatial(1024, j_0 * 64 + i_1_j_1_fused % 8 * 8 + j_2)
                                vk = T.axis.reduce(1024, k_0 * 8 + k_1)
                                T.reads(A_shared[vi, vk], B[vk, vj])
                                T.writes(C_local[vi, vj])
                                with T.init():
                                    C_local[vi, vj] = T.float32(0)
                                C_local[vi, vj] = C_local[vi, vj] + A_shared[vi, vk] * B[vk, vj]
                    for ax0, ax1 in T.grid(8, 8):
                        with T.block("C_local"):
                            v0 = T.axis.spatial(1024, i_0 * 64 + i_1_j_1_fused // 8 * 8 + ax0)
                            v1 = T.axis.spatial(1024, j_0 * 64 + i_1_j_1_fused % 8 * 8 + ax1)
                            T.reads(C_local[v0, v1])
                            T.writes(C[v0, v1])
                            C[v0, v1] = C_local[v0, v1]
```

获取`read_cache`中最内两层循环，并且融合，这里的`read_cache`也就是`A_shared`块

```python
inner0, inner1 = sch.get_loops(block=read_cache)[-2:]
inner = sch.fuse(inner0, inner1)
```

```python
@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((1024, 1024), "float32"), B: T.Buffer((1024, 1024), "float32"), C: T.Buffer((1024, 1024), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        C_local = T.alloc_buffer((1024, 1024), scope="local")
        A_shared = T.alloc_buffer((1024, 1024), scope="shared")
        for i_0 in T.thread_binding(16, thread="blockIdx.y"):
            for j_0 in T.thread_binding(16, thread="blockIdx.x"):
                for i_1_j_1_fused in T.thread_binding(64, thread="threadIdx.x"):
                    for k_0 in range(128):
                        for ax0_ax1_fused in range(512):
                            with T.block("A_shared"):
                                v0 = T.axis.spatial(1024, i_0 * 64 + ax0_ax1_fused // 8)
                                v1 = T.axis.spatial(1024, k_0 * 8 + ax0_ax1_fused % 8)
                                T.reads(A[v0, v1])
                                T.writes(A_shared[v0, v1])
                                A_shared[v0, v1] = A[v0, v1]
                        for k_1, i_2, j_2 in T.grid(8, 8, 8):
                            with T.block("C"):
                                vi = T.axis.spatial(1024, i_0 * 64 + i_1_j_1_fused // 8 * 8 + i_2)
                                vj = T.axis.spatial(1024, j_0 * 64 + i_1_j_1_fused % 8 * 8 + j_2)
                                vk = T.axis.reduce(1024, k_0 * 8 + k_1)
                                T.reads(A_shared[vi, vk], B[vk, vj])
                                T.writes(C_local[vi, vj])
                                with T.init():
                                    C_local[vi, vj] = T.float32(0)
                                C_local[vi, vj] = C_local[vi, vj] + A_shared[vi, vk] * B[vk, vj]
                    for ax0, ax1 in T.grid(8, 8):
                        with T.block("C_local"):
                            v0 = T.axis.spatial(1024, i_0 * 64 + i_1_j_1_fused // 8 * 8 + ax0)
                            v1 = T.axis.spatial(1024, j_0 * 64 + i_1_j_1_fused % 8 * 8 + ax1)
                            T.reads(C_local[v0, v1])
                            T.writes(C[v0, v1])
                            C[v0, v1] = C_local[v0, v1]
```

分割刚刚融合的循环，绑定线程坐标以及最内层向量化

```python
_, tx, vec = sch.split(loop=inner, factors=[None, nthread, 4])
sch.vectorize(vec)
sch.bind(tx, "threadIdx.x")
```

```python
@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((1024, 1024), "float32"), B: T.Buffer((1024, 1024), "float32"), C: T.Buffer((1024, 1024), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        C_local = T.alloc_buffer((1024, 1024), scope="local")
        A_shared = T.alloc_buffer((1024, 1024), scope="shared")
        for i_0 in T.thread_binding(16, thread="blockIdx.y"):
            for j_0 in T.thread_binding(16, thread="blockIdx.x"):
                for i_1_j_1_fused in T.thread_binding(64, thread="threadIdx.x"):
                    for k_0 in range(128):
                        for ax0_ax1_fused_0 in range(2):
                            for ax0_ax1_fused_1 in T.thread_binding(64, thread="threadIdx.x"):
                                for ax0_ax1_fused_2 in T.vectorized(4):
                                    with T.block("A_shared"):
                                        v0 = T.axis.spatial(1024, i_0 * 64 + (ax0_ax1_fused_0 * 256 + ax0_ax1_fused_1 * 4 + ax0_ax1_fused_2) // 8)
                                        v1 = T.axis.spatial(1024, k_0 * 8 + (ax0_ax1_fused_0 * 256 + ax0_ax1_fused_1 * 4 + ax0_ax1_fused_2) % 8)
                                        T.reads(A[v0, v1])
                                        T.writes(A_shared[v0, v1])
                                        A_shared[v0, v1] = A[v0, v1]
                        for k_1, i_2, j_2 in T.grid(8, 8, 8):
                            with T.block("C"):
                                vi = T.axis.spatial(1024, i_0 * 64 + i_1_j_1_fused // 8 * 8 + i_2)
                                vj = T.axis.spatial(1024, j_0 * 64 + i_1_j_1_fused % 8 * 8 + j_2)
                                vk = T.axis.reduce(1024, k_0 * 8 + k_1)
                                T.reads(A_shared[vi, vk], B[vk, vj])
                                T.writes(C_local[vi, vj])
                                with T.init():
                                    C_local[vi, vj] = T.float32(0)
                                C_local[vi, vj] = C_local[vi, vj] + A_shared[vi, vk] * B[vk, vj]
                    for ax0, ax1 in T.grid(8, 8):
                        with T.block("C_local"):
                            v0 = T.axis.spatial(1024, i_0 * 64 + i_1_j_1_fused // 8 * 8 + ax0)
                            v1 = T.axis.spatial(1024, j_0 * 64 + i_1_j_1_fused % 8 * 8 + ax1)
                            T.reads(C_local[v0, v1])
                            T.writes(C[v0, v1])
                            C[v0, v1] = C_local[v0, v1]
```

第二个`cache_read_and_coop_fetch(sch, block_C, nthread, 1, k0)`，对应于`B_shared`

```python
@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((1024, 1024), "float32"), B: T.Buffer((1024, 1024), "float32"), C: T.Buffer((1024, 1024), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        C_local = T.alloc_buffer((1024, 1024), scope="local")
        A_shared = T.alloc_buffer((1024, 1024), scope="shared")
        B_shared = T.alloc_buffer((1024, 1024), scope="shared")
        for i_0 in T.thread_binding(16, thread="blockIdx.y"):
            for j_0 in T.thread_binding(16, thread="blockIdx.x"):
                for i_1_j_1_fused in T.thread_binding(64, thread="threadIdx.x"):
                    for k_0 in range(128):
                        for ax0_ax1_fused_0 in range(2):
                            for ax0_ax1_fused_1 in T.thread_binding(64, thread="threadIdx.x"):
                                for ax0_ax1_fused_2 in T.vectorized(4):
                                    with T.block("A_shared"):
                                        v0 = T.axis.spatial(1024, i_0 * 64 + (ax0_ax1_fused_0 * 256 + ax0_ax1_fused_1 * 4 + ax0_ax1_fused_2) // 8)
                                        v1 = T.axis.spatial(1024, k_0 * 8 + (ax0_ax1_fused_0 * 256 + ax0_ax1_fused_1 * 4 + ax0_ax1_fused_2) % 8)
                                        T.reads(A[v0, v1])
                                        T.writes(A_shared[v0, v1])
                                        A_shared[v0, v1] = A[v0, v1]
                        for ax0_ax1_fused_0 in range(2):
                            for ax0_ax1_fused_1 in T.thread_binding(64, thread="threadIdx.x"):
                                for ax0_ax1_fused_2 in T.vectorized(4):
                                    with T.block("B_shared"):
                                        v0 = T.axis.spatial(1024, k_0 * 8 + (ax0_ax1_fused_0 * 256 + ax0_ax1_fused_1 * 4 + ax0_ax1_fused_2) // 64)
                                        v1 = T.axis.spatial(1024, j_0 * 64 + (ax0_ax1_fused_0 * 256 + ax0_ax1_fused_1 * 4 + ax0_ax1_fused_2) % 64)
                                        T.reads(B[v0, v1])
                                        T.writes(B_shared[v0, v1])
                                        B_shared[v0, v1] = B[v0, v1]
                        for k_1, i_2, j_2 in T.grid(8, 8, 8):
                            with T.block("C"):
                                vi = T.axis.spatial(1024, i_0 * 64 + i_1_j_1_fused // 8 * 8 + i_2)
                                vj = T.axis.spatial(1024, j_0 * 64 + i_1_j_1_fused % 8 * 8 + j_2)
                                vk = T.axis.reduce(1024, k_0 * 8 + k_1)
                                T.reads(A_shared[vi, vk], B_shared[vk, vj])
                                T.writes(C_local[vi, vj])
                                with T.init():
                                    C_local[vi, vj] = T.float32(0)
                                C_local[vi, vj] = C_local[vi, vj] + A_shared[vi, vk] * B_shared[vk, vj]
                    for ax0, ax1 in T.grid(8, 8):
                        with T.block("C_local"):
                            v0 = T.axis.spatial(1024, i_0 * 64 + i_1_j_1_fused // 8 * 8 + ax0)
                            v1 = T.axis.spatial(1024, j_0 * 64 + i_1_j_1_fused % 8 * 8 + ax1)
                            T.reads(C_local[v0, v1])
                            T.writes(C[v0, v1])
                            C[v0, v1] = C_local[v0, v1]
```

分离初始化和计算操作：

```python
sch.decompose_reduction(block_C, k0)
```

```python
@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((1024, 1024), "float32"), B: T.Buffer((1024, 1024), "float32"), C: T.Buffer((1024, 1024), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        C_local = T.alloc_buffer((1024, 1024), scope="local")
        A_shared = T.alloc_buffer((1024, 1024), scope="shared")
        B_shared = T.alloc_buffer((1024, 1024), scope="shared")
        for i_0 in T.thread_binding(16, thread="blockIdx.y"):
            for j_0 in T.thread_binding(16, thread="blockIdx.x"):
                for i_1_j_1_fused in T.thread_binding(64, thread="threadIdx.x"):
                    for i_2_init, j_2_init in T.grid(8, 8):
                        with T.block("C_init"):
                            vi = T.axis.spatial(1024, i_0 * 64 + i_1_j_1_fused // 8 * 8 + i_2_init)
                            vj = T.axis.spatial(1024, j_0 * 64 + i_1_j_1_fused % 8 * 8 + j_2_init)
                            T.reads()
                            T.writes(C_local[vi, vj])
                            C_local[vi, vj] = T.float32(0)
                    for k_0 in range(128):
                        for ax0_ax1_fused_0 in range(2):
                            for ax0_ax1_fused_1 in T.thread_binding(64, thread="threadIdx.x"):
                                for ax0_ax1_fused_2 in T.vectorized(4):
                                    with T.block("A_shared"):
                                        v0 = T.axis.spatial(1024, i_0 * 64 + (ax0_ax1_fused_0 * 256 + ax0_ax1_fused_1 * 4 + ax0_ax1_fused_2) // 8)
                                        v1 = T.axis.spatial(1024, k_0 * 8 + (ax0_ax1_fused_0 * 256 + ax0_ax1_fused_1 * 4 + ax0_ax1_fused_2) % 8)
                                        T.reads(A[v0, v1])
                                        T.writes(A_shared[v0, v1])
                                        A_shared[v0, v1] = A[v0, v1]
                        for ax0_ax1_fused_0 in range(2):
                            for ax0_ax1_fused_1 in T.thread_binding(64, thread="threadIdx.x"):
                                for ax0_ax1_fused_2 in T.vectorized(4):
                                    with T.block("B_shared"):
                                        v0 = T.axis.spatial(1024, k_0 * 8 + (ax0_ax1_fused_0 * 256 + ax0_ax1_fused_1 * 4 + ax0_ax1_fused_2) // 64)
                                        v1 = T.axis.spatial(1024, j_0 * 64 + (ax0_ax1_fused_0 * 256 + ax0_ax1_fused_1 * 4 + ax0_ax1_fused_2) % 64)
                                        T.reads(B[v0, v1])
                                        T.writes(B_shared[v0, v1])
                                        B_shared[v0, v1] = B[v0, v1]
                        for k_1, i_2, j_2 in T.grid(8, 8, 8):
                            with T.block("C_update"):
                                vi = T.axis.spatial(1024, i_0 * 64 + i_1_j_1_fused // 8 * 8 + i_2)
                                vj = T.axis.spatial(1024, j_0 * 64 + i_1_j_1_fused % 8 * 8 + j_2)
                                vk = T.axis.reduce(1024, k_0 * 8 + k_1)
                                T.reads(C_local[vi, vj], A_shared[vi, vk], B_shared[vk, vj])
                                T.writes(C_local[vi, vj])
                                C_local[vi, vj] = C_local[vi, vj] + A_shared[vi, vk] * B_shared[vk, vj]
                    for ax0, ax1 in T.grid(8, 8):
                        with T.block("C_local"):
                            v0 = T.axis.spatial(1024, i_0 * 64 + i_1_j_1_fused // 8 * 8 + ax0)
                            v1 = T.axis.spatial(1024, j_0 * 64 + i_1_j_1_fused % 8 * 8 + ax1)
                            T.reads(C_local[v0, v1])
                            T.writes(C[v0, v1])
                            C[v0, v1] = C_local[v0, v1]
```

编译运行

```python
rt_mod = tvm.build(sch.mod, target="cuda")
dev = tvm.cuda(0)
evaluator = rt_mod.time_evaluator("main", dev, number=10)

print("GEMM-Blocking: %f GFLOPS" % (num_flop / evaluator(A_nd, B_nd, C_nd).mean / 1e9))
```

