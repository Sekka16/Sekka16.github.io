---
title: MLC Lesson7 引入特殊内存层级示例
date: 2024-03-31 19:16:40
tags:
description: 所谓特殊内存层级，实际上就是类似于Cache的存在，具体看硬件实现。
---

# MLC-Lesson8 引入特殊内存层级示例

变换程序：

```python
A_reg = sch.cache_read(block_mm, 0, storage_scope="global.A_reg")
B_reg = sch.cache_read(block_mm, 1, storage_scope="global.B_reg")
sch.compute_at(A_reg, k)
sch.compute_at(B_reg, k)

write_back_block = sch.cache_write(block_mm, 0, storage_scope="global.accumulator")
sch.reverse_compute_at(write_back_block, j)
sch.mod.show()
```

原始的IRModule

```python
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((1024, 1024), "float32"), B: T.Buffer((1024, 1024), "float32"), C: T.Buffer((1024, 1024), "float32")):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # with T.block("root"):
        for i_0, j_0, k_0 in T.grid(64, 64, 64):
            with T.block("matmul_o"):
                vi_o, vj_o, vk_o = T.axis.remap("SSR", [i_0, j_0, k_0])
                T.reads(A[vi_o * 16:vi_o * 16 + 16, vk_o * 16:vk_o * 16 + 16], B[vj_o * 16:vj_o * 16 + 16, vk_o * 16:vk_o * 16 + 16])
                T.writes(C[vi_o * 16:vi_o * 16 + 16, vj_o * 16:vj_o * 16 + 16])
                with T.init():
                    for i_1, j_1 in T.grid(16, 16):
                        with T.block("matmul_init"):
                            vi_i_init, vj_i_init = T.axis.remap("SS", [i_1, j_1])
                            T.reads()
                            T.writes(C[vi_o * 16 + vi_i_init, vj_o * 16 + vj_i_init])
                            C[vi_o * 16 + vi_i_init, vj_o * 16 + vj_i_init] = T.float32(0)
                for i_1, j_1, k_1 in T.grid(16, 16, 16):
                    with T.block("matmul"):
                        vi_i, vj_i, vk_i = T.axis.remap("SSR", [i_1, j_1, k_1])
                        T.reads(C[vi_o * 16 + vi_i, vj_o * 16 + vj_i], A[vi_o * 16 + vi_i, vk_o * 16 + vk_i], B[vj_o * 16 + vj_i, vk_o * 16 + vk_i])
                        T.writes(C[vi_o * 16 + vi_i, vj_o * 16 + vj_i])
                        C[vi_o * 16 + vi_i, vj_o * 16 + vj_i] = C[vi_o * 16 + vi_i, vj_o * 16 + vj_i] + A[vi_o * 16 + vi_i, vk_o * 16 + vk_i] * B[vj_o * 16 + vj_i, vk_o * 16 + vk_i]
```

引入特殊内存层级

```python
A_reg = sch.cache_read(block_mm, 0, storage_scope="global.A_reg")
B_reg = sch.cache_read(block_mm, 1, storage_scope="global.B_reg")
```

```python
@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((1024, 1024), "float32"), B: T.Buffer((1024, 1024), "float32"), C: T.Buffer((1024, 1024), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
############################################################################################
        A_global_A_reg = T.alloc_buffer((1024, 1024), scope="global.A_reg")
        B_global_B_reg = T.alloc_buffer((1024, 1024), scope="global.B_reg")
        for ax0, ax1 in T.grid(1024, 1024):
            with T.block("B_global.B_reg"):
                v0, v1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(B[v0, v1])
                T.writes(B_global_B_reg[v0, v1])
                B_global_B_reg[v0, v1] = B[v0, v1]
        for ax0, ax1 in T.grid(1024, 1024):
            with T.block("A_global.A_reg"):
                v0, v1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(A[v0, v1])
                T.writes(A_global_A_reg[v0, v1])
                A_global_A_reg[v0, v1] = A[v0, v1]
############################################################################################
        for i_0, j_0, k_0 in T.grid(64, 64, 64):
            with T.block("matmul_o"):
                vi_o, vj_o, vk_o = T.axis.remap("SSR", [i_0, j_0, k_0])
                T.reads(A_global_A_reg[vi_o * 16:vi_o * 16 + 16, vk_o * 16:vk_o * 16 + 16], B_global_B_reg[vj_o * 16:vj_o * 16 + 16, vk_o * 16:vk_o * 16 + 16])
                T.writes(C[vi_o * 16:vi_o * 16 + 16, vj_o * 16:vj_o * 16 + 16])
                with T.init():
                    for i_1, j_1 in T.grid(16, 16):
                        with T.block("matmul_init"):
                            vi_i_init, vj_i_init = T.axis.remap("SS", [i_1, j_1])
                            T.reads()
                            T.writes(C[vi_o * 16 + vi_i_init, vj_o * 16 + vj_i_init])
                            C[vi_o * 16 + vi_i_init, vj_o * 16 + vj_i_init] = T.float32(0)
                for i_1, j_1, k_1 in T.grid(16, 16, 16):
                    with T.block("matmul"):
                        vi_i, vj_i, vk_i = T.axis.remap("SSR", [i_1, j_1, k_1])
                        T.reads(C[vi_o * 16 + vi_i, vj_o * 16 + vj_i], A_global_A_reg[vi_o * 16 + vi_i, vk_o * 16 + vk_i], B_global_B_reg[vj_o * 16 + vj_i, vk_o * 16 + vk_i])
                        T.writes(C[vi_o * 16 + vi_i, vj_o * 16 + vj_i])
                        C[vi_o * 16 + vi_i, vj_o * 16 + vj_i] = C[vi_o * 16 + vi_i, vj_o * 16 + vj_i] + A_global_A_reg[vi_o * 16 + vi_i, vk_o * 16 + vk_i] * B_global_B_reg[vj_o * 16 + vj_i, vk_o * 16 + vk_i]
```

将缓存操作挂到k循环下

```python
sch.compute_at(A_reg, k)
sch.compute_at(B_reg, k)
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
        A_global_A_reg = T.alloc_buffer((1024, 1024), scope="global.A_reg")
        B_global_B_reg = T.alloc_buffer((1024, 1024), scope="global.B_reg")
        for i_0, j_0, k_0 in T.grid(64, 64, 64):
############################################################################################
            for ax0, ax1 in T.grid(16, 16):
                with T.block("A_global.A_reg"):
                    v0 = T.axis.spatial(1024, i_0 * 16 + ax0)
                    v1 = T.axis.spatial(1024, k_0 * 16 + ax1)
                    T.reads(A[v0, v1])
                    T.writes(A_global_A_reg[v0, v1])
                    A_global_A_reg[v0, v1] = A[v0, v1]
            for ax0, ax1 in T.grid(16, 16):
                with T.block("B_global.B_reg"):
                    v0 = T.axis.spatial(1024, j_0 * 16 + ax0)
                    v1 = T.axis.spatial(1024, k_0 * 16 + ax1)
                    T.reads(B[v0, v1])
                    T.writes(B_global_B_reg[v0, v1])
                    B_global_B_reg[v0, v1] = B[v0, v1]
############################################################################################
            with T.block("matmul_o"):
                vi_o, vj_o, vk_o = T.axis.remap("SSR", [i_0, j_0, k_0])
                T.reads(A_global_A_reg[vi_o * 16:vi_o * 16 + 16, vk_o * 16:vk_o * 16 + 16], B_global_B_reg[vj_o * 16:vj_o * 16 + 16, vk_o * 16:vk_o * 16 + 16])
                T.writes(C[vi_o * 16:vi_o * 16 + 16, vj_o * 16:vj_o * 16 + 16])
                with T.init():
                    for i_1, j_1 in T.grid(16, 16):
                        with T.block("matmul_init"):
                            vi_i_init, vj_i_init = T.axis.remap("SS", [i_1, j_1])
                            T.reads()
                            T.writes(C[vi_o * 16 + vi_i_init, vj_o * 16 + vj_i_init])
                            C[vi_o * 16 + vi_i_init, vj_o * 16 + vj_i_init] = T.float32(0)
                for i_1, j_1, k_1 in T.grid(16, 16, 16):
                    with T.block("matmul"):
                        vi_i, vj_i, vk_i = T.axis.remap("SSR", [i_1, j_1, k_1])
                        T.reads(C[vi_o * 16 + vi_i, vj_o * 16 + vj_i], A_global_A_reg[vi_o * 16 + vi_i, vk_o * 16 + vk_i], B_global_B_reg[vj_o * 16 + vj_i, vk_o * 16 + vk_i])
                        T.writes(C[vi_o * 16 + vi_i, vj_o * 16 + vj_i])
                        C[vi_o * 16 + vi_i, vj_o * 16 + vj_i] = C[vi_o * 16 + vi_i, vj_o * 16 + vj_i] + A_global_A_reg[vi_o * 16 + vi_i, vk_o * 16 + vk_i] * B_global_B_reg[vj_o * 16 + vj_i, vk_o * 16 + vk_i]

```

创建写回的cache

```python
write_back_block = sch.cache_write(block_mm, 0, storage_scope="global.accumulator")
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
        A_global_A_reg = T.alloc_buffer((1024, 1024), scope="global.A_reg")
        B_global_B_reg = T.alloc_buffer((1024, 1024), scope="global.B_reg")
        C_global_accumulator = T.alloc_buffer((1024, 1024), scope="global.accumulator")
        for i_0, j_0, k_0 in T.grid(64, 64, 64):
            for ax0, ax1 in T.grid(16, 16):
                with T.block("A_global.A_reg"):
                    v0 = T.axis.spatial(1024, i_0 * 16 + ax0)
                    v1 = T.axis.spatial(1024, k_0 * 16 + ax1)
                    T.reads(A[v0, v1])
                    T.writes(A_global_A_reg[v0, v1])
                    A_global_A_reg[v0, v1] = A[v0, v1]
            for ax0, ax1 in T.grid(16, 16):
                with T.block("B_global.B_reg"):
                    v0 = T.axis.spatial(1024, j_0 * 16 + ax0)
                    v1 = T.axis.spatial(1024, k_0 * 16 + ax1)
                    T.reads(B[v0, v1])
                    T.writes(B_global_B_reg[v0, v1])
                    B_global_B_reg[v0, v1] = B[v0, v1]
            with T.block("matmul_o"):
                vi_o, vj_o, vk_o = T.axis.remap("SSR", [i_0, j_0, k_0])
                T.reads(A_global_A_reg[vi_o * 16:vi_o * 16 + 16, vk_o * 16:vk_o * 16 + 16], B_global_B_reg[vj_o * 16:vj_o * 16 + 16, vk_o * 16:vk_o * 16 + 16])
                T.writes(C_global_accumulator[vi_o * 16:vi_o * 16 + 16, vj_o * 16:vj_o * 16 + 16])
                with T.init():
                    for i_1, j_1 in T.grid(16, 16):
                        with T.block("matmul_init"):
                            vi_i_init, vj_i_init = T.axis.remap("SS", [i_1, j_1])
                            T.reads()
                            T.writes(C_global_accumulator[vi_o * 16 + vi_i_init, vj_o * 16 + vj_i_init])
                            C_global_accumulator[vi_o * 16 + vi_i_init, vj_o * 16 + vj_i_init] = T.float32(0)
                for i_1, j_1, k_1 in T.grid(16, 16, 16):
                    with T.block("matmul"):
                        vi_i, vj_i, vk_i = T.axis.remap("SSR", [i_1, j_1, k_1])
                        T.reads(C_global_accumulator[vi_o * 16 + vi_i, vj_o * 16 + vj_i], A_global_A_reg[vi_o * 16 + vi_i, vk_o * 16 + vk_i], B_global_B_reg[vj_o * 16 + vj_i, vk_o * 16 + vk_i])
                        T.writes(C_global_accumulator[vi_o * 16 + vi_i, vj_o * 16 + vj_i])
                        C_global_accumulator[vi_o * 16 + vi_i, vj_o * 16 + vj_i] = C_global_accumulator[vi_o * 16 + vi_i, vj_o * 16 + vj_i] + A_global_A_reg[vi_o * 16 + vi_i, vk_o * 16 + vk_i] * B_global_B_reg[vj_o * 16 + vj_i, vk_o * 16 + vk_i]
############################################################################################
        for ax0, ax1 in T.grid(1024, 1024):
            with T.block("C_global.accumulator"):
                v0, v1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(C_global_accumulator[v0, v1])
                T.writes(C[v0, v1])
                C[v0, v1] = C_global_accumulator[v0, v1]
############################################################################################
```

最后：

`compute_at`和`reverse_compute_at`得区别就是移动的是生产者块还是消费者块

```python
sch.reverse_compute_at(write_back_block, j)
```

```python
@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((1024, 1024), "float32"), B: T.Buffer((1024, 1024), "float32"), C: T.Buffer((1024, 1024), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        A_global_A_reg = T.alloc_buffer((1024, 1024), scope="global.A_reg")
        B_global_B_reg = T.alloc_buffer((1024, 1024), scope="global.B_reg")
        C_global_accumulator = T.alloc_buffer((1024, 1024), scope="global.accumulator")
        for i_0, j_0 in T.grid(64, 64):
            for k_0 in range(64):
                for ax0, ax1 in T.grid(16, 16):
                    with T.block("A_global.A_reg"):
                        v0 = T.axis.spatial(1024, i_0 * 16 + ax0)
                        v1 = T.axis.spatial(1024, k_0 * 16 + ax1)
                        T.reads(A[v0, v1])
                        T.writes(A_global_A_reg[v0, v1])
                        A_global_A_reg[v0, v1] = A[v0, v1]
                for ax0, ax1 in T.grid(16, 16):
                    with T.block("B_global.B_reg"):
                        v0 = T.axis.spatial(1024, j_0 * 16 + ax0)
                        v1 = T.axis.spatial(1024, k_0 * 16 + ax1)
                        T.reads(B[v0, v1])
                        T.writes(B_global_B_reg[v0, v1])
                        B_global_B_reg[v0, v1] = B[v0, v1]
                with T.block("matmul_o"):
                    vi_o, vj_o, vk_o = T.axis.remap("SSR", [i_0, j_0, k_0])
                    T.reads(A_global_A_reg[vi_o * 16:vi_o * 16 + 16, vk_o * 16:vk_o * 16 + 16], B_global_B_reg[vj_o * 16:vj_o * 16 + 16, vk_o * 16:vk_o * 16 + 16])
                    T.writes(C_global_accumulator[vi_o * 16:vi_o * 16 + 16, vj_o * 16:vj_o * 16 + 16])
                    with T.init():
                        for i_1, j_1 in T.grid(16, 16):
                            with T.block("matmul_init"):
                                vi_i_init, vj_i_init = T.axis.remap("SS", [i_1, j_1])
                                T.reads()
                                T.writes(C_global_accumulator[vi_o * 16 + vi_i_init, vj_o * 16 + vj_i_init])
                                C_global_accumulator[vi_o * 16 + vi_i_init, vj_o * 16 + vj_i_init] = T.float32(0)
                    for i_1, j_1, k_1 in T.grid(16, 16, 16):
                        with T.block("matmul"):
                            vi_i, vj_i, vk_i = T.axis.remap("SSR", [i_1, j_1, k_1])
                            T.reads(C_global_accumulator[vi_o * 16 + vi_i, vj_o * 16 + vj_i], A_global_A_reg[vi_o * 16 + vi_i, vk_o * 16 + vk_i], B_global_B_reg[vj_o * 16 + vj_i, vk_o * 16 + vk_i])
                            T.writes(C_global_accumulator[vi_o * 16 + vi_i, vj_o * 16 + vj_i])
                            C_global_accumulator[vi_o * 16 + vi_i, vj_o * 16 + vj_i] = C_global_accumulator[vi_o * 16 + vi_i, vj_o * 16 + vj_i] + A_global_A_reg[vi_o * 16 + vi_i, vk_o * 16 + vk_i] * B_global_B_reg[vj_o * 16 + vj_i, vk_o * 16 + vk_i]
############################################################################################
            for ax0, ax1 in T.grid(16, 16):
                with T.block("C_global.accumulator"):
                    v0 = T.axis.spatial(1024, i_0 * 16 + ax0)
                    v1 = T.axis.spatial(1024, j_0 * 16 + ax1)
                    T.reads(C_global_accumulator[v0, v1])
                    T.writes(C[v0, v1])
                    C[v0, v1] = C_global_accumulator[v0, v1]
############################################################################################
```

