---
title: llvm-01
date: 2024-07-08 22:54:35
tags:
categories:
description:
---

# LLVM IR初体验

## 1 生成LLVM IR

以这样一段源程序`gcd.c`为例：

```C
unsigned gcd(unsigned a, unsigned b) {
  if (b == 0)
    return a;
  while (b != 0) {
    unsigned t = a % b;
    a = b;
    b = t;
  }
  return a;
}
```

运行命令获得`gcd.ll`，即获得其对应的`llvm ir`

```bash
clang --target=aarch64-linux-gnu -O1 -S -emit-llvm gcd.c
```

```llvm
; ModuleID = 'gcd.c'
source_filename = "gcd.c"
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

; Function Attrs: norecurse nounwind readnone uwtable
define dso_local i32 @gcd(i32 %0, i32 %1) local_unnamed_addr #0 {
  %3 = icmp eq i32 %1, 0
  br i1 %3, label %9, label %4

4:                                                ; preds = %2, %4
  %5 = phi i32 [ %7, %4 ], [ %1, %2 ]
  %6 = phi i32 [ %5, %4 ], [ %0, %2 ]
  %7 = urem i32 %6, %5
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %9, label %4, !llvm.loop !6

9:                                                ; preds = %4, %2
  %10 = phi i32 [ %0, %2 ], [ %5, %4 ]
  ret i32 %10
}

attributes #0 = { norecurse nounwind readnone uwtable "disable-tail-calls"="false" "frame-pointer"="non-leaf" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+neon" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"branch-target-enforcement", i32 0}
!2 = !{i32 1, !"sign-return-address", i32 0}
!3 = !{i32 1, !"sign-return-address-all", i32 0}
!4 = !{i32 1, !"sign-return-address-with-bkey", i32 0}
!5 = !{!"Ubuntu clang version 12.0.0-3ubuntu1~20.04.5"}
!6 = distinct !{!6, !7, !8}
!7 = !{!"llvm.loop.mustprogress"}
!8 = !{!"llvm.loop.unroll.disable"}
```

## Basic Properties

```llvm
; ModuleID = 'gcd.c'
source_filename = "gcd.c"
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"
```

### target datalayout

**target layout**建立了一些基础属性，具体来说

- 小写的 e 表示内存中的字节按照小端序进行存储。要指定大端序，必须使用大写的 E。
- M: 指定应用于符号的名称修饰方式。在这里，m:e 表示使用 ELF 名称修饰。
- 以 iN:A:P 形式的条目（例如 i8:8:32）指定数据的对齐方式，单位是比特。第一个数字是 ABI 要求的对齐方式，第二个数字是首选对齐方式。对于字节（i8）, ABI 对齐方式是 1 字节（8 比特），首选对齐方式是 4 字节（32 比特）。
- n 指定哪些本机寄存器大小是可用的。n32:64 意味着本机支持 32 位和 64 位宽的整数。
- S 指定栈的对齐方式，同样以比特为单位。S128 意味着栈保持 16 字节对齐。

### target triple

最后，目标三元组字符串指定我们正在编译的架构。这反映了我们在命令行上提供的信息。三元组是一个配置字符串，通常由 CPU 架构、供应商和操作系统组成。

## 2 basic blocks

对于一个基本块，有以下描述：

> A well-formed basic block is a linear sequence of instructions, which begins with an optional label and ends with a terminator instruction.

也就是说，基本块由一个可选的`label`标记开始，由分支语句(br)或者返回语句(ret)标记结束，中间是一段线性的指令序列。

```llvm
define dso_local i32 @gcd(i32 %0, i32 %1) local_unnamed_addr #0 {
  %3 = icmp eq i32 %1, 0
  br i1 %3, label %9, label %4

4:                                                ; preds = %2, %4
  %5 = phi i32 [ %7, %4 ], [ %1, %2 ]
  %6 = phi i32 [ %5, %4 ], [ %0, %2 ]
  %7 = urem i32 %6, %5
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %9, label %4, !llvm.loop !6

9:                                                ; preds = %4, %2
  %10 = phi i32 [ %0, %2 ], [ %5, %4 ]
  ret i32 %10
}
```

对于如上的程序，共分为三个`basic block`，其中第一个`basic block`隐藏了其`label`(2)。

## 3 SSA

IR 代码的另一个特点是它采用静态单赋值（SSA）形式。代码使用无限数量的虚拟寄存器，但每个寄存器仅被写入一次。比较的结果被赋值给命名的虚拟寄存器。这个寄存器随后被使用，但不会再被写入。

常量传播和公共子表达式消除等优化在 SSA 形式下效果非常好，现代编译器都采用这种形式。

**SSA 的一个基本特性是它建立了定义-使用（def-use）和使用-定义（use-def）链：对于一个单一的定义，你知道所有使用（def-use），而对于每一个使用，你知道唯一的定义（use-def）。**

```llvm
4:                                                ; preds = %2, %4
  %5 = phi i32 [ %7, %4 ], [ %1, %2 ]
  %6 = phi i32 [ %5, %4 ], [ %0, %2 ]
  %7 = urem i32 %6, %5
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %9, label %4, !llvm.loop !6
```

在这个循环中，`%5`和`%6`被赋予了新的值，这显然是不符合SSA的。解决方案是使用特殊的`phi`指令。

`phi` 指令将基本块和值的列表作为参数。基本块表示来自哪个基本块的传入边，值是该基本块的值。

具体来说，以下代码的含义为：
- 如果之前的基本块是 %4，那么 %5 的值为 %7。
- 如果之前的基本块是 %2，那么 %5 的值为 %1。
```llvm
  %5 = phi i32 [ %7, %4 ], [ %1, %2 ]
```

> **注意：**`phi`命令只能用在一个`basic block`的开始，并且由于第一个`basic block`没有前置块，所以第一条命令必然不能是`phi`命令