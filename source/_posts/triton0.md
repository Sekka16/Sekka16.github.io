---
title: triton-01-环境配置
date: 2024-06-18 23:50:05
tags:
categories:
description:
---

# triton环境配置

## LLVM编译

1. 下载编译制定版本的llvm
   - 在Triton代码仓`cmake/llvm-hash.txt`中找到当前依赖的llvm源码版本
   - 使用git checkout <hash-id>切到该制定llvm版本
   - 编译llvm

```bash
$ cd $HOME/llvm-project  # your clone of LLVM.
$ mkdir build
$ cd build
$ cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON  ../llvm -DLLVM_ENABLE_PROJECTS="mlir;llvm"
$ ninja
```

2. 指定llvm路径编译Triton

```bash
git clone https://github.com/openai/triton.git
cd triton

# 按需创建一个python venv环境，以免与系统python环境影响
python -m venv .venv --prompt triton
source .venv/bin/activate
# 按需选择安装
pip install ninja cmake wheel 

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Modify as appropriate to point to your LLVM build.
$ export LLVM_BUILD_DIR=$HOME/llvm-project/build
$ LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include \
  LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib \
  LLVM_SYSPATH=$LLVM_BUILD_DIR \
  pip install -e python
# If you want see more build log, you can use "pip install -e python -vvv"
```

## 激活虚拟环境

在`triton`下`vim env.sh`

```bash
#!/bin/bash

# 设置LLVM环境变量
export LLVM_BUILD_DIR=$HOME/compiler/llvm-project/build
export LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include
export LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib
export LLVM_SYSPATH=$LLVM_BUILD_DIR

# 激活Python虚拟环境
source .venv/bin/activate
```