---
title: vim
date: 2024-05-22 23:36:49
tags:
categories:
description:
---

# VIM入门指南

## 标签窗口相关

1. 暂时挂起进程，回到终端
```bash
# 挂起
CTRL + Z

# 回到进程
fg
```

2. vim分割窗口
```bash
# 竖直分割，vertical split
:vs

# 水平分割
:sp
```

3. 打开指定的文件
```bash
:find <filename>
```

4. 标签页
```bash
# 新建标签页
:tabe <pagename>

# 标签页切换
gT
gt

# 或者
:tabn
:tabp
```

