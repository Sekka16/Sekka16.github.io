---
title: wsl2内存扩充
date: 2024-06-17 22:57:07
tags:
categories:
description:
---

# 标题五个字

打开Windows资源管理器，地址栏输入 `%UserProfile%` 回车，在该目录下创建一个文件，名字为`.wslconfig`

```conf
[wsl2]
memory=2GB
swap=4GB
localhostForwarding=true
```

接着在`powershell`中`wsl --shutdown`再重新打开即可