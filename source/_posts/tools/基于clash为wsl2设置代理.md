---
title: 基于clash为wsl2设置代理
date: 2024-06-07 23:40:35
tags:
categories:
description:
---


# 标题五个字

## 一、配置clash

找到General > Allow LAN，打开开关。

## 二、配置防火墙

打开控制面板，找到**系统和安全 > Windows Defender 防火墙 > 允许应用通过 Windows 防火墙**，勾选上所有Clash相关的应用，包括但不限于**Clash for Windows、clash-win64**等。

## 三、配置WLS2

1、创建`.proxy`文件，并且添加以下内容

```bash
#!/bin/bash
hostip=$(cat /etc/resolv.conf |grep -oP '(?<=nameserver\ ).*')
export https_proxy="http://${hostip}:7890"
export http_proxy="http://${hostip}:7890"
export all_proxy="socks5://${hostip}:7890"
```

2、编辑`.bashrc`或`.zshrc`文件，在文末添加一下内容

```bash
source ~/.proxy
```

## 四、测试

```bash
wget www.google.com
```

```bash
--2024-06-07 23:35:46--  http://www.google.com/
Connecting to 172.25.16.1:7890... connected.
Proxy request sent, awaiting response... 200 OK
Length: unspecified [text/html]
Saving to: ‘index.html’

index.html                                 [ <=>                                                                         ]  20.50K  --.-KB/s    in 0.02s

2024-06-07 23:35:46 (1.07 MB/s) - ‘index.html’ saved [20989]
```