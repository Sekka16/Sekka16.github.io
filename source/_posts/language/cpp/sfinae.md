---
title: C++中的SFINAE
date: 2024-07-11 21:00:00
tags:
categories:
description: 
---

# Substitution Failure Is Not An Error

## 模板的特化（非必须）


## SFINAE的基本原理

**核心思想：**当编译器在模板实例化过程中尝试替换模板参数时，如果替换失败，编译器不会报错而是尝试其他可能的重载，只有当所有可能的重载全部都失败时，编译器才会报错。


