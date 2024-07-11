---
title: C++中的SFINAE
date: 2024-07-11 21:00:00
tags:
categories:
description: 
---

# Substitution Failure Is Not An Error

## 模板的特化（非必须）
```cpp
template<typename T>
struct EnableIf;

template<>
struct EnableIf<int> {
  using type = int;
};

template<>
struct EnableIf<float> {
  using type = float;
};
```

特化版本的目的是为了在模板实例化过程中，根据模板参数的类型提供不同的行为。

例如上例中当`EnableIf`的模板参数`T`是`int`时，`EnableIf<int>`结构体将包含一个`type`成员，其类型为`int`。

## SFINAE的基本原理

**核心思想：**当编译器在模板实例化过程中尝试替换模板参数时，如果替换失败，编译器不会报错而是尝试其他可能的重载，只有当所有可能的重载全部都失败时，编译器才会报错。

```cpp
#include <type_traits>

template<typename T>
struct EnableIf;

template<>
struct EnableIf<int> {
  using type = int;
};

template<>
struct EnableIf<float> {
  using type = int;
};

// T must be int, float
template<typename T, typename = typename EnableIf<T>::type>
void foo() {}

int main() {
  foo<int>(); // OK
  foo<double>(); // Error
  return 0;
}
```

上述代码运行`foo<double>()`将会报错。