---
title: c++运算符重载.md
date: 2024-06-04 00:37:58
tags:
---

# 标题五个字


```cpp
#include <iostream>

template<typename T> class Point {
  public:
    T x;
    T y;

  public:
    
    Point(T x, T y): x(x), y(y) {}

    // 成员函数形式的运算符重载，相当于a.operator+(b)
    Point operator+(const Point& other) {
      return Point(x+other.x, y+other.y);
    }

    // 成员函数形式的运算符重载，相当于a.operator+(scalar)
    Point operator+(T scalar) {
      return Point(x+scalar, y+scalar);
    }

    // 非成员函数形式的重载，相当于operator+(scalar, p)
    friend Point operator+(T scalar, const Point& p) {
      return Point(scalar+p.x, scalar+p.y);
    }

    // 输出运算符重载
    friend std::ostream& operator<<(std::ostream& os, const Point& p) {
      os << "(" << p.x << "," << p.y << ")";
      return os;
    }
};

int main() {

  Point<float> a(2.0f, 1.5f);
  Point<float> b(-1.0f, 3.0f);
  Point<float> c(0.0f, 5.5f);

  Point<float> F1 = 2.0f + a;
  Point<float> F2 = b + 2.0f;
  Point<float> F3 = a + c;

  std::cout << F1 << std::endl;
  std::cout << F2 << std::endl;
  std::cout << F3 << std::endl;

  return 0;
}
```