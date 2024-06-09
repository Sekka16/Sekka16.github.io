# C++运算符重载

## 安装g++13

http://www.laoxu.date/html/202403/2248.html

## c++中左值引用和右值引用

### 左值和右值

- 左值：变量名、解引用的指针变量。**总体而言，非临时的，可以取地址的对象就是左值。**
- 右值：字面常量、表达式返回值（传值而不是传引用）。**总体而言，临时的，不可以取地址的对象就是右值。**

eg:
```cpp
// 以下写法均不能通过编译
10 = 4;
x + y = 4;
min(x, y) = 4;
&10
&(x + y)
&min(x, y)
```

### 左值引用和右值引用
 
#### 左值引用

左值引用就是给左值取别名，**主要作用是避免对象拷贝**

eg：
```cpp
// 以下几个是对上面左值的左值引用
int& ra = a;
int*& rp = p;
int& r = *p;
const int& rb = b;
const int& ref_a = 5; // const左值引用可以指向右值，因为const左值引用不会修改指向值
```

`int& ra = a;`：这一行创建了一个名为 ra 的左值引用，它引用了一个 int 类型的变量 a。
这意味着 ra 和 a 引用了同一个内存位置，对 ra 的修改会影响到 a，反之亦然。
`int*& rp = p;`：这一行创建了一个名为 rp 的左值引用，它引用了一个指针变量 p。rp 是一个指向指针的引用。
这意味着 rp 和 p 指向同一个内存位置，对 rp 的修改会影响到 p，而不是影响到 p 指向的内存中的内容。
`int& r = *p;`：这一行创建了一个名为 r 的左值引用，它引用了 p 所指向的内存中的值。
即 r 引用了指针 p 所指向的内存位置中存储的 int 类型的值。这意味着对 r 的修改会直接修改指针 p 所指向的内存位置中的值。
`const int& rb = b;`：这一行创建了一个名为 rb 的左值引用，它引用了一个 const int 类型的变量 b。
这里使用了常量左值引用，因此无法通过 rb 修改 b 的值，但可以使用 rb 来读取 b 的值。

#### 右值引用

右值引用就是给右值取别名，**主要作用是延长对象的生命周期，一般是延长到作用域的scope之外**

eg:
```cpp
int &&ref_a_right = 5; // ok
 
int a = 5;
int &&ref_a_left = a; // 编译不过，右值引用不可以指向左值
 
ref_a_right = 6; // 右值引用的用途：可以修改右值

```

> 更多参考：https://zhuanlan.zhihu.com/p/335994370


## c++中的运算符重载

```cpp
#include <iostream>
#include <print>

template <typename T> struct Point {
  T x;
  T y;

  Point(T xCoord, T yCoord) : x(xCoord), y(yCoord) {}

  Point operator+(const Point &other) const {
    return Point(x + other.x, y + other.y);
  }

  Point operator*(const Point &other) const {
    return Point(x * other.x, y * other.y);
  }

  Point operator-(const Point &other) const {
    return Point(x - other.x, y - other.y);
  }

  friend std::ostream &operator<<(std::ostream &os, const Point &p) {
    os << "(" << p.x << ", " << p.y << ")";
    return os;
  }
};

template <typename T, typename T2>
Point<T> operator*(const T2 &scalar_p, const Point<T> &p) {
  T scalar = static_cast<T>(scalar_p);
  return Point<T>(p.x * scalar, p.y * scalar);
}

template <typename T> Point<T> operator+(const T &scalar, const Point<T> &p) {
  return Point<T>(scalar + p.x, scalar + p.y);
}

template <typename T> Point<T> operator-(const T &scalar, const Point<T> &p) {
  return Point<T>(scalar - p.x, scalar - p.y);
}

template <typename T> Point<T> operator*(const Point<T> &p, const T &scalar) {
  return Point<T>(p.x * scalar, p.y * scalar);
}

template <typename T> Point<T> operator+(const Point<T> &p, const T &scalar) {
  return Point<T>(scalar + p.x, scalar + p.y);
}

template <typename T> Point<T> operator-(const Point<T> &p, const T &scalar) {
  return Point<T>(scalar - p.x, scalar - p.y);
}

int main() {

  Point<int> a(1, 2);
  Point<int> b(1, 2);
  Point<short> c(1, 2);
  Point<unsigned int> d(1, 2);
  Point<long> e(1, 2);
  Point<float> f1(1.0f, 2.0f);
  Point<float> f2(2.5f, 3.0f);
  Point<float> f3(-2.0f, 0.0f);

  Point<float> F = 2.0f * f1 + f2 - 4 * f3;

  std::cout << F << std::endl;

  return 0;
}

```
