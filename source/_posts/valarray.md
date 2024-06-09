# c++中vector的实现

参考：https://blog.csdn.net/weixin_50941083/article/details/122354948

## vector的成员变量

```cpp
public:
    typedef T value_type;  // 这个并不是成员变量，主要时为了增强可读性
    typedef T* iterator;   // 其时就是重新起别名
private:
    value_type* _data;
    size_t _size;
    size_t _capacity;
```

> 关于_capacity参数：若没有该参数，每增添一个新的元素，就需要重新申请空间存放新的元素，这会消耗较多时间，效率不高。
故而在实现的时候我们采用倍增的方式提前申请较大空间，倍增的意思就是当元素个数大于空间大小时，申请的空间将会是原空间的二倍。

## 构造函数

```cpp
// 构造函数
vector():_data(NULL),_size(0),_capacity(0){}

// 析构函数
~vector(){
    delete [] _data;
    _data = NULL;
    _size = 0;
    _capacity = 0;
}

// 拷贝构造函数(深拷贝)
vector(const vector& vec){
    _size = vec._size;
    _capacity = vec._capacity;
    _data = new value_type[_capacity];
    for(int i=0;i<_size;++i){
        _data[i] = vec._data[i];
    }
}
```

> 浅拷贝是拷贝了对象的引用，当原对象发生变化的时候，拷贝对象也跟着变化；
深拷贝是另外申请了一块内存，内容和原对象一样，更改原对象，拷贝对象不会发生变化。 

## this指针

在 C++ 中，this 指针是一个特殊的指针，它指向当前对象的实例。
在 C++ 中，每一个对象都能通过 this 指针来访问自己的地址。
this是一个隐藏的指针，可以在类的成员函数中使用，它可以用来指向调用对象。
当一个对象的成员函数被调用时，编译器会隐式地传递该对象的地址作为 this 指针。
友元函数没有 this 指针，因为友元不是类的成员，只有成员函数才有 this 指针。

```cpp

```
