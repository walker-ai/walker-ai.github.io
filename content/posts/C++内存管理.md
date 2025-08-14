---
title: C++内存管理
subtitle:
date: 2025-03-23T11:51:02+08:00
slug: 01dc872
draft: false
tags: [C++]
categories: [八股]
math: true
author:
  name: "yitao"
---

介绍 C++ 中的内存管理分配、管理以及智能指针的原理及使用

<!--more-->


## 程序地址空间

C++ 中，内存分布分为五块区域，分别是：==栈；堆；全局变量和静态变量（存放于 data 段和 bss 段）；常量；代码；==

![Image](https://cdn.ipfsscan.io/weibo/large/005wRZF3ly1i4cmz66hemj30ie0kyjun.jpg)

上图是内核和用户的虚拟地址空间分布情况，其中，==局部变量==和==参数==等都存放在==栈==中，这部分空间由系统进行管理；而==堆==中空间主要是用于用户调用 `new`或 `malloc` 时分配的空间。这部分区域由用户管理，因此容易造成内存泄漏。

### 代码区

也就是 .text 段， 代码区存放程序的二进制代码，它是只读的，以防止程序在运行过程中被意外修改。

```cpp
#include <iostream>
int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
```

比如上面这段代码中的 main 函数，编译为二进制后，函数的逻辑就存放在代码区。

当然这段区域也有可能包含一些只读的常数变量，例如字符串常量等。


### 全局/静态存储区

全局变量和静态变量都存放在全局/静态存储区。以前在 C 语言中全局变量又分为初始化的和未初始化的，分别放在上面图中的 .bss 和 .data 段，但在 C++里面没有这个区分了，他们共同占用同一块内存区，就叫做全局存储区。这个区域的内存在程序的生命周期几乎都是全局的，举例:

```cpp
#include <iostream>
int globalVar = 0; // 全局变量
void function() {
    static int staticVar = 0; // 静态变量
    staticVar++;
    std::cout << staticVar << std::endl;
}
int main() {
    function();
    function();
    return 0;
}
```

`globalVar` 是一个全局变量，`staticVar` 是一个静态变量，它们都存放在全局/静态存储区。

### 栈区

栈区用于存储函数调用时的局部变量、函数参数以及返回地址。

当函数调用完成后，分配给这个函数的栈空间会被释放。例如

```cpp
#include <iostream>
void function(int a, int b) {
    int localVar = a + b;
    std::cout << localVar << std::endl;
}
int main() {
    function(3, 4);
    return 0;
}
```

在这个例子中，a、b和localVar都是局部变量，它们存放在栈区。

当 function 函数调用结束后，对应的函数栈所占用的空间(参数 a、b，局部变量 localVar等)都会被回收。

### 堆区

堆区是用于动态内存分配的区域，当使用new（C++）或者malloc（C）分配内存时，分配的内存块就位于堆区。

我们需要手动释放这些内存，否则可能导致内存泄漏。例如：

```cpp
#include <iostream>
int main() {
    int* dynamicArray = new int[10]; // 动态分配内存
    // 使用动态数组...
    delete[] dynamicArray; // 释放内存
    return 0;
}
```

### 常量区

常量区用于存储常量数据，例如字符串字面量和其他编译时常量。这个区域通常也是只读的。例如：

```cpp
#include <iostream>
int main() {
	char* c="abc";  // abc在常量区，c在栈上。
  return 0;
}
```

> [!NOTE] 总结：代码和数据是分开存储的；堆和栈的不同区别；全局变量和局部变量的存储区别

## C++指针与引用区别

指针和引用在 C++ 中都用于间接访问变量，但它们有一些区别

1. 指针是一个变量，它保存了另一个变量的内存地址；引用是另一个变量的别名，与原变量共享内存地址。

2. 指针(除指针常量)可以被重新赋值，指向不同的变量；引用在初始化后不能更改，始终指向同一个变量。

3. 指针可以为 nullptr，表示不指向任何变量；引用必须绑定到一个变量，不能为 nullptr。

4. 使用指针需要对其进行解引用以获取或修改其指向的变量的值；引用可以直接使用，无需解引用。

> [!TIP] 指针与引用在汇编层面完全等价

![image](https://cdn.ipfsscan.io/weibo/large/005wRZF3ly1i4duk3y754j317w0twajr.jpg)

1. 引用只是C++语法糖，可以看作编译器自动完成取地址、解引用的指针常量
2. 引用区别于指针的特性都是编译器约束完成的，一旦编译成汇编就喝指针一样
3. 由于引用只是指针包装了下，所以也存在风险，比如如下代码:

```cpp
int *a = new int;
int &b = *a;
delete a;
b = 12;    // 对已经释放的内存解引用
```

4. 引用由编译器保证初始化，使用起来较为方便(如不用检查空指针等)
5. 尽量用引用代替指针
6. 引用没有顶层const即int & const，因为引用本身就不可变，所以在加顶层const也没有意义； 但是可以有底层const即 const int&，这表示引用所引用的对象本身是常量
7. 指针既有顶层const(int * const--指针本身不可变)，也有底层const(const int *--指针所指向的对象不可变)
8. 有指针引用--是引用，绑定到指针， 但是没有引用指针--这很显然，因为很多时候指针存在的意义就是间接改变对象的值，但是引用本身的值我们上面说过了是所引用对象的地址，但是引用不能更改所引用的对象，也就当然不能有引用指针了。
9. 指针和引用的自增（`++`）和自减含义不同，指针是指针运算, 而引用是代表所指向的对象对象执行++或--


## C++指针传递、值传递、引用传递

在 C++ 中，函数参数传递有三种常见的方式：值传递、引用传递和指针传递。以下分别给出这三种方式的示例：

### 值传递

值传递是将实参的值传递给形参。在这种情况下，函数内对形参的修改不会影响到实参。

### 引用传递

引用传递是将实参的引用传递给形参。在这种情况下，函数内对形参的修改会影响到实参。

```cpp
#include <iostream>

void swap_reference(int &a, int &b) {
    int temp = a;
    a = b;
    b = temp;
}

int main() {
    int x = 10;
    int y = 20;
    swap_reference(x, y);
    std::cout << "x: " << x << ", y: " << y << std::endl; // 输出：x: 20, y: 10
    return 0;
}
```

### 指针传递

指针传递是将实参的地址传递给形参。在这种情况下，函数内对形参的修改会影响到实参。

```cpp
#include <iostream>

void swap_pointer(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

int main() {
    int x = 10;
    int y = 20;
    swap_pointer(&x, &y);
    std::cout << "x: " << x << ", y: " << y << std::endl; // 输出：x: 20, y: 10
    return 0;
}
```

## C++ RAII

RAII 即 Resource Acquisition Is Initialization，资源获取即初始化。是一种 C++ 编程技术，它将在使用前获取（分配的堆内存、执行线程、打开的套接字、打开的文件、锁定的互斥量、磁盘空间、数据库连接等有限资源）的资源的生命周期与某个对象的生命周期绑定在一起。确保在控制对象的生命周期结束时，按照资源获取的相反顺序释放所有资源。同样，如果资源获取失败（构造函数退出并带有异常），则按照初始化的相反顺序释放所有已完全构造的成员和基类子对象所获取的资源。这利用了核心语言特性（对象生命周期、作用域退出、初始化顺序和堆栈展开），以消除资源泄漏并确保异常安全。

> [!NOTE] 核心思想：利用栈上局部变量的自动析构来保证资源一定会被释放

**实现步骤**

- 设计一个类封装资源，资源可以是内存、文件、socket、锁等等一切
- 在构造函数中执行资源的初始化，比如申请内存、打开文件、申请锁
- 在析构函数中执行销毁操作，比如释放内存、关闭文件、释放锁
- 使用时声明一个该对象的类，一般在你希望的作用域声明即可，比如在函数开始，或者作为类的成员变量

下面写一个 RAII 示例，用 RAII 思想包装 mutex:

```cpp
#include <iostream>
#include <mutex>
#include <thread>

class LockGuard {
public:
    explicit LockGuard(std::mutex &mtx) : mutex_(mtx) {
        mutex_.lock();
    }

    ~LockGuard() {
        mutex_.unlock();
    }

    // 禁止复制
    LockGuard(const LockGuard &) = delete;
    LockGuard &operator=(const LockGuard &) = delete;

private:
    std::mutex &mutex_;
};

// 互斥量
std::mutex mtx;
// 多线程操作的变量
int shared_data = 0;

void increment() {
    for (int i = 0; i < 10000; ++i) {
        // 申请锁
        LockGuard lock(mtx);
        ++shared_data;
        // 作用域结束后会析构 然后释放锁
    }
}

int main() {
    std::thread t1(increment);
    std::thread t2(increment);

    t1.join();
    t2.join();

    std::cout << "Shared data: " << shared_data << std::endl;

    return 0;
}
```

上面定义了一个 LockGuard 类，该类在构造函数中接收一个互斥量（mutex）引用并对其进行锁定，在析构函数中对互斥量进行解锁。这样，我们可以将互斥量传递给 LockGuard 对象，并在需要保护的代码块内创建该对象，确保在执行保护代码期间始终正确锁定和解锁互斥量。在 main 函数中，用两个线程同时更新一个共享变量，通过 RAII 包装的 LockGuard 确保互斥量的正确使用。




## 智能指针

智能指针分为三类：`shared_ptr`，`unique_ptr`，`weak_ptr`（c98还引入了 `auto_ptr`，但已在 c++11中被废弃）

#### `unique_ptr`

`unique_ptr` 表示专属所有权，用 `unique_ptr` 管理的内存，只能被一个对象持有。故不支持**复制**和**赋值**。

```cpp
auto w = std::make_unique<Widget>();  // 在 c++14 中，可以用 make_unique 方法来构造。
auto w2 = w; // 编译错误
```

因此只能通过移动来更改专属所有权：

```cpp
auto w = std::make_unique<Widget>();
auto w2 = std::move(w); // w2 获得内存所有权，w 此时等于 nullptr
```

用法：需要引入头文件 `<memory>`，可以使用右值拷贝构造或 make 方法来构造指针。

```cpp
unique_ptr<int> p1 = make_unique<int>(100);
unique_ptr<string> ps1(new string("good luck"));
```

适用场景

1. 忘记 `delete`

```cpp
class Box {
public:
    Box() : w(new Widget()) {}

    ~Box() {
      // 析构函数中忘记 delete w
    }
private:
    Widget* w;
};
```

2. 异常安全

```cpp
void process() {
    Widget* w = new Widget();
    w->do_something(); // 如果发生异常，那么 delete w 将不会执行，此时就会发生内存泄露
    delete w;  // 也可以用 try...catch 块捕捉异常，并在 catch 语句中 delete，但是不太美观 + 容易漏写
}
```


#### `shared_ptr`

`shared_ptr` 代表的是共享所有权，即多个 `shared_ptr` 可以共享同一块内存。`shared_ptr` 内部是利用引用计数来实现内存的自动管理，每当复制一个 `shared_ptr`，引用计数会 + 1。当一个 `shared_ptr` 离开作用域时，引用计数会 - 1。当引用计数为 0 的时候，则 `delete` 内存。

```cpp
auto w = std::make_shared<Widget>();
auto w2 = w;
cout << w.use_count() << endl;  // g++ -std=c++11 main main.cc  output->2
```

同时，`shared_ptr `也支持移动。从语义上来看，移动指的是所有权的传递。如下：

```cpp
auto w = std::make_shared<Widget>();
auto w2 = std::move(w); // 此时 w 等于 nullptr，w2.use_count() 等于 1
```

> [!NOTE]
> - `shared_ptr` 性能开销更大，几乎是 `unique_ptr` 的两倍（因为还要维护一个计数）
> - 考虑到线程安全问题，引用计数的增减必须是原子操作。而原子操作一般情况下都比非原子操作慢
> - 使用移动优化性能，尽量使用 `std::move` 来将 `shared_ptr` 转移给新对象。因为移动不用增加引用计数，性能更好

使用场景：通常用于指定，有可能多个对象同时管理一个内存的时候。

#### `weak_ptr`

`weak_ptr` 是为了解决 `shared_ptr` 双向引用的问题。即：

```cpp
class B;
struct A {
    shared_ptr<B> b;
};
struct B {
    shared_ptr<A> a;
};
auto pa = make_shared<A>();
auto pb = make_shared<B>();
pa->b = pb;
pb->a = pa;
```

`pa` 和 `pb` 存在着循环引用，根据 `shared_ptr` 引用计数的原理，`pa` 和 `pb` 都无法被正常的释放。
对于这种情况, 我们可以使用 `weak_ptr`：

```cpp
class B;
struct A {
    shared_ptr<B> b;
};
struct B {
    weak_ptr<A> a;
};
auto pa = make_shared<A>();
auto pb = make_shared<B>();
pa->b = pb;
pb->a = pa;
```

`weak_ptr` 不会增加引用计数，因此可以打破 `shared_ptr` 的循环引用。
通常做法是 parent 类持有 child 的 `shared_ptr`, child 持有指向 parent 的 `weak_ptr`。这样也更符合语义。

#### 实现过程

鉴于看到面经中有同学被问到过智能指针的底层实现，因此这里给出两种智能指针（`weak_ptr`略）的简单实现方式。

> [!TIPS]
> 1. `this` 本身是一个指针，指向该类实例化后的对象本身；`*this`表示解引用，C++中对一个指针进行解引用，得到的是当前对象的引用，也就是对象本身
> 2. 注意这里 `(*this->_count) ++ ` 的用法
> 3. 注意这里 `= delete` 的语法，用于显示地**禁用**特定的函数

`shared_ptr`：

```cpp
template<typename T>
class shared_ptr {
private:
    T* _ptr;
    int* _count;  // 引用计数

public:
    // 构造函数
    shared_ptr(T* ptr = nullptr) : _ptr(ptr) {
        if (_ptr) _count = new int(1);
        else _count = new int(10);
    }

    // 拷贝构造
    shared_ptr(const shared_ptr& ptr) {
        if (this != ptr) {
            this->_ptr = ptr._ptr;
            this->_count = ptr._count;
            (*this->_count) ++ ;
        }
    }

    // 重载operator=
    shared_ptr& operator=(const shared_ptr & ptr) {
        if (this->_ptr == ptr._ptr) {
            return *this;
        }

        if (this->_ptr) {
            (*this->_count) -- ;
            if (*this->_count == 0) {
                delete this->_ptr;
                delete this->_count;
            }
        }
        this->_ptr = ptr._ptr;
        this->_count = ptr._count;
        (*this->_count) ++ ;
        return *this;
    }

    // operator*重载
    T& operator*() {
        if (this->_ptr) {
            return *(this->_ptr);
        }
    }

    // operator->重载
    T* operator->() {
        if (this->_ptr) {
            return this->_ptr;
        }
    }

    // 析构函数
    ~shared_ptr() {
        (*this->_count) -- ;
        if (*this->_count == 0) {
            delete this->_ptr;
            delete this->_count;
        }
    }

    // 返回引用计数
    int use_count() {
        return *this->_count;
    }
};
```

`unique_ptr`：

```cpp
template<typename T>
class unique_ptr {
private:
    T* _ptr;

public:
    // 构造函数
    unique_ptr(T* ptr = nullptr) : _ptr(ptr) {}
    // 析构函数
    ~unique_ptr() { del() };

    // 先释放资源（如果持有），再持有资源
    void reset(T* ptr) {
        del();
        _ptr = ptr;
    }

    // 返回资源，资源的释放由调用方处理
    T* release() {
        T* ptr = _ptr;
        _ptr = nullptr;
        return ptr;
    }

    // 获取资源，调用方应该只使用不释放，否则会两次delete资源
    T* get() {
        return _ptr;
    }

private:
    // 释放
    void del() {
        if (_ptr == nullptr) return;
        delete _ptr;
        _ptr = nullptr;
    }

    // 禁用拷贝构造
    unique_ptr(const unique_ptr &) = delete;

    // 禁用拷贝赋值
    unique_ptr& operator = (const unique_ptr &) = delete;
};
```

## malloc-free 内存分配原理

关于动态内存管理，主要就是两块内容：

- ==虚拟内存机制：物理和虚拟地址空间、TLB 页表、内存映射==
- ==动态内存管理：内存管理、分配方式、内存回收、GC等等==

> [!NOTE]
> 1. malloc() 分配的是==虚拟内存==

主要通过两种方式：

1. step1：从内存池中分配。若所需要内存 < 128KB，则从内存池中尝试分配。若无，则进行 `brk` 系统调用，从堆上申请内存。

![image](https://tvax2.sinaimg.cn/large/005wRZF3ly1i4dvbpdeq7j30zr0n90xc.jpg)

2. step2：若 > 128KB，不看内存池，直接使用 `mmap` 系统调用，从文件映射区（同时还存放动态库）中获得内存。

![image](https://tvax4.sinaimg.cn/large/005wRZF3ly1i4dvbqe053j31150n9q7y.jpg)

### free 释放内存，会归还给操作系统吗？

1. malloc 通过 brk() 方式申请的内存，free 释放内存的时候，并不会把内存归还给操作系统，而是缓存在 malloc 的内存池中，待下次使用；
2. malloc 通过 mmap() 方式申请的内存，free 释放内存的时候，会把内存归还给操作系统，内存得到真正的释放。

### 为什么不全部使用 mmap 来分配内存？

因为向操作系统申请内存，是要通过系统调用的，执行系统调用是要进入内核态的，然后在回到用户态，运行态的切换会耗费不少时间。

所以，申请内存的操作应该避免频繁的系统调用，如果都用 mmap 来分配内存，等于每次都要执行系统调用。

另外，因为 mmap 分配的内存每次释放的时候，都会归还给操作系统，于是每次 mmap 分配的虚拟地址都是缺页状态的，然后在第一次访问该虚拟地址的时候，就会触发缺页中断。

也就是说，频繁通过 mmap 分配的内存话，不仅每次都会发生运行态的切换，还会发生缺页中断（在第一次访问虚拟地址后），这样会导致 CPU 消耗较大。

### 为什么不全部使用 brk 来分配内存？

前面我们提到通过 brk 从堆空间分配的内存，并不会归还给操作系统，那么我们那考虑这样一个场景。

如果我们连续申请了 10k，20k，30k 这三片内存，如果 10k 和 20k 这两片释放了，变为了空闲内存空间（不会归还给操作系统），如果下次申请的内存小于 30k，那么就可以重用这个空闲内存空间。

![image](https://tvax4.sinaimg.cn/large/005wRZF3ly1i4dvk3jc99j30mc0j8778.jpg)

但是如果下次申请的内存大于 30k，没有可用的空闲内存空间，必须向 OS 申请，实际使用内存继续增大。

因此，随着系统频繁地 malloc 和 free，尤其对于小块内存，堆内将产生越来越多不可用的碎片，导致“内存泄露”。而这种“泄露”现象使用 valgrind 是无法检测出来的。

所以，malloc 实现中，充分考虑了 brk 和 mmap 行为上的差异及优缺点，默认分配大块内存 (128KB) 才使用 mmap 分配内存空间。

### free() 函数只传入一个内存地址，为什么能知道要释放多大的内存？

malloc 返回给用户态的内存起始地址比进程的堆空间起始地址多了 16 字节，这样当执行 free() 函数时，free 会对传入进来的内存地址向左偏移 16 字节，然后从这个 16 字节的分析出当前的内存块的大小，自然就知道要释放多大的内存了。

![image](https://tvax1.sinaimg.cn/large/005wRZF3ly1i4dvl5ly61j30en07kmxv.jpg)




## malloc/free 与 new/delete区别

1. 语法不同：malloc/free是一个C语言的函数，而new/delete是C++的运算符。
2. 分配内存的方式不同：malloc只分配内存，而new会分配内存并且调用对象的构造函数来初始化对象。
3. 返回值不同：malloc返回一个 void 指针，需要自己强制类型转换，而new返回一个指向对象类型的指针。
4. malloc 需要传入需要分配的大小，而 new 编译器会自动计算所构造对象的大小

### 申请的内存所在位置

new 操作符从自由存储区（free store）上为对象动态分配内存空间，而malloc函数从堆上动态分配内存。自由存储区是 C++ 基于new操作符的一个抽象概念，凡是通过 new 操作符进行内存申请，该内存即为自由存储区。而堆是操作系统中的术语，是操作系统所维护的一块特殊内存，用于程序的内存动态分配，C语言使用malloc从堆上分配内存，使用free释放已分配的对应内存。那么自由存储区是否能够是堆（问题等价于new是否能在堆上动态分配内存），这取决于operator new 的实现细节。自由存储区不仅可以是堆，还可以是静态存储区，这都看operator new在哪里为对象分配内存。

### 内存分配失败时返回值

new内存分配失败时，会抛出bac_alloc异常，它不会返回NULL；malloc分配内存失败时返回NULL

```cpp
// malloc
int *a  = (int *)malloc ( sizeof (int ));
if(NULL == a) {
    ...
} else {
    ...
}

// new
try {
    int *a = new int();
} catch (bad_alloc) {
    ...
}  
```

### 是否调用构造函数/析构函数

使用new操作符来分配对象内存时会经历三个步骤：

- 第一步：调用operator new 函数（对于数组是operator new[]）分配一块足够大的，原始的，未命名的内存空间以便存储特定类型的对象。
- 第二步：编译器运行相应的构造函数以构造对象，并为其传入初值。
- 第三步：对象构造完成后，返回一个指向该对象的指针。

使用delete操作符来释放对象内存时会经历两个步骤：

- 第一步：调用对象的析构函数。
- 第二步：编译器调用operator delete(或operator delete[])函数释放内存空间

### 对数组的处理

C++ 提供了 new[] 与 delete[] 来专门处理数组类型:

```cpp
A * ptr = new A[10];//分配10个A对象
```

使用 new[] 分配的内存必须使用 delete[] 进行释放：

```cpp
delete [] ptr;
```

new 对数组的支持体现在它会分别调用构造函数函数初始化每一个数组元素，释放对象时为每个对象调用析构函数。注意 delete[] 要与new[] 配套使用，不然会找出数组对象部分释放的现象，造成内存泄漏。至于 malloc，它并知道你在这块内存上要放的数组还是啥别的东西，反正它就给你一块原始的内存，在给你个内存的地址就完事。所以如果要动态分配一个数组的内存，还需要我们手动自定数组的大小：

```cpp
int * ptr = (int *) malloc( sizeof(int)* 10 );//分配一个10个int元素的数组
```

### new 和 malloc 是否可以相互调用

operator new /operator delete的实现可以基于malloc，而malloc的实现不可以去调用new。下面是编写operator new /operator delete 的一种简单方式，其他版本也与之类似：

```cpp
void * operator new (sieze_t size)
{
    if(void * mem = malloc(size)
        return mem;
    else
        throw bad_alloc();
}
void operator delete(void *mem) noexcept
{
    free(mem);
}
```

### 能够直观地重新分配内存

使用 malloc 分配的内存后，如果在使用过程中发现内存不足，可以使用 realloc 函数进行内存重新分配实现内存的扩充。

realloc 先判断当前的指针所指内存是否有足够的连续空间，如果有，原地扩大可分配的内存地址，并且返回原来的地址指针；

如果空间不够，先按照新指定的大小分配空间，将原有数据从头到尾拷贝到新分配的内存区域，而后释放原来的内存区域。

new 没有这样直观的配套设施来扩充内存。

> [!NOTE] 总结
| 特征 | new/delete | malloc/free |
|:-: |:-: | :-:|
| 分配内存的位置 | 自由存储区 | 堆 |
| 内存分配成功的返回值    |   完整类型指针  |    void*    |
| 内存分配失败的返回值 | 默认抛出异常，为bad_alloc类型 | 返回NULL |
| 分配内存的大小 | 由编译器根据类型计算得出 | 必须显式指定字节数 |
| 处理数组 | 有处理数组的new版本new[] | 需要用户计算数组的大小后进行内存分配 |
| 已分配内存的扩充 | 无法直观地处理 | 使用realloc简单完成 |
| 是否相互调用 | 可以，看具体的operator new/delete实现 |  不可调用new |
|分配内存时内存不足  | 客户能够指定处理函数或重新制定分配器 | 无法通过用户代码进行处理 |
|函数重载  | 允许 |不允许  |
| 构造函数与析构函数 | 调用 | 不调用 |

## C/C++ 内存泄露如何定位、检测以及避免


### 内存泄露是什么？

简单来说就是：在程序中申请了动态内存，却没有释放，如果程序长期运行下去，最终会导致没有内存可供分配。

### 如何检测？

1. 手动检查代码：仔细检查代码中的内存分配和释放，确保每次分配内存后都有相应的释放操作。比如 malloc和free、new和delete是否配对使用了。

2. 使用调试器和工具：有一些工具可以帮助检测内存泄露。例如：

- Valgrind（仅限于Linux和macOS）：Valgrind是一个功能强大的内存管理分析工具，可以检测内存泄露、未初始化的内存访问、数组越界等问题。使用Valgrind分析程序时，只需在命令行中输入valgrind --leak-check=yes your_program即可。
- Visual Studio中的CRT（C Runtime）调试功能：Visual Studio提供了一些用于检测内存泄露的C Runtime库调试功能。例如，_CrtDumpMemoryLeaks函数可以在程序结束时报告内存泄露。
- AddressSanitizer：AddressSanitizer是一个用于检测内存错误的编译器插件，适用于GCC和Clang。要启用AddressSanitizer，只需在编译时添加-fsanitize=address选项。


### 如何避免内存泄露

1. 使用智能指针（C++）：在C++中，可以使用智能指针（如std::unique_ptr和std::shared_ptr）来自动管理内存。这些智能指针在作用域结束时会自动释放所指向的内存，从而降低忘记释放内存或者程序异常导致内存泄露的风险。

2. 异常安全：在C++中，如果程序抛出异常，需要确保在异常处理过程中正确释放已分配的内存。使用try-catch块来捕获异常并在适当的位置释放内存。 或者使用RAII（Resource Acquisition Is Initialization）技术（关于RAII可以看这篇文章: 如何理解RAII，将资源（如内存）的管理与对象的生命周期绑定。

## C/C++ 野指针和空悬指针

野指针（Wild Pointer）和空悬指针（Dangling Pointer）都是指向无效内存的指针，但它们的成因和表现有所不同，区别如下：

### 野指针 (Wild Pointer)

野指针是一个未被初始化或已被释放的指针。

所以它的值是不确定的，可能指向任意内存地址。

访问野指针可能导致未定义行为，如程序崩溃、数据损坏等。

以下是一个野指针的例子：

```cpp
#include <iostream>

int main() {
    int *wild_ptr; // 未初始化的指针，值不确定
    std::cout << *wild_ptr << std::endl; // 访问野指针，可能导致未定义行为
    return 0;
}
```

### 空悬指针（Dangling Pointer）

空悬指针是指向已经被释放（如删除、回收）的内存的指针。

这种指针仍然具有以前分配的内存地址，但是这块内存可能已经被其他对象或数据占用。

访问空悬指针同样会导致未定义行为。

以下是一个空悬指针的例子：


```cpp
#include <iostream>

int main() {
    int *ptr = new int(42);
    delete ptr; // 释放内存

    // 此时，ptr成为一个空悬指针，因为它指向的内存已经被释放
    std::cout << *ptr << std::endl; // 访问空悬指针，可能导致未定义行为
    return 0;
}
```

为了避免野指针和空悬指针引发的问题，我们应该：

1. 在使用指针前对其进行初始化，如将其初始化为nullptr。
2. 在释放指针指向的内存后，将指针设为nullptr，避免误访问已释放的内存。
3. 在使用指针前检查其有效性，确保指针指向合法内存。


## 常见的 C/C++ 内存错误

### 间接引用坏指针


程的虚拟地址空间中有较大的空洞，没有映射到任何有意义的数据。如果我们试图间接引用一个指向这些洞的指针，那么操作系统就会以段异常中止程序。而且，虚拟内存的某些区域是只读的，试图写这些区域将会以保护异常中止这个程序。间接引用坏指针的一个常见示例是经典的 scanf 错误。假设我们想要使用 scanf 从 stdin 读一个整数到一个变量。正确的方法是传递给 scanf 一个格式串和变量的地址：

```cpp
int val;
scanf("%d", &val)
```

在这种情况下，scanf 将把 val 的内容解释为一个地址，并试图将一个字写到这个位置。在最好的情况下，程序立即以异常终止。在最糟糕的情况下，val 的内容对应于虚拟内存的某个合法的读/写区域，于是我们就覆盖了这块内存。


### 读未初始化的内存

虽然 bss 内存位置（诸如未初始化的全局 C 变量）总是被加载器初始化为零，但是对于堆内存却并不是这样的。

一个常见的错误就是假设堆内存被初始化为零：

```cpp
/* Return y = Ax */
int *matvec(int **A, int *x, int n)
{
    int i, j;

    int *y = (int *)Malloc(n * sizeof(int));

    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            y[i] += A[i][j] * x[j];
    return y;
}
```

在这个示例中，程序员不正确地假设向量 y 被初始化为零。正确的实现方式是显式地将 y[i] 设置为零，或者使用 calloc。

### 栈缓冲区溢出

如果一个程序不检查输入串的大小就写入栈中的目标缓冲区，那么这个程序就会有缓冲区溢出错误（buffer overflow bug）。例如，下面的函数就有缓冲区溢出错误，因为 gets 函数复制一个任意长度的串到缓冲区。为了纠正这个错误，必须使用 fgets 函数，这个函数限制了输入串的大小：

```cpp
void bufoverflow()
{
    char buf[64];
    gets(buf); /* Here is the stack buffer overflow bug */
    return;
}
```

### 误解指针运算

另一种常见的错误是忘记了指针的算术操作是以它们指向的对象的大小为单位来进行的，而这种大小単位并不一定是字节。

例如，下面函数的目的是扫描一个 int 的数组，并返回一个指针，指向 val 的首次出现：

```cpp
int *search(int *p, int val)
{
    while (*p && *p != val)
        p += sizeof(int); /* Should be p++ */
    return p;
}
```

### 引用不存在的变量

```cpp
int *stackref ()
{
    int val;

    return &val;
}
```

这个函数返回一个指针（比如说是 p），指向栈里的一个局部变量，然后弹出它的栈帧。 尽管 p 仍然指向一个合法的内存地址，但是它已经不再指向一个合法的变量了。 当以后在程序中调用其他函数时，内存将重用它们的栈帧。再后来，如果程序分配某个值给 *p，那么它可能实际上正在修改另一个函数的栈帧中的一个条目，从而潜在地带来灾难性的、令人困惑的后果。

### 引起内存泄漏

内存泄漏是缓慢、隐性的杀手，当程序员不小心忘记释放已分配块，而在堆里创建了垃圾时，会发生这种问题。例如，下面的函数分配了一个堆块 x，然后不释放它就返回：

```cpp
void leak(int n)
{
    int *x = (int *)Malloc(n * sizeof(int));
    return;  /* x is garbage at this point */
}
```

## C++ nullptr 和 NULL的区别

在 C++11 之前，我们通常使用 NULL 来表示空指针。

然而，在 C++ 中，NULL 的定义实际上是一个整数值 0，而不是一个真正的指针类型。

在函数重载和模板编程中这可能会导致一些问题和歧义。

为了解决这个问题，C++11 引入了一个新的关键字 nullptr，用于表示空指针。

nullptr 是一种特殊类型的字面值，类型为 std::nullptr_t，定义为: typedef decltype(nullptr) nullptr_t，可以隐式转换为任何指针类型。

与 NULL 不同，nullptr 是一个真正的指针类型，因此可以避免一些由于 NULL 是整数类型而引起的问题。

以下是 nullptr 和 NULL 之间区别的一些例子：

1. 函数重载

```cpp
#include <iostream>

void foo(int x) {
    std::cout << "foo() called with an int: " << x << std::endl;
}

void foo(char* x) {
    std::cout << "foo() called with a char*: " << x << std::endl;
}

int main() {
    // foo(NULL); // 编译错误：因为 NULL 会被解析为整数 0，导致二义性
    foo(nullptr); // 无歧义：调用 void foo(char* x)
}
```

2. 函数模版

```cpp
#include <iostream>
#include <type_traits>

template <typename T>
void bar(T x) {
    if (std::is_same<T, std::nullptr_t>::value) {
        std::cout << "bar() called with nullptr" << std::endl;
    } else {
        std::cout << "bar() called with a non-nullptr value" << std::endl;
    }
}

int main() {
    bar(NULL); // 输出：bar() called with a non-nullptr value，因为 NULL 被解析为整数 0
    bar(nullptr); // 输出：bar() called with nullptr
}
```

总之，C++ 11 引入了 nullptr 作为一个更安全、更明确的空指针表示，可以避免与整数 0（即 NULL）相关的一些问题。在 C++11 及以后的代码中，建议使用 nullptr 代替 NULL 表示空指针。
