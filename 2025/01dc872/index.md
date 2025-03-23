# 内存分配


介绍 C&#43;&#43; 中的内存分配、管理以及智能指针的原理及使用

&lt;!--more--&gt;

## C&#43;&#43; 中内存的管理

### 程序地址空间

C&#43;&#43; 中，内存分布分为五块区域，分别是：==栈；堆；全局变量和静态变量（存放于 data 段和 bss 段）；常量；代码；==

![Image](https://github.com/user-attachments/assets/65d4aef7-e5d3-4a5a-90d4-68e8c9081544)

上图是内核和用户的虚拟地址空间分布情况，其中，==局部变量==和==参数==等都存放在==栈==中，这部分空间由系统进行管理；而==堆==中空间主要是用于用户调用 `new`或 `malloc` 时分配的空间。这部分区域由用户管理，因此容易造成内存泄漏。

#### `malloc` 底层实现原理

1. step1：从内存池中分配。若所需要内存 &lt; 128KB，则从内存池中尝试分配。若无，则进行 `brk` 系统调用，从堆上申请内存。
2. step2：若 &gt; 128KB，不看内存池，直接使用 `mmap` 系统调用，从文件映射区（同时还存放动态库）中获得内存。

&gt; [!NOTE] 这里我觉得这种虚拟地址空间划分方式粒度太粗，不足以说明具体情况




### 智能指针

智能指针分为三类：`shared_ptr`，`unique_ptr`，`weak_ptr`（c98还引入了 `auto_ptr`，但已在 c&#43;&#43;11中被废弃）

#### `unique_ptr`

`unique_ptr` 表示专属所有权，用 `unique_ptr` 管理的内存，只能被一个对象持有。故不支持**复制**和**赋值**。

```cpp
auto w = std::make_unique&lt;Widget&gt;();  // 在 c&#43;&#43;14 中，可以用 make_unique 方法来构造。
auto w2 = w; // 编译错误
```

因此只能通过移动来更改专属所有权：

```cpp
auto w = std::make_unique&lt;Widget&gt;();
auto w2 = std::move(w); // w2 获得内存所有权，w 此时等于 nullptr
```

用法：需要引入头文件 `&lt;memory&gt;`，可以使用右值拷贝构造或 make 方法来构造指针。

```cpp
unique_ptr&lt;int&gt; p1 = make_unique&lt;int&gt;(100);
unique_ptr&lt;string&gt; ps1(new string(&#34;good luck&#34;));
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
    w-&gt;do_something(); // 如果发生异常，那么 delete w 将不会执行，此时就会发生内存泄露
    delete w;  // 也可以用 try...catch 块捕捉异常，并在 catch 语句中 delete，但是不太美观 &#43; 容易漏写
}
```


#### `shared_ptr`

`shared_ptr` 代表的是共享所有权，即多个 `shared_ptr` 可以共享同一块内存。`shared_ptr` 内部是利用引用计数来实现内存的自动管理，每当复制一个 `shared_ptr`，引用计数会 &#43; 1。当一个 `shared_ptr` 离开作用域时，引用计数会 - 1。当引用计数为 0 的时候，则 `delete` 内存。

```cpp
auto w = std::make_shared&lt;Widget&gt;();
auto w2 = w;
cout &lt;&lt; w.use_count() &lt;&lt; endl;  // g&#43;&#43; -std=c&#43;&#43;11 main main.cc  output-&gt;2 
```

同时，`shared_ptr `也支持移动。从语义上来看，移动指的是所有权的传递。如下：

```cpp
auto w = std::make_shared&lt;Widget&gt;();
auto w2 = std::move(w); // 此时 w 等于 nullptr，w2.use_count() 等于 1
```

&gt; [!NOTE]
&gt; - `shared_ptr` 性能开销更大，几乎是 `unique_ptr` 的两倍（因为还要维护一个计数）
&gt; - 考虑到线程安全问题，引用计数的增减必须是原子操作。而原子操作一般情况下都比非原子操作慢
&gt; - 使用移动优化性能，尽量使用 `std::move` 来将 `shared_ptr` 转移给新对象。因为移动不用增加引用计数，性能更好

使用场景：通常用于指定，有可能多个对象同时管理一个内存的时候。

#### `weak_ptr` 

`weak_ptr` 是为了解决 `shared_ptr` 双向引用的问题。即：

```cpp
class B;
struct A {
    shared_ptr&lt;B&gt; b;
};
struct B {
    shared_ptr&lt;A&gt; a;
};
auto pa = make_shared&lt;A&gt;();
auto pb = make_shared&lt;B&gt;();
pa-&gt;b = pb;
pb-&gt;a = pa;
```

`pa` 和 `pb` 存在着循环引用，根据 `shared_ptr` 引用计数的原理，`pa` 和 `pb` 都无法被正常的释放。
对于这种情况, 我们可以使用 `weak_ptr`：

```cpp
class B;
struct A {
    shared_ptr&lt;B&gt; b;
};
struct B {
    weak_ptr&lt;A&gt; a;
};
auto pa = make_shared&lt;A&gt;();
auto pb = make_shared&lt;B&gt;();
pa-&gt;b = pb;
pb-&gt;a = pa;
```

`weak_ptr` 不会增加引用计数，因此可以打破 `shared_ptr` 的循环引用。
通常做法是 parent 类持有 child 的 `shared_ptr`, child 持有指向 parent 的 `weak_ptr`。这样也更符合语义。

### 实现过程

鉴于看到面经中有同学被问到过智能指针的底层实现，因此这里给出两种智能指针（`weak_ptr`略）的简单实现方式。

&gt; [!TIPS]
&gt; 1. `this` 本身是一个指针，指向该类实例化后的对象本身；`*this`表示解引用，C&#43;&#43;中对一个指针进行解引用，得到的是当前对象的引用，也就是对象本身
&gt; 2. 注意这里 `(*this-&gt;_count) &#43;&#43; ` 的用法
&gt; 3. 注意这里 `= delete` 的语法，用于显示地**禁用**特定的函数

`shared_ptr`：

```cpp
template&lt;typename T&gt;
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
    shared_ptr(const shared_ptr&amp; ptr) {
        if (this != ptr) {
            this-&gt;_ptr = ptr._ptr;
            this-&gt;_count = ptr._count;
            (*this-&gt;_count) &#43;&#43; ;
        }
    }

    // 重载operator=
    shared_ptr&amp; operator=(const shared_ptr &amp; ptr) {
        if (this-&gt;_ptr == ptr._ptr) {
            return *this;
        }

        if (this-&gt;_ptr) {
            (*this-&gt;_count) -- ;
            if (*this-&gt;_count == 0) {
                delete this-&gt;_ptr;
                delete this-&gt;_count;
            }
        }
        this-&gt;_ptr = ptr._ptr;
        this-&gt;_count = ptr._count;
        (*this-&gt;_count) &#43;&#43; ;
        return *this;
    }

    // operator*重载
    T&amp; operator*() {
        if (this-&gt;_ptr) {
            return *(this-&gt;_ptr);
        }
    }

    // operator-&gt;重载
    T* operator-&gt;() {
        if (this-&gt;_ptr) {
            return this-&gt;_ptr;
        }
    }

    // 析构函数
    ~shared_ptr() {
        (*this-&gt;_count) -- ;
        if (*this-&gt;_count == 0) {
            delete this-&gt;_ptr;
            delete this-&gt;_count;
        }
    }

    // 返回引用计数
    int use_count() {
        return *this-&gt;_count;
    }
};
```

`unique_ptr`：

```cpp
template&lt;typename T&gt;
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
    unique_ptr(const unique_ptr &amp;) = delete; 

    // 禁用拷贝赋值
    unique_ptr&amp; operator = (const unique_ptr &amp;) = delete;
};
```

---

> 作者:   
> URL: https://walker-ai.github.io/2025/01dc872/  

