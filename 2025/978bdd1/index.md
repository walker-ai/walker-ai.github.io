# C&#43;&#43;语法及面向对象特性


记录准备面试过程中看到的一些 C&#43;&#43; 的常用语法及特性

&lt;!--more--&gt;

### `const` 和 `constexpr` 区别

`const` 并未区分编译期常量和运行期常量，并且 `const` 只保证了运行时不直接被修改，而 `constexpr` 是限定在了编译器常量。所以 `constexpr` 相当于是把 `const` 的职责拆出来一部分，`const` 只做只读语义的保证，而常量语义交给了 `constexpr` 负责。

### 左值、右值与左值引用和右值引用

左值指既能够出现在等号左边，也能出现在等号右边的变量（可以取地址）；右值则是只能出现在等号右边的变量（不可取地址）。

- 左值是可寻址的变量，有持久性；
- 右值一般是不可寻址的常量，或在表达式求值过程中创建的无名临时对象，短暂性的。
- 左值引用就是对左值的引用，给左值取别名，避免对象拷贝
- 右值引用就是对右值的引用，给右值取别名。主要作用是把延长对象的生命周期，一般是延长到作用域之外

### 字节序—大端序与小端序

字节序是指在多字节数据类型（如整数、浮点数等）中，字节在内存中的存储顺序。主要有两种字节序：大端字节序（Big-endian）和小端字节序（Little-endian）。

- 大端序：高位字节存储在低地址处，低位字节存储在高地址处。例如，一个4字节的整数0x12345678，在大端字节序的系统中，内存布局如下（从左侧的低地址到右侧的高地址）：`0x12 | 0x34 | 0x56 | 0x78`
- 小端序：低位字节存储在低地址处，高位字节存储在高地址处。例如，一个4字节的整数0x12345678，在小端字节序的系统中，内存布局如下（从左侧的低地址到右侧的高地址）：`0x78 | 0x56 | 0x34 | 0x12`

常见大小端字节序应用场景：

1. 网络传输，通常用大端序，也称网络字节序
2. 操作系统一般主要是小端序

### 字节对齐

理论上，任何类型的变量都可以从任意地址开始存放。然而实际上，访问特定类型的变量通常需要从特定对齐的内存地址开始。如果不对数据存储进行适当的对齐，可能会导致存取效率降低。

例如，有些平台每次读取都是从偶数地址开始。如果一个 int 类型（假设为 32 位系统）存储在偶数地址开始的位置，那么一个读周期就可以读取这 32 位。但如果存储在奇数地址开始的位置，则需要两个读周期，并将两次读取的结果的高低字节拼凑才能得到这 32 位数据。显然这会显著降低读取效率。

==总结： 字节对齐有助于提高内存访问速度，因为许多处理器都优化了对齐数据的访问。但是，这可能会导致内存中的一些空间浪费。==

字节对齐的一些规则：

1. 自然规则对齐：按照数据本身的数据类型进行对齐，例如，char 类型的自然对齐边界为 1 字节，short 为 2 字节，int 和 float 为 4 字节，double 和 64 位指针为 8 字节。具体数值可能因编译器和平台而异。
2. 结构体对齐：结构体内部的每个成员都根据其自然对齐边界进行对齐。也就是可能在成员之间插入填充字节。结构体本身的总大小也会根据其最大对齐边界的成员进行对齐（比如结构体成员包含的最长类型为int类型，那么整个结构体要按照4的倍数对齐），以便在数组中正确对齐。
3. 可以使用编译器指令（如 `#pragma pack`）更改默认的对齐规则。这个命令是全局生效的。这可以用于减小数据结构的大小，但可能会降低访问性能。

### C&#43;&#43;中class和struct区别

C&#43;&#43; 中为了兼容 C 语言而保留了 C 语言的 struct 关键字，并且加以扩充了含义。

在 C 语言中，struct 只能包含成员变量，不能包含成员函数。

而在 C&#43;&#43; 中，struct 类似于 class，既可以包含成员变量，又可以包含成员函数。

#### 不同点

- class 中类中的成员默认都是 private 属性的。

- 而在 struct 中结构体中的成员默认都是 public 属性的。

- class 继承默认是 private 继承，而 struct 继承默认是 public 继承。

- class 可以用于定义模板参数，struct 不能用于定义模板参数。

这样写是正确的：

```cpp
template &lt;class T&gt;
struct  Person {
public:
    T age;
};
```

而这样写是错误的：

```cpp
template &lt;struct T&gt;
struct  Person {
public:
    T age;
};
```

#### 使用习惯

实际使用中，struct 我们通常用来定义一些 POD(plain old data)

在 C&#43;&#43;11 及之后的标准中，POD 类型需要同时满足两个独立条件：

- ​​平凡（Trivial）​​：类型具有默认的构造/拷贝/移动/析构函数（可自动生成且非虚）
- ​​标准布局（Standard Layout）​​：内存布局与 C 兼容，成员排列顺序符合特定规则

同时满足平凡性和标准布局的类型称为 POD 类型，这类数据可以安全使用 memcpy 等底层内存操作，因为它们的内存布局与 C 完全兼容且没有特殊处理需求。


### C&#43;&#43;四种强制类型转换

&gt; [!NOTE] B站一面面试题

#### static_cast

用法：`static_cast&lt;new_type&gt;(expression)`

其实static_cast 和 C 语言 () 做强制类型转换基本是等价的。

主要用于以下场景:

1. 基本类型之间的转换

将一个基本类型转换为另一个基本类型，例如将整数转换为浮点数或将字符转换为整数。

```cpp
int a = 42;
double b = static_cast&lt;double&gt;(a); // 将整数a转换为双精度浮点数b
```

2. 指针类型之间的转换

将一个指针类型转换为另一个指针类型，尤其是在类层次结构中从基类指针转换为派生类指针。这种转换不执行运行时类型检查，可能不安全，要自己保证指针确实可以互相转换。

```cpp
class Base {};
class Derived : public Base {};

Base* base_ptr = new Derived();
Derived* derived_ptr = static_cast&lt;Derived*&gt;(base_ptr); // 将基类指针base_ptr转换为派生类指针derived_ptr
```

3. 引用类型之间的转换

类似于指针类型之间的转换，可以将一个引用类型转换为另一个引用类型。在这种情况下，也应注意安全性。

```cpp
Derived derived_obj;
Base&amp; base_ref = derived_obj;
Derived&amp; derived_ref = static_cast&lt;Derived&amp;&gt;(base_ref); // 将基类引用base_ref转换为派生类引用derived_ref
```

static_cast在编译时执行类型转换，在进行指针或引用类型转换时，需要自己保证合法性。

如果想要运行时类型检查，可以使用dynamic_cast进行安全的向下类型转换。

#### dynamic_cast

用法: `dynamic_cast&lt;new_type&gt;(expression)`

dynamic_cast在C&#43;&#43;中主要应用于父子类层次结构中的安全类型转换。它在运行时执行类型检查，因此相比于static_cast，它更加安全。dynamic_cast的主要应用场景：

1. 向下类型转换

当需要将基类指针或引用转换为派生类指针或引用时，dynamic_cast可以确保类型兼容性。

如果转换失败，dynamic_cast将返回空指针（对于指针类型）或抛出异常（对于引用类型）。

```cpp
class Base { virtual void dummy() {} };
class Derived : public Base { int a; };

Base* base_ptr = new Derived();
Derived* derived_ptr = dynamic_cast&lt;Derived*&gt;(base_ptr); // 将基类指针base_ptr转换为派生类指针derived_ptr，如果类型兼容，则成功
```

2. 用于多态类型检查

处理多态对象时，dynamic_cast可以用来确定对象的实际类型，例如：

```cpp
class Animal { public: virtual ~Animal() {} };
class Dog : public Animal { public: void bark() { /* ... */ } };
class Cat : public Animal { public: void meow() { /* ... */ } };

Animal* animal_ptr = /* ... */;

// 尝试将Animal指针转换为Dog指针
Dog* dog_ptr = dynamic_cast&lt;Dog*&gt;(animal_ptr);
if (dog_ptr) {
    dog_ptr-&gt;bark();
}

// 尝试将Animal指针转换为Cat指针
Cat* cat_ptr = dynamic_cast&lt;Cat*&gt;(animal_ptr);
if (cat_ptr) {
    cat_ptr-&gt;meow();
}
```

另外，要使用dynamic_cast有效，基类至少需要一个虚拟函数。

因为，dynamic_cast只有在基类存在虚函数(虚函数表)的情况下才有可能将基类指针转化为子类。

3. dynamic_cast 底层原理

dynamic_cast的底层原理依赖于运行时类型信息（RTTI, Runtime Type Information）。

C&#43;&#43;编译器在编译时为支持多态的类生成RTTI，它包含了类的类型信息和类层次结构。

我们都知道当使用虚函数时，编译器会为每个类生成一个虚函数表（vtable），并在其中存储指向虚函数的指针。

伴随虚函数表的还有 RTTI(运行时类型信息)，这些辅助的信息可以用来帮助我们运行时识别对象的类型信息。

《深度探索C&#43;&#43;对象模型》中有个例子：

```cpp
class Point
{
public:
	Point(float xval);
	virtual ~Point();

	float x() const;
	static int PointCount();

protected:
	virtual ostream&amp; print(ostream&amp; os) const;

	float _x;
	static int _point_count;
};
```


![image](https://cdn.ipfsscan.io/weibo/large/005wRZF3ly1i4g7emk7ybj30xn0i0ad4.jpg)

首先，每个多态对象都有一个指向其vtable的指针，称为vptr。

RTTI（就是上面图中的 type_info 结构)通常与vtable关联。

dynamic_cast就是利用RTTI来执行运行时类型检查和安全类型转换。

以下是dynamic_cast的工作原理的简化描述：

1. 首先，dynamic_cast通过查询对象的 vptr 来获取其RTTI（这也是为什么 dynamic_cast 要求对象有虚函数）

2. 然后，dynamic_cast比较请求的目标类型与从RTTI获得的实际类型。如果目标类型是实际类型或其基类，则转换成功。

3. 如果目标类型是派生类，dynamic_cast会检查类层次结构，以确定转换是否合法。如果在类层次结构中找到了目标类型，则转换成功；否则，转换失败。

4. 当转换成功时，dynamic_cast返回转换后的指针或引用。

5. 如果转换失败，对于指针类型，dynamic_cast返回空指针；对于引用类型，它会抛出一个std::bad_cast异常。

因为dynamic_cast依赖于运行时类型信息，它的性能可能低于其他类型转换操作（如static_cast），static 是编译器静态转换，编译时期就完成了。

#### const_cast

用法: `const_cast&lt;new_type&gt;(expression)`

new_type 必须是一个指针、引用或者指向对象类型成员的指针。

1. 修改const对象

当需要修改const对象时，可以使用const_cast来删除const属性。

```cpp
const int a = 42;
int* mutable_ptr = const_cast&lt;int*&gt;(&amp;a); // 删除const属性，使得可以修改a的值
*mutable_ptr = 43; // 修改a的值
```

2. const对象调用非const成员函数

当需要使用const对象调用非const成员函数时，可以使用const_cast删除对象的const属性。

```cpp
class MyClass {
public:
    void non_const_function() { /* ... */ }
};

const MyClass my_const_obj;
MyClass* mutable_obj_ptr = const_cast&lt;MyClass*&gt;(&amp;my_const_obj); // 删除const属性，使得可以调用非const成员函数
mutable_obj_ptr-&gt;non_const_function(); // 调用非const成员函数
```

不过上述行为都不是很安全，可能导致未定义的行为，因此应谨慎使用。

#### reinterpret_cast

用法: `reinterpret_cast&lt;new_type&gt;(expression)`

reinterpret_cast用于在不同类型之间进行低级别的转换。

首先从英文字面的意思理解，interpret是“解释，诠释”的意思，加上前缀“re”，就是“重新诠释”的意思；

cast 在这里可以翻译成“转型”（在侯捷大大翻译的《深度探索C&#43;&#43;对象模型》、《Effective C&#43;&#43;（第三版）》中，cast都被翻译成了转型），这样整个词顺下来就是“重新诠释的转型”。

它仅仅是重新解释底层比特（也就是对指针所指针的那片比特位换个类型做解释），而不进行任何类型检查。

因此，reinterpret_cast可能导致未定义的行为，应谨慎使用。

reinterpret_cast的一些典型应用场景：

1. 指针类型之间的转换

在某些情况下，需要在不同指针类型之间进行转换，如将一个int指针转换为char指针。

这在 C 语言中用的非常多，C语言中就是直接使用 () 进行强制类型转换

```cpp
int a = 42;
int* int_ptr = &amp;a;
char* char_ptr = reinterpret_cast&lt;char*&gt;(int_ptr); // 将int指针转换为char指针
```

&gt; [!TIP] 其实在 CUDA 中能经常见到 reinterpret_cast 的使用，例如这里的向量化加载：
&gt; ```CUDA
&gt; #define INT4(value) (reinterpret_cast&lt;int4 *&gt;(&amp;(value))[0])
&gt; #define FLOAT4(value) (reinterpret_cast&lt;float4 *&gt;(&amp;(value))[0])
&gt; #define HALF2(value) (reinterpret_cast&lt;half2 *&gt;(&amp;(value))[0])
&gt; #define BFLOAT2(value) (reinterpret_cast&lt;__nv_bfloat162 *&gt;(&amp;(value))[0])
&gt; #define LDST128BITS(value) (reinterpret_cast&lt;float4 *&gt;(&amp;(value))[0])
&gt; ```

### 面向对象特性

#### 封装

实现一个class，将数据属性，方法等集成到一个类中的过程，隐藏内部实现细节，仅暴露接口给外部

#### 继承

一个类从另外一个类中获得属性与方法的过程。通过创建具有共享代码的类层次结构，减少重复代码，提高代码复用性和可维护性。

#### 多态

多态是允许不同类的对象使用相同的接口名字，但具有不同实现的特性。在 C&#43;&#43; 中，多态主要通过虚函数（Virtual Function）和抽象基类（Abstract Base Class）来实现。虚函数允许在派生类中重写基类的方法，而抽象基类包含至少一个纯虚函数（Pure Virtual Function），不能被实例化，只能作为其他派生类的基类。

### 重载、重写和隐藏

#### 重载

重载是指相同作用域(比如命名空间或者同一个类)内拥有相同的方法名，但具有不同的参数类型和/或参数数量的方法。 重载允许根据所提供的参数不同来调用不同的函数。它主要在以下情况下使用：

- 方法具有相同的名称。
- 方法具有不同的参数类型或参数数量。
- 返回类型可以相同或不同。
- 同一作用域，比如都是一个类的成员函数，或者都是全局函数

#### 重写

重写是指在派生类中重新定义基类中的方法。当派生类需要改变或扩展基类方法的功能时，就需要用到重写。重写的条件包括：

- 方法具有相同的名称。
- 方法具有相同的参数类型和数量。
- 方法具有相同的返回类型。
- 重写的基类中被重写的函数必须有virtual修饰。重
- ==写主要在继承关系的类之间发生。==

#### 隐藏

隐藏是指派生类的函数屏蔽了与其同名的基类函数。注意只要同名函数，不管参数列表是否相同，基类函数都会被隐藏。

### 类初始化顺序

#### 基类初始化顺序

如果当前类继承自一个或多个基类，它们将按照声明顺序进行初始化，但是在有虚继承和一般继承存在的情况下，优先虚继承。比如虚继承：class MyClass : public Base1, public virtual Base2，此时应当先调用 Base2 的构造函数，再调用 Base1 的构造函数。


#### 成员变量初始化顺序

类的成员变量按照它们在类定义中的声明顺序进行初始化

#### 执行构造函数

在基类和成员变量初始化完成后，执行类的构造函数。


### 类的析构顺序

==记住一点即可，类的析构顺序和构造顺序完全相反==


### 析构函数中可以抛出异常吗？

==可以但不建议==

由于析构函数常常被自动调用，在析构函数中抛出的异常往往会难以捕获，引发程序非正常退出或未定义行为。另外，我们都知道在容器析构时，会逐个调用容器中的对象析构函数，而某个对象析构时抛出异常还会引起后续的对象无法被析构，导致资源泄漏。

### 深拷贝和浅拷贝

C&#43;&#43;中的深拷贝和浅拷贝涉及到对象的复制。

当对象包含指针成员时，这两种拷贝方式的区别变得尤为重要。

#### 浅拷贝

浅拷贝是一种简单的拷贝方式，它仅复制对象的基本类型成员和指针成员的值，而不复制指针所指向的内存。

这可能导致两个对象共享相同的资源，从而引发潜在的问题，如内存泄漏、意外修改共享资源等。

#### 深拷贝

深拷贝不仅复制对象的基本类型成员和指针成员的值，还复制指针所指向的内存。

因此，两个对象不会共享相同的资源，避免了潜在问题。

深拷贝通常需要显式实现拷贝构造函数和赋值运算符重载。

### C&#43;&#43; 多态实现方式

C&#43;&#43;实现多态的方法主要包括==虚函数、纯虚函数和模板函数==，其中虚函数、纯虚函数实现的多态叫**动态多态**，模板函数、重载等实现的叫**静态多态**。

区分静态多态和动态多态的一个方法就是看决定所调用的具体方法是在编译期还是运行时，运行时就叫动态多态。

#### 虚函数、纯虚函数实现多态

虚函数是指在基类中声明的函数，它在派生类中可以被重写。当我们使用基类指针或引用指向派生类对象时，通过虚函数的机制，可以调用到派生类中重写的函数，从而实现多态。

C&#43;&#43; 的多态必须满足两个条件：

1. 必须通过基类的指针或者引用调用虚函数
2. 被调用的函数是虚函数，且必须完成对基类虚函数的重写

```cpp
class Shape {
   public:
      virtual int area() = 0;
};

class Rectangle: public Shape {
   public:
      int area () {
         cout &lt;&lt; &#34;Rectangle class area :&#34;;
         return (width * height);
      }
};

class Triangle: public Shape{
   public:
      int area () {
         cout &lt;&lt; &#34;Triangle class area :&#34;;
         return (width * height / 2);
      }
};

int main() {
   Shape *shape;
   Rectangle rec(10,7);
   Triangle  tri(10,5);

   shape = &amp;rec;
   shape-&gt;area();

   shape = &amp;tri;
   shape-&gt;area();

   return 0;
}
```

#### 模板函数多态

模板函数可以根据传递参数的不同类型，自动生成相应类型的函数代码。模板函数可以用来实现多态。

```cpp
template &lt;class T&gt;
T GetMax (T a, T b) {
   return (a&gt;b?a:b);
}

int main () {
   int i=5, j=6, k;
   long l=10, m=5, n;
   k=GetMax&lt;int&gt;(i,j);
   n=GetMax&lt;long&gt;(l,m);
   cout &lt;&lt; k &lt;&lt; endl;
   cout &lt;&lt; n &lt;&lt; endl;
   return 0;
}
```

#### 函数重载多态

见函数重载

### this 指针

#### 关于

1. this 是一个指向当前对象的指针。
2. 其实在面向对象的编程语言中，都存在this指针这种机制， Java、C&#43;&#43; 叫 this，而 Python 中叫 self。
3. 在类的成员函数中访问类的成员变量或调用成员函数时，编译器会隐式地将当前对象的地址作为 this 指针传递给成员函数。

因此，this 指针可以用来访问类的成员变量和成员函数，以及在成员函数中引用当前对象。

#### static 函数不能访问成员变量，因此不可使用 this 指针

static 函数是一种静态成员函数，它与类本身相关，而不是与类的对象相关。因为静态函数没有 this 指针，所以它不能访问任何非静态成员变量。

### 虚函数表

虚函数（Virtual Function）是通过一张虚函数表（Virtual Table）来实现的，简称为V-Table。在这个表中，存放的是一个类的虚函数的地址表，这张表解决了继承、覆盖的问题，保证其真实反应实际的函数。

#### C&#43;&#43; 对象模型

```cpp
class Point {
public:
	Point(float xval );
	virtual ~Point();
	float x() const,
	static int PointCount();
protected:
	virtual ostream&amp; print( ostream &amp;os ) const;
	float _x;
	static int _point_count;
};
```

比如上面这个类，它的对象模型如下：

![image](https://cdn.ipfsscan.io/weibo/large/005wRZF3ly1i4cp1651wgj30ti0jawhc.jpg)

在上面的示例中，意思就是一个对象在内存中一般由成员变量（非静态）、虚函数表指针(vptr)构成。虚函数表指针指向一个数组，数组的元素就是各个虚函数的地址，通过函数的索引，我们就能直接访问对应的虚函数。

#### 动态多态底层原理

当基类指针或引用指向一个派生类对象时，调用虚函数时，实际上会调用派生类中的虚函数，而不是基类中的虚函数。

==在底层，当一个类声明一个虚函数时，编译器会为该类创建一个虚函数表（Virtual Table）==。
这个表存储着该类的虚函数指针，这些指针指向实际实现该虚函数的代码地址。

每个对象都包含一个指向该类的虚函数表的指针，这个指针在对象创建时被初始化，通常是作为对象的第一个成员变量。

当调用一个虚函数时，编译器会通过对象的虚函数指针查找到该对象所属的类的虚函数表，并根据函数的索引值（通常是函数在表中的位置，编译时就能确定）来找到对应的虚函数地址。

然后将控制转移到该地址，实际执行该函数的代码。

对于派生类，其虚函数表通常是在基类的虚函数表的基础上扩展而来的。

==在派生类中，如果重写了基类的虚函数，那么该函数在派生类的虚函数表中的地址会被更新为指向派生类中实际实现该函数的代码地址。==

### C&#43;&#43; 纯虚函数

纯虚函数是一种在基类中声明但没有实现的虚函数。

==它的作用是定义了一种接口，这个接口需要由派生类来实现。==（PS: C&#43;&#43; 中没有接口，纯虚函数可以提供类似的功能

包含纯虚函数的类称为抽象类（Abstract Class）。

抽象类仅仅提供了一些接口，但是没有实现具体的功能。作用就是制定各种接口，通过派生类来实现不同的功能，从而实现代码的复用和可扩展性。

另外，抽象类无法实例化，也就是无法创建对象。原因很简单，纯虚函数没有函数体，不是完整的函数，无法调用，也无法为其分配内存空间。

```cpp
#include &lt;iostream&gt;
using namespace std;

class Shape {
   public:
      // 纯虚函数
      virtual void draw() = 0;
};

class Circle : public Shape {
   public:
      void draw() {
         cout &lt;&lt; &#34;画一个圆形&#34; &lt;&lt; endl;
      }
};

class Square : public Shape {
   public:
      void draw() {
         cout &lt;&lt; &#34;画一个正方形&#34; &lt;&lt; endl;
      }
};

int main() {
   Circle circle;
   Square square;

   Shape *pShape1 = &amp;circle;
   Shape *pShape2 = &amp;square;

   pShape1-&gt;draw();
   pShape2-&gt;draw();

   return 0;
}

/*
在上面的代码中，定义了一个抽象类 Shape，它包含了一个纯虚函数 draw()。
Circle 和 Square 是 Shape 的两个派生类，它们必须实现 draw() 函数，否则它们也会是一个抽象类。
在 main() 函数中，创建了 Circle 和 Square 的实例，并且使用指向基类 Shape 的指针来调用 draw() 函数。
由于 Shape 是一个抽象类，不能创建 Shape 的实例，但是可以使用 Shape 类型指针来指向派生类，从而实现多态。
*/
```

### 为什么 C&#43;&#43; 构造函数不能是虚函数？

1. 从语法层面来说

虚函数的主要目的是实现多态，即允许在派生类中覆盖基类的成员函数。

但是，构造函数负责初始化类的对象，每个类都应该有自己的构造函数。

在派生类中，基类的构造函数会被自动调用，用于初始化基类的成员。因此，构造函数没有被覆盖的必要，不需要使用虚函数来实现多态。

2. 从虚函数表机制来回答

虚函数使用了一种称为虚函数表（vtable）的机制。然而，在调用构造函数时，对象还没有完全创建和初始化，所以虚函数表可能尚未设置。

这意味着在构造函数中使用虚函数表会导致未定义的行为。只有执行完了对象的构造，虚函数表才会被正确的初始化。

### 为什么 C&#43;&#43; 基类析构函数需要是虚函数？

首先我们需要知道析构函数的作用是什么。

析构函数是进行类的清理工作，比如释放内存、关闭DB链接、关闭Socket等等，

前面我们在介绍虚函数的时候就说到，为实现多态性（C&#43;&#43;多态），可以通过基类的指针或引用访问派生类的成员。

也就是说，声明一个基类指针，这个基类指针可以指向派生类对象。


- 若基类Base的析构函数没有定义为虚函数，当创建一个派生类Derived的对象，并通过基类指针ptr删除它时，只有基类Base的析构函数被调用（因为这里没有多态，构造多态的必要条件就是虚函数）。派生类Derived的析构函数不会被调用，导致资源（这里是resource）没有被释放，从而产生资源泄漏。
- 若基类Base的析构函数是虚函数，所以当删除ptr时，会首先调用派生类Derived的析构函数，然后调用基类Base的析构函数，这样可以确保对象被正确销毁。

### 为什么默认的析构函数不是虚函数？

原因是虚函数不同于普通成员函数，当类中有虚成员函数时，类会自动进行一些额外工作。

这些额外的工作包括生成虚函数表和虚表指针，虚表指针指向虚函数表。

每个类都有自己的虚函数表，虚函数表的作用就是保存本类中虚函数的地址，我们可以把虚函数表形象地看成一个数组，这个数组的每个元素存放的就是各个虚函数的地址。

这样一来，就会占用额外的内存，当们定义的类不被其他类继承时，这种内存开销无疑是浪费的。

### 为什么C&#43;&#43;的成员模板函数不能是virtual的？

为什么在C&#43;&#43;里面，一个类的成员函数不能既是 template 又是 virtual 的。比如，下面的代码编译会报错：

```cpp
class Animal{
  public:
      template&lt;typename T&gt;
      virtual void make_sound(){
        //...
      }
};
```

因为C&#43;&#43;的编译与链接模型是&#34;分离&#34;的(至少是部分原因吧)。从Unix/C开始，一个C/C&#43;&#43;程序就可以被分开编译，然后用一个linker链接起来。这种模型有一个问题，就是各个编译单元可能对另一个编译单元一无所知。一个 function template最后到底会被 instantiate 为多少个函数，要等整个程序(所有的编译单元)全部被编译完成才知道。同时，virtual function的实现大多利用了一个&#34;虚函数表&#34;（参考: 虚函数机制）的东西，这种实现中，一个类的内存布局(或者说虚函数表的内存布局)需要在这个类编译完成的时候就被完全确定。所以当一个虚拟函数是模板函数时，编译器在编译时无法为其生成一个确定的虚函数表条目，因为模板函数可以有无数个实例。但是编译时无法确定需要调用哪个特定的模板实例。因此，C&#43;&#43;标准规定member function 不能既是 template 又是 virtual 的。

### C&#43;&#43;中 sizeof 一个空类大小是多大

```cpp
class Empty {};
```

大多数情况下 sizeof(Empty) = 1，这是因为C&#43;&#43;标准要求每个对象都必须具有独一无二的内存地址。

为了满足这一要求，编译器会给每个空类分配一定的空间，通常是1字节。

这样，即使是空类，也能保证每个实例都有不同的地址。


---

> 作者: yitao  
> URL: https://yitaonote.com/2025/978bdd1/  

