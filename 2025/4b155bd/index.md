# 并发编程（二）


[《现代C&#43;&#43;并发编程教程》](https://mq-b.github.io/ModernCpp-ConcurrentProgramming-Tutorial/) —— C&#43;&#43;并发编程学习笔记（二）

&lt;!--more--&gt;

## 等待事件或条件

假设你正在一辆夜间运行的地铁上，那么你要如何在正确的站点下车呢？

&gt; 1.一直不休息，每一站都能知道，这样就不会错过你要下车的站点，但是这会很疲惫。

这种方法被称为“忙等待（busy waiting）”也称 ==**“自旋“**==。

```cpp
bool flag = false;
std::mutex m;

void wait_for_flag() {
    std::unique_lock&lt;std::mutex&gt; lk{ m };
    while (!flag){
        lk.unlock();    // 1 解锁互斥量
        lk.lock();      // 2 上锁互斥量
    }
}
```

&gt; 2.可以看一下时间，估算一下地铁到达目的地的时间，然后设置一个稍早的闹钟，就休息。这个方法听起来还行，但是你可能被过早的叫醒，甚至估算错误导致坐过站，又或者闹钟没电了睡过站。

第二种方法就是加个延时，这种实现进步了很多，减少浪费的执行时间，但很难确定正确的休眠时间。这会影响到程序的行为，在需要快速响应的程序中就意味着丢帧或错过了一个时间片。循环中，**休眠 ②** 前函数对互斥量解锁 **①**，再休眠结束后再对互斥量上锁，让另外的线程有机会获取锁并设置标识（因为修改函数和等待函数共用一个互斥量）。

```cpp
void wait_for_flag() {
    std::unique_lock&lt;std::mutex&gt; lk{ m };
    while (!flag){
        lk.unlock();    // 1 解锁互斥量
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // 2 休眠
        lk.lock();      // 3 上锁互斥量
    }
}
```

&gt; 3.事实上最简单的方式是，到站的时候有人或者其它东西能将你叫醒（比如手机的地图，到达设置的位置就提醒）。

第三种方式（也是最好的）实际上就是使用条件变量了。通过另一线程触发等待事件的机制是最基本的唤醒方式，这种机制就称为“条件变量”。

C&#43;&#43; 标准库对条件变量有两套实现：`std::condition_variable` 和 `std::condition_variable_any`，这两个实现都包含在 `&lt;condition_variable&gt;` 这个头文件中。

`condition_variable_any` 类是 `std::condition_variable` 的泛化。相对于只在 `std::unique_lock&lt;std::mutex&gt;` 上工作的 `std::condition_variable`，`condition_variable_any` 能在任何满足 *[可基本锁定(BasicLockable)](https://zh.cppreference.com/w/cpp/named_req/BasicLockable)* 要求的锁上工作，所以增加了 _any 后缀。显而易见，这种区分必然是 any 版更加通用但是却有更多的性能开销。所以通常首选 `std::condition_variable`。有特殊需求，才会考虑 `std::condition_variable_any`。

```cpp
std::mutex mtx;  // 创建了一个互斥量，用于保护共享数据的访问，确保在多线程环境下的数据同步。
std::condition_variable cv;  // 创建了一个条件变量，用于线程间的同步，当条件不满足时，线程可以等待，直到条件满足时被唤醒。
bool arrived = false;  // 设置了一个标志位，表示是否到达目的地。

void wait_for_arrival() {
    std::unique_lock&lt;std::mutex&gt; lck(mtx);  // 使用互斥量创建了一个独占锁。
    cv.wait(lck, []{ return arrived; }); // 阻塞当前线程，释放（unlock）锁，直到条件被满足。
    std::cout &lt;&lt; &#34;到达目的地，可以下车了！&#34; &lt;&lt; std::endl;
}

void simulate_arrival() {
    std::this_thread::sleep_for(std::chrono::seconds(5)); // 模拟地铁到站，假设5秒后到达目的地
    {
        std::lock_guard&lt;std::mutex&gt; lck(mtx);
        arrived = true; // 设置条件变量为 true，表示到达目的地
    }
    cv.notify_one(); // 通知等待的线程
}
```

这样，当 `simulate_arrival` 函数执行后，`arrived` 被设置为 `true`，并且通过 `cv.notify_one()` 唤醒了等待在条件变量上的线程，从而使得 `wait_for_arrival` 函数中的等待结束，可以执行后续的操作，即输出提示信息。

条件变量的 `wait` 成员函数有两个版本，以上代码使用的就是第二个版本，传入了一个谓词。

```cpp
void wait(std::unique_lock&lt;std::mutex&gt;&amp; lock);                 // 1

template&lt;class Predicate&gt;
void wait(std::unique_lock&lt;std::mutex&gt;&amp; lock, Predicate pred); // 2
```

②等价于：
```cpp
while (!pred())
    wait(lock);
```
第二个版本只是对第一个版本的包装，等待并判断谓词，会调用第一个版本的重载。这可以避免 ==*虚假唤醒*==


&gt; 条件变量虚假唤醒是指在使用条件变量进行线程同步时，有时候线程可能会在没有收到通知的情况下被唤醒。问题取决于程序和系统的具体实现。解决方法很简单，在循环中等待并判断条件可一并解决。使用 C&#43;&#43; 标准库则没有这个烦恼了。

## 线程安全的队列

这里介绍一个更为复杂的示例，用于巩固条件变量的学习。在实现一个线程安全的队列过程中，需要注意两点内容：

&gt; 1. 当执行 `push` 操作时，需要确保没有其他线程正在执行 `push` 或 `pop` 操作；同样，在执行 `pop` 操作时，也需要确保没有其他线程正在执行 `push` 或 `pop` 操作。

&gt; 2. 当队列为空时，不应该执行 `pop` 操作。因此，我们需要使用条件变量来传递一个谓词，以确保在执行 `pop` 操作时队列不为空。

以下是一个线程安全的模版类 `threadsafe_queue`：

```cpp
template&lt;typename T&gt;
class threadsafe_queue {
    mutable std::mutex m;              // 互斥量，用于保护队列操作的独占访问
    std::condition_variable data_cond; // 条件变量，用于在队列为空时等待
    std::queue&lt;T&gt; data_queue;          // 实际存储数据的队列
public:
    threadsafe_queue() {}              // 无参构造
    void push(T new_value) {
        {
            std::lock_guard&lt;std::mutex&gt; lk { m };
            data_queue.push(new_value);
        }
        data_cond.notify_one();
    }
    // 从队列中弹出元素（阻塞直到队列不为空）
    void pop(T&amp; value) {
        std::unique_lock&lt;std::mutex&gt; lk{ m };
        data_cond.wait(lk, [this] {return !data_queue.empty(); });  // 这里的 this 表示按值传递 this，见 lambda 表达式用法
        value = data_queue.front();
        data_queue.pop();
    }
    // 从队列中弹出元素（阻塞直到队列不为空），并返回一个指向弹出元素的 shared_ptr
    std::shared_ptr&lt;T&gt; pop() {
        std::unique_lock&lt;std::mutex&gt; lk{ m };
        data_cond.wait(lk, [this] {return !data_queue.empty(); });
        std::shared_ptr&lt;T&gt; res { std::make_shared&lt;T&gt;(data_queue.front()) };
        data_queue.pop();
        return res;
    }
    bool empty()const {
        std::lock_guard&lt;std::mutex&gt; lk (m);
        return data_queue.empty();
    }
};
```

## 使用 `future`

举个例子，我们在车站等车，你可能会做一些别的事情打发时间，比如学习现代C&#43;&#43;并发编程教程、玩手机等，但始终在等待一件事情：==*车到站。*==

C&#43;&#43; 标准库将这种事件称为 `future`。它用于处理线程中需要等待某个事件的情况，线程知道预期结果。等待的同时也可以执行其它的任务。

C&#43;&#43; 标准库有两种 `future`，都声明在 `&lt;future&gt;` 头文件中：独占的 `std::future` 、共享的 `std::shared_future`。它们的区别与 `std::unique_ptr` 和 `std::shared_ptr` 类似。`std::future` 只能与单个指定事件关联，而 `std::shared_future` 能关联多个事件。它们都是模板，它们的模板类型参数，就是其关联的事件（函数）的返回类型。当多个线程需要访问一个独立 `future` 对象时， 必须使用互斥量或类似同步机制进行保护。而多个线程访问同一共享状态，若每个线程都是通过其自身的 `shared_future` 对象副本进行访问，则是安全的。

最简单有效的使用是，我们先前讲的 `std::thread` 在线程中执行任务是没有返回值的，这个问题就能使用 `future` 解决。

### 创建异步任务获取返回值

假设需要执行一个耗时任务并获取其返回值，但是并不急切的需要它。那么就可以启动新线程计算，然而 `std::thread` 没提供直接从线程获取返回值的机制。所以我们可以使用 `std::async` 函数模板。

使用 `std::async` 启动一个异步任务，它会返回一个 `std::future` 对象，这个对象和任务关联，将持有最终计算出来的结果。当需要任务执行完的结果的时候，只需要调用 `get()` 成员函数，就会阻塞直到 `future` 为就绪为止（即任务执行完毕），返回执行结果。`valid()` 成员函数检查 `future` 当前是否关联共享状态，即是否当前关联任务。还未关联，或者任务已经执行完（调用了 `get()`、`set()`），都会返回 `false`。

```cpp
#include &lt;iostream&gt;
#include &lt;thread&gt;
#include &lt;future&gt; // 引入 future 头文件

int task(int n) {
    std::cout &lt;&lt; &#34;异步任务 ID: &#34; &lt;&lt; std::this_thread::get_id() &lt;&lt; &#39;\n&#39;;
    return n * n;
}

int main() {
    std::future&lt;int&gt; future = std::async(task, 10);
    std::cout &lt;&lt; &#34;main: &#34; &lt;&lt; std::this_thread::get_id() &lt;&lt; &#39;\n&#39;;
    std::cout &lt;&lt; std::boolalpha &lt;&lt; future.valid() &lt;&lt; &#39;\n&#39;; // true
    std::cout &lt;&lt; future.get() &lt;&lt; &#39;\n&#39;;
    std::cout &lt;&lt; std::boolalpha &lt;&lt; future.valid() &lt;&lt; &#39;\n&#39;; // false
}
```

关于 `std::async` 的参数传递，这里不再展开记录，用时再查。

## 信号量

信号量是一个非常轻量简单的同步设施（在 C&#43;&#43; 20中被引入），它维护一个计数，这个计数不能小于 0。信号量提供两种基本操作：释放（增加计数）和等待（减少计数）。如果当前信号量的计数值为 0，那么执行“等待”操作的线程将会一直阻塞，直到计数大于 0，也就是其它线程执行了 ==“释放”== 操作。

C&#43;&#43; 提供了两个信号量类型：`std::counting_semaphore` 与 `std::binary_semaphore`，定义在 `&lt;semaphore&gt;` 中。其中 `binary_semaphore` 只是 `counting_semaphore` 的一个特化别名（其 `LeastMaxValue` 为1，`LeastMaxValue` 意思是信号量维护的计数最大值。）：

```cpp
using binary_semaphore = counting_semaphore&lt;1&gt;;
```

举个具体使用信号量的例子：
```cpp
// 全局二元信号量对象
// 设置对象初始计数为 0
std::binary_semaphore smph_signal_main_to_thread{ 0 };
std::binary_semaphore smph_signal_thread_to_main{ 0 };

void thread_proc() {
    smph_signal_main_to_thread.acquire();
    std::cout &lt;&lt; &#34;[线程] 获得信号&#34; &lt;&lt; std::endl;

    std::this_thread::sleep_for(3s);

    std::cout &lt;&lt; &#34;[线程] 发送信号\n&#34;;
    smph_signal_thread_to_main.release();
}

int main() {
    std::jthread thr_worker{ thread_proc };

    std::cout &lt;&lt; &#34;[主] 发送信号\n&#34;;
    smph_signal_main_to_thread.release();

    smph_signal_thread_to_main.acquire();
    std::cout &lt;&lt; &#34;[主] 获得信号\n&#34;;
}
```

结果：

```text
[主] 发送信号
[线程] 获得信号
[线程] 发送信号
[主] 获得信号
```

`acquire` 函数就是我们先前说的“等待”（原子地减少计数），`release` 函数就是&#34;释放&#34;（原子地增加计数）。

&gt; [!TIP]
&gt; 信号量常用于 **发信/提醒** 而非**互斥**，通过初始化该信号量为 0 从而阻塞尝试 `acquire()` 的接收者，直至提醒者通过调用 `release(n)` **“发信”**。在此方面可把信号量当作条件变量的替代品，通常它有更好的性能。

假设我们有一个 Web 服务器，它只能处理有限数量的并发请求。为了防止服务器过载，我们可以使用信号量来限制并发请求的数量。

```cpp
// 定义一个信号量，最大并发数为 3
std::counting_semaphore&lt;3&gt; semaphore{ 3 };  // counting_semaphore 轻量同步原语，允许同一资源进行多个并发的访问，至少允许 LeastMaxValue 个同时访问者

void handle_request(int request_id) {
    // 请求到达，尝试获取信号量
    std::cout &lt;&lt; &#34;进入 handle_request 尝试获取信号量\n&#34;;

    semaphore.acquire();

    std::cout &lt;&lt; &#34;成功获取信号量\n&#34;;

    // 此处延时三秒可以方便测试，会看到先输出 3 个“成功获取信号量”，因为只有三个线程能成功调用 acquire，剩余的会被阻塞
    std::this_thread::sleep_for(3s);

    // 模拟处理时间
    std::random_device rd;
    std::mt19937 gen{ rd() };
    std::uniform_int_distribution&lt;&gt; dis(1, 5);
    int processing_time = dis(gen);
    std::this_thread::sleep_for(std::chrono::seconds(processing_time));

    std::cout &lt;&lt; std::format(&#34;请求 {} 已被处理\n&#34;, request_id);

    semaphore.release();
}

int main() {
    // 模拟 10 个并发请求
    std::vector&lt;std::jthread&gt; threads;
    for (int i = 0; i &lt; 10; &#43;&#43;i) {
        threads.emplace_back(handle_request, i);
    }
}
```

牢记信号量的基本的概念不变，计数的值不能小于 0，如果当前信号量的计数值为 0，那么执行 ==“等待”（acquire）== 操作的线程将会一直阻塞。明白这点，那么就都不存在问题。


---

> 作者: yitao  
> URL: https://yitaonote.com/2025/4b155bd/  

