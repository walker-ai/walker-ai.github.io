# 并发编程（三）


[《现代C&#43;&#43;并发编程教程》](https://mq-b.github.io/ModernCpp-ConcurrentProgramming-Tutorial/) —— C&#43;&#43;并发编程学习笔记（三）

&lt;!--more--&gt;

## 原子操作

这里只简单介绍 `std::atomic&lt;bool&gt;`（包含在 `&lt;atomic&gt;` 中），最基本的整数原子类型。虽然同样不可复制不可移动，但可以使用非原子的 `bool` 类型进行构造，初始化为 `true` 或 `false`，并且能从非原子的 `bool` 对象赋值给 `std::atomic&lt;bool&gt;`：
```cpp
std::atomic&lt;bool&gt; b{ true };
b = false;
```

## 线程池

抽象的来说，可以当做是一个池子中存放了一堆线程，故称作**线程池**。简而言之，线程池是指代一组**预先创建的、可以复用的线程集合**。这些线程由线程池管理，用于执行多个任务而**无需频繁地创建和销毁**线程。

![Image](https://github.com/user-attachments/assets/4ca07a4d-56bd-45df-a23d-9444e7eb1bf5)

&gt; 这是一个典型的线程池结构。线程池包含一个任务队列，当有新任务加入时，调度器会将任务分配给线程池中的空闲线程进行执行。线程在执行完任务后会进入休眠状态，等待调度器的下一次唤醒。当有新的任务加入队列，并且有线程处于休眠状态时，调度器会唤醒休眠的线程，并分配新的任务给它们执行。线程执行完新任务后，会再次进入休眠状态，直到有新的任务到来，调度器才可能会再次唤醒它们。
图中线程1 就是被调度器分配了任务1，执行完毕后休眠，然而新任务的到来让调度器再次将它唤醒，去执行任务6，执行完毕后继续休眠。

使用线程池的益处我们已经加粗了，然而这其实并不是“线程池”独有的，任何创建和销毁存在较大开销的设施，都可以进行所谓的“池化”。

常见的还有：==套接字连接池、数据库连接池、内存池、对象池==。

下面简单介绍下常用的线程池。

### `boost::asio::thread_pool`

`boost::asio::thread_pool` 是 `Boost.Asio` 库提供的一种线程池实现。
&gt; Asio 是一个跨平台的 C&#43;&#43; 库，用于网络和低级 I/O 编程，使用 现代C&#43;&#43; 方法为开发人员提供一致的异步模型。

使用方法：
1. 创建线程池对象，指定或让 Asio 自动决定线程数量。
2. 提交任务：通过 `boost::asio::post` 函数模板提交任务到线程池中。
3. 阻塞，直到池中的线程完成任务。

```cpp
#include &lt;boost/asio.hpp&gt;
#include &lt;iostream&gt;

std::mutex m;

void print_task(int n) {
    std::lock_guard&lt;std::mutex&gt; lc{ m };
    std::cout &lt;&lt; &#34;Task &#34; &lt;&lt; n &lt;&lt; &#34; is running on thr: &#34; &lt;&lt;
        std::this_thread::get_id() &lt;&lt; &#39;\n&#39;;
}

int main() {
    boost::asio::thread_pool pool{ 4 }; // 创建一个包含 4 个线程的线程池

    for (int i = 0; i &lt; 10; &#43;&#43;i) {
        boost::asio::post(pool, [i] { print_task(i); });
    }

    pool.join(); // 等待所有任务执行完成
}
```

详情见 `boost/asio` 的使用，这里不再展开。

### 实现线程池

实现一个普通的能够满足日常开发需求的线程池实际上非常简单，只需要不到一百行代码。其实绝大部分开发者使用线程池，只是为了不重复多次创建线程罢了。所以只需要一个提供一个外部接口，可以传入任务到任务队列，然后安排线程去执行。无非是使用条件变量、互斥量、原子标志位，这些东西，就足够编写一个满足绝大部分业务需求的线程池。

我们先编写一个最基础的线程池，首先确定它的数据成员：

```cpp
class ThreadPool {
    std::mutex                mutex_;           // 用于保护共享资源（如任务队列）在多线程环境中的访问，避免数据竞争。
    std::condition_variable   cv_;              // 用于线程间的同步，允许线程等待特定条件（如新任务加入队列）并在条件满足时唤醒线程。
    std::atomic&lt;bool&gt;         stop_;            // 指示线程池是否停止。
    std::atomic&lt;std::size_t&gt;  num_threads_;     // 表示线程池中的线程数量。
    std::queue&lt;Task&gt;          tasks_;           // 任务队列，存储等待执行的任务，任务按提交顺序执行。
    std::vector&lt;std::thread&gt;  pool_;            // 线程容器，存储管理线程对象，每个线程从任务队列中获取任务并执行。
};
```

标头依赖：
```cpp
#include &lt;iostream&gt;
#include &lt;thread&gt;
#include &lt;mutex&gt;
#include &lt;condition_variable&gt;
#include &lt;future&gt;
#include &lt;atomic&gt;
#include &lt;queue&gt;
#include &lt;vector&gt;
#include &lt;syncstream&gt;
#include &lt;functional&gt;
```

提供构造析构函数，以及一些外部接口：`submit()`、`start()`、`stop()`、`join()`，也就完成了：

```cpp
inline std::size_t default_thread_pool_size()noexcept {
    std::size_t num_threads = std::thread::hardware_concurrency() * 2;
    num_threads = num_threads == 0 ? 2 : num_threads;
    return num_threads;
}

class ThreadPool {
private:
    std::mutex                mutex_;
    std::condition_variable   cv_;
    std::atomic&lt;bool&gt;         stop_;
    std::atomic&lt;std::size_t&gt;  num_threads_;
    std::queue&lt;Task&gt;          tasks_;
    std::vector&lt;std::thread&gt;  pool_;

public:
    using Task = std::packaged_task&lt;void()&gt;;

    ThreadPool(const ThreadPool&amp;) = delete;
    ThreadPool&amp; operator=(const ThreadPool&amp;) = delete;

    ThreadPool(std::size_t num_thread = default_thread_pool_size())
        : stop_{ false }, num_threads_{ num_thread } {
        start();
    }

    ~ThreadPool() {
        stop();
    }

    void stop() {
        stop_.store(true);
        cv_.notify_all();
        for (auto&amp; thread : pool_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        pool_.clear();
    }

    template&lt;typename F, typename... Args&gt;
    std::future&lt;std::invoke_result_t&lt;std::decay_t&lt;F&gt;, std::decay_t&lt;Args&gt;...&gt;&gt; submit(F&amp;&amp; f, Args&amp;&amp;...args) {
        using RetType = std::invoke_result_t&lt;std::decay_t&lt;F&gt;, std::decay_t&lt;Args&gt;...&gt;;
        if (stop_.load()) {
            throw std::runtime_error(&#34;ThreadPool is stopped&#34;);
        }

        auto task = std::make_shared&lt;std::packaged_task&lt;RetType()&gt;&gt;(
            std::bind(std::forward&lt;F&gt;(f), std::forward&lt;Args&gt;(args)...));
        std::future&lt;RetType&gt; ret = task-&gt;get_future();

        {
            std::lock_guard&lt;std::mutex&gt; lc{ mutex_ };
            tasks_.emplace([task] {(*task)(); });
        }
        cv_.notify_one();
        return ret;
    }

    void start() {
        for (std::size_t i = 0; i &lt; num_threads_; &#43;&#43;i) {
            pool_.emplace_back([this] {
                while (!stop_) {
                    Task task;
                    {
                        std::unique_lock&lt;std::mutex&gt; lc{ mutex_ };
                        cv_.wait(lc, [this] {return stop_ || !tasks_.empty(); });
                        if (tasks_.empty())
                            return;
                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }
                    task();
                }
            });
        }
    }
};
```

测试 demo:

```cpp
int main() {
    ThreadPool pool{ 4 }; // 创建一个有 4 个线程的线程池
    std::vector&lt;std::future&lt;int&gt;&gt; futures; // future 集合，获取返回值

    for (int i = 0; i &lt; 10; &#43;&#43;i) {
        futures.emplace_back(pool.submit(print_task, i));
    }

    for (int i = 0; i &lt; 10; &#43;&#43;i) {
        futures.emplace_back(pool.submit(print_task2, i));
    }

    int sum = 0;
    for (auto&amp; future : futures) {
        sum &#43;= future.get(); // get() 成员函数 阻塞到任务执行完毕，获取返回值
    }
    std::cout &lt;&lt; &#34;sum: &#34; &lt;&lt; sum &lt;&lt; &#39;\n&#39;;
} // 析构自动 stop()
```

可能的运行结果：

```cpp
Task 0 is running on thr: 6900
Task 1 is running on thr: 36304
Task 5 is running on thr: 36304
Task 3 is running on thr: 6900
Task 7 is running on thr: 6900
Task 2 is running on thr: 29376
Task 6 is running on thr: 36304
Task 4 is running on thr: 31416
🐢🐢🐢 1 🐉🐉🐉
Task 9 is running on thr: 29376
🐢🐢🐢 0 🐉🐉🐉
Task 8 is running on thr: 6900
🐢🐢🐢 2 🐉🐉🐉
🐢🐢🐢 6 🐉🐉🐉
🐢🐢🐢 4 🐉🐉🐉
🐢🐢🐢 5 🐉🐉🐉
🐢🐢🐢 3 🐉🐉🐉
🐢🐢🐢 7 🐉🐉🐉
🐢🐢🐢 8 🐉🐉🐉
🐢🐢🐢 9 🐉🐉🐉
sum: 90
```

它支持任意可调用类型，当然也包括非静态成员函数。我们使用了 `std::decay_t`，所以参数的传递其实是按值复制，而不是引用传递，这一点和大部分库的设计一致。示例如下：

```cpp
struct X {
    void f(const int&amp; n) const {
        std::osyncstream{ std::cout } &lt;&lt; &amp;n &lt;&lt; &#39;\n&#39;;
    }
};

int main() {
    ThreadPool pool{ 4 }; // 创建一个有 4 个线程的线程池

    X x;
    int n = 6;
    std::cout &lt;&lt; &amp;n &lt;&lt; &#39;\n&#39;;
    auto t = pool.submit(&amp;X::f, &amp;x, n); // 默认复制，地址不同
    auto t2 = pool.submit(&amp;X::f, &amp;x, std::ref(n));
    t.wait();
    t2.wait();
} // 析构自动 stop()
```

我们的线程池的 `submit` 成员函数在传递参数的行为上，与先前介绍的 `std::thread` 和 `std::async` 等设施基本一致。

构造函数和析构函数：

- 构造函数：初始化线程池并**启动线程**。
- 析构函数：停止线程池并等待所有线程结束。

外部接口：

- `stop()`：停止线程池，通知所有线程退出（不会等待所有任务执行完毕）。
- `submit()`：将任务提交到任务队列，并返回一个 `std::future` 对象用于获取任务结果以及确保任务执行完毕。
- `start()`：启动线程池，创建并启动指定数量的线程。


---

> 作者: yitao  
> URL: https://yitaonote.com/2025/80f2e62/  

