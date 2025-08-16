# 并查集


&lt;!--more--&gt;

# 并查集

### 适用场景

1、将两个集合合并

2、询问两个元素是否在一个集合当中

3、维护一个集合中的点的个数：```siz[root]```，见 [AcWing 837. 连通块中点的数量](https://www.acwing.com/activity/content/code/content/1246674/)

4、维护一个集合中各个点到根节点的距离：```d[]```，见： [AcWing 240. 食物链](https://www.acwing.com/activity/content/code/content/1491107/)

核心操作：**路径压缩**和**按秩合并**（按秩合并不常用，这里省略）

上述两个操作最坏情况下为 $O(\log n)$

```cpp
int find(int x)
{
    if (p[x] != x)  p[x] = find(p[x]);
    return p[x];
}
```

### 算法原理 &amp; 实现

每个集合用**一棵树**来表示，**数根的编号**就是**整个集合的编号**，每个节点存储它的**父节点**，`p[x]`表示`x`的父节点。

![QQ截图20210629135632.png](https://cdn.acwing.com/media/article/image/2021/06/29/94631_c66fe7f0d8-QQ截图20210629135632.png)

问题1：如何判断树根？  
```cpp
if (p[x] == x)
```

问题2：如何求x的集合编号？
```cpp
while (p[x] != x)
    x = p[x];
```

问题3：如何合并两个集合？

`p[x]` 是 `x` 的集合编号，`p[y]`是 `y`的集合编号。

```cpp
p[x] = y
```


#### 初始化

初始时，每个点都是一个集合，故每个点的**树根**就是它本身。
```cpp
for (int i = 1; i &lt;= m; i &#43;&#43; )  
    p[i] = i;
```


### 完整代码
```cpp
#include &lt;iostream&gt;

using namespace std;

const int N = 1e5 &#43; 10;
int p[N], n, m;

int find(int x)
{
    if(p[x] != x)  p[x] = find(p[x]);
    return p[x];
}

int main()
{
    scanf(&#34;%d%d&#34;, &amp;n, &amp;m);
    for (int i = 1; i &lt;=m ; i &#43;&#43; ) p[i] = i;

    while (m -- )
    {
        char op[2];
        int a, b;
        scanf(&#34;%s%d%d&#34;, op, &amp;a, &amp;b);

        if (op[0] == &#39;M&#39;)  p[find(a)] = find(b);
        else
        {
            if (find(a) == find(b))  puts(&#34;Yes&#34;);
            else puts(&#34;No&#34;);
        }
    }
    return 0;
}
```


---

> 作者: yitao  
> URL: https://yitaonote.com/2025/55e4c93/  

