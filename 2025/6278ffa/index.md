# Floyd求最短路


&lt;!--more--&gt;

#  Floyd 算法

### 算法框架

![IMG_20210628_164616_edit_183458334133463.jpg](https://cdn.acwing.com/media/article/image/2021/06/28/94631_603f3714d7-IMG_20210628_164616_edit_183458334133463.jpg)

### 适用场景

多源汇最短路，可以有负权边，但是不能有负权回路。

### 算法原理概述

基于动态规划。

闫氏DP分析法：

状态表示

&gt; $d(k,i,j)$：从点 $i$ 出发，只经过 $1\sim k$ 这些中间点到达 $j$ 的最短距离。

状态计算

&gt; $d(k,i,j) = d(k-1,i,k) &#43; d(k-1,k,j) \Rightarrow d(i,j) = d(i,k) &#43; d(k,j)$

### 时间复杂度分析

三重循环，故复杂度为 $O(n^{3})$。

### 完整代码
注：一定要先循环```k```，```i```和```j```的顺序可以任意颠倒。
```cpp
#include &lt;iostream&gt;
#include &lt;cstring&gt;
#include &lt;algorithm&gt;

using namespace std;
const int N = 210, INF = 1e9;

int n, m, Q;
int d[N][N];// 邻接矩阵，也是floyd算法处理的距离

void floyd()
{
    for (int k = 1; k &lt;= n; k &#43;&#43; )
        for (int i = 1; i &lt;= n; i &#43;&#43; )
            for (int j = 1; j &lt;= n; j &#43;&#43; )
                d[i][j] = min(d[i][j], d[i][k] &#43; d[k][j]);
}

int main()
{
    scanf(&#34;%d%d%d&#34;, &amp;n, &amp;m, &amp;Q);
    for (int i = 1; i &lt;= n; i &#43;&#43; )
        for (int j = 1; j &lt;= n; j &#43;&#43; )
            if (i == j) d[i][j] = 0; // 去掉自环
            else d[i][j] = INF;
    while (m -- )
    {
        int a, b, w;
        scanf(&#34;%d%d%d&#34;, &amp;a, &amp;b, &amp;w);
        d[a][b] = min(d[a][b], w); // 若有重边，则只保留最短的边
    }
    floyd();
    while (Q -- )
    {
        int a, b;
        scanf(&#34;%d%d&#34;, &amp;a, &amp;b);
        // 和bellman ford类似，即使终点与起点不连通，也还是可能会被负权邻边更新，所以适当放宽条件
        if (d[a][b] &gt; INF / 2) puts(&#34;impossible&#34;);
        else printf(&#34;%d\n&#34;, d[a][b]);
    }
    return 0;
}
```


---

> 作者: yitao  
> URL: https://yitaonote.com/2025/6278ffa/  

