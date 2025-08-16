# Kruskal算法求最小生成树


&lt;!--more--&gt;

&gt; Kruskal 算法原理及证明见[提高课](https://www.acwing.com/activity/content/code/content/1745411/)。

## Kruskal算法求最小生成树 $O(m\log m)$
![IMG_20210629_141909_edit_223503600585686.jpg](https://cdn.acwing.com/media/article/image/2021/06/29/94631_ff56a31cd8-IMG_20210629_141909_edit_223503600585686.jpg)

**这里重载运算符的方式也可以通过写cmp()来作为sort的第三个参数**

```cpp
bool cmp(Edge a, Edge b)
{
    return a.w &lt; b.w;
}
```


### 完整代码
```cpp
#include &lt;iostream&gt;
#include &lt;algorithm&gt;
#include &lt;cstring&gt;
using namespace std;

const int N = 100010, M = 200010, INF = 0x3f3f3f3f;
int p[N];
int n, m;

struct Edge
{
    int a, b, w;
    bool operator&lt; (const Edge &amp;W)const
    {
        return w &lt; W.w;
    }
}edges[M];


int find(int x)
{
    if (p[x] != x)  p[x] = find(p[x]);
    return p[x];
}

int kruskal()
{
    sort(edges, edges &#43; m);

    for (int i = 1; i &lt;= n; i &#43;&#43; )  p[i] = i;

    int res = 0, cnt = 0;
    for (int i = 0; i &lt; m; i &#43;&#43; )
    {
        int a = edges[i].a, b = edges[i].b, w = edges[i].w;

        a = find(a), b = find(b);
        if (a != b)
        {
            p[a] = b;
            res &#43;= w;
            cnt &#43;&#43;;
        }
    }
    if (cnt &lt; n - 1)    return INF;
    return res;
}

int main()
{
    cin &gt;&gt; n &gt;&gt; m;

    for (int i = 0; i &lt; m; i &#43;&#43; )
    {
        int a, b, w;
        scanf(&#34;%d%d%d&#34;, &amp;a, &amp;b, &amp;w);
        edges[i] = {a, b, w};
    }

    int t = kruskal();

    if (t == INF)   puts(&#34;impossible&#34;);
    else cout &lt;&lt; t &lt;&lt; endl;
    return 0;
}
```


---

> 作者: yitao  
> URL: https://yitaonote.com/2025/acc2381/  

