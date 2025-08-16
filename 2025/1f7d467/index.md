# Prim算法求最小生成树


&lt;!--more--&gt;

&gt; Prim 算法原理及证明见[提高课](https://www.acwing.com/activity/content/code/content/1745411/)。

Prim算法求最小生成树 $O(n^{2})$，跟Dijkstra很像

![IMG_20210628_204051_edit_188738021723283.jpg](https://cdn.acwing.com/media/article/image/2021/06/28/94631_240b2fb5d8-IMG_20210628_204051_edit_188738021723283.jpg)

### 完整代码
```
#include &lt;iostream&gt;
#include &lt;algorithm&gt;
#include &lt;cstring&gt;

using namespace std;

const int N = 510, INF = 0x3f3f3f3f;

int n, m;
int g[N][N];
int dist[N];    // 当前点距离集合的距离
bool st[N];

int prim()
{
    memset(dist, 0x3f, sizeof dist);
    int res = 0; // 最小生成树中所有边长度之和
    for (int i = 0; i &lt; n; i &#43;&#43; ) // 每次找到集合外的，距集合距离最小的点
    {   
        int t = -1;     // t = -1 表示当前还没还有找到任何一个点
        for (int j = 1; j &lt;= n; j &#43;&#43; )
            if (!st[j] &amp;&amp; (t == -1 || dist[t] &gt; dist[j]))
                t = j;
        // 当前图是不连通的，不存在最小生成树
        if (i &amp;&amp; dist[t] == INF)    return INF;
        if (i) res &#43;= dist[t];     // dist[t] 表示一条树边，加到生成树中去

        for (int j = 1; j &lt;= n; j &#43;&#43; )  dist[j] = min(dist[j], g[t][j]);
        st[t] = true;
    }
    return res;
}

int main()
{
    scanf(&#34;%d%d&#34;, &amp;n, &amp;m);
    memset(g, 0x3f, sizeof g);

    while (m -- )
    {
        int a, b, c;
        scanf(&#34;%d%d%d&#34;, &amp;a, &amp;b, &amp;c);
        g[a][b] = g[b][a] = min(g[a][b], c);    // 无向图，处理重边
    }
    int t = prim();
    if (t == INF) puts(&#34;impossible&#34;);
    else printf(&#34;%d\n&#34;, t);
    return 0;
}
```


---

> 作者: yitao  
> URL: https://yitaonote.com/2025/1f7d467/  

