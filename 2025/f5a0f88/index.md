# N皇后问题


&lt;!--more--&gt;

### 第一种写法，按行枚举
```cpp
#include &lt;iostream&gt;
using namespace std;


const int N = 20;

char g[N][N];
bool col[N], dg[N], udg[N];
int n;




void dfs(int u)
{
    if(u == n)
    {
        for(int i = 0; i &lt; n; i &#43;&#43; )  puts(g[i]);
        puts(&#34;&#34;);
        return;
    }

    for(int i = 0; i &lt; n; i &#43;&#43; )
    {
        if(!col[i] &amp;&amp; !dg[u &#43; i] &amp;&amp; !udg[n - u &#43; i])
        {
            g[u][i] = &#39;Q&#39;;
            col[i] = dg[u &#43; i] = udg[n - u &#43; i] = true;
            dfs(u &#43; 1);
            col[i] = dg[u &#43; i] = udg[n - u &#43; i] = false;
            g[u][i]= &#39;.&#39;;
        }
    }
}

int main()
{
    cin &gt;&gt; n;
    for(int i = 0; i &lt; n; i &#43;&#43; )
        for(int j = 0; j &lt; n; j &#43;&#43; )
            g[i][j] = &#39;.&#39;;

    dfs(0);
    return 0;
}
```

### 第二种写法，一格一格枚举
```cpp
#include &lt;iostream&gt;
using namespace std;

const int N = 20;
bool row[N], col[N], dg[N], udg[N];
char g[N][N];
int n;


void dfs(int x, int y, int s)
{
    if(y == n)  y = 0, x &#43;&#43;;

    if(x == n)
    {
        if(s == n)
        {
            for(int i = 0; i &lt; n; i &#43;&#43;) puts(g[i]);
            puts(&#34;&#34;);
        }
        return;
    }

    dfs(x, y &#43; 1, s);

    if(!row[x] &amp;&amp; !col[y] &amp;&amp; !dg[x &#43; y] &amp;&amp; !udg[x - y &#43; n])
    {
        g[x][y] = &#39;Q&#39;;
        row[x] = col[y] = dg[x &#43; y] = udg[x - y &#43; n] = true;
        dfs(x, y &#43; 1, s &#43; 1);
        row[x] = col[y] = dg[x &#43; y] = udg[x - y &#43; n] = false;
        g[x][y] = &#39;.&#39;;
    }
}

int main()
{
    cin &gt;&gt; n;

    for(int i = 0; i &lt; n; i &#43;&#43; )
        for(int j = 0; j &lt; n; j &#43;&#43; )
            g[i][j] = &#39;.&#39;;

    dfs(0, 0, 0);
    return 0;
}
```


---

> 作者: yitao  
> URL: https://yitaonote.com/2025/f5a0f88/  

