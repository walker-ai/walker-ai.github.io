# 快速幂求逆元


&lt;!--more--&gt;


# 求逆元

### 前置知识

1、设 $m$ 为正整数，若 $a$ 和 $b$ 是整数，且 $m\mid (a - b)$    ，则称 **$a$ 和 $b$ 模 $m$ 同余**，记为 $a\equiv b\pmod m$。

2、若 $a$ 和 $b$ 是整数，则 $a\equiv b\pmod m$ 当且仅当存在整数 $k$，使得 $a = b &#43; km$。

3、设 $m$ 为正整数，模 $m$ 的同余满足以下性质：

- 自反性：若 $a$ 是整数，则 $a\equiv a\pmod m$

- 对称性：若 $a$ 和 $b$ 是整数，且 $a\equiv b\pmod m$，则 $b\equiv a\pmod m$

- 传递性：若 $a, b$ 和 $c$ 是整数，且 $a\equiv b\pmod m$ 和 $b\equiv c\pmod m$，则 $a\equiv c\pmod m$   

4、若 $a, b, c$ 和 $m$ 是整数，$m &gt; 0$，且 $a\equiv b\pmod m$，则有：

$$
\begin{align}
a &#43; c  &amp;\equiv b &#43; c\pmod m \\\
a - c &amp;\equiv b - c\pmod m \\\
ac &amp;\equiv bc\pmod m
\end{align}
$$

5、乘法逆元：若整数 $b$ 与 $m$ 互质，并且 $b\mid a$，则存在一个整数 $x$，使得 $\dfrac{a}{b} \equiv a\cdot x\pmod m$，则称 $x$ 为 $b$ 的模 $m$ 乘法逆元，记为 $b^{-1}\pmod m$ $\Rightarrow$ $\dfrac{a}{b} \equiv a\cdot b^{-1}\pmod m \Rightarrow bb^{-1}\equiv 1\pmod m$，其中 $b^{-1}$ 是 $b$ 的逆元。

6、费马小定理：如果 $p$ 是一个质数，并且 $a$ 与 $p$ 互质，则有 $a^{p-1}\equiv 1\pmod p$

7、[快速幂](https://www.acwing.com/activity/content/code/content/1259683/)

8、[扩展欧几里得算法](https://www.acwing.com/activity/content/code/content/1261566/)

9、一个质数和比它小的每一个非零自然数都互质

### 适用场景

一般在做除法时，有余数存在是一件很麻烦的事，两个整数相除不一定是整数，两个整数相乘一定整数，因此我们希望将除法转化成乘法（模上一个数余数相同）。

### 算法分析

根据前置知识中的内容，当模数 $p$ 为质数时：

**1、快速幂求逆元：**

若 $b$ 与 $p$ 互质，根据费马小定理，有：

$$b^{p-1}\equiv 1\pmod p \iff b\cdot b^{p - 2}\equiv 1\pmod p$$

此时 $b^{p - 2}$ 为 $b$ 的逆元，这时即可用快速幂来求解逆元。

若 $b$ 与 $p$ 不互质，则 $b$ 的乘法逆元不存在。

**2、扩展欧几里得求逆元：**

上面已经证明了，对于一个整数 $a$ 来说，若存在乘法逆元 $x = a^{-1}$，则有 $a\cdot a^{-1}\equiv 1\pmod p$，即存在整数 $y$ 使得 $ax = 1 &#43; yp$，整理得 $ax &#43; py = 1$。

而存在乘法逆元 $x = a^{-1}$ 的充要条件为 $a$ 与模数 $p$ 互质，即 $\gcd(a, p) = 1$，故原式转化为：

$$ ax &#43; py = 1 = \gcd(a, p)$$

符合**裴蜀定理**的表达式，因此我们可以用扩展欧几里得算法求解逆元 $x = a^{-1}$。


### 完整代码

#### 快速幂求逆元
```cpp
#include &lt;cstdio&gt;

using namespace std;

typedef long long LL;

int qmi(int a, int k, int p)
{
    int res = 1;
    while (k)
    {
        if (k &amp; 1) res = (LL)res * a % p;
        k &gt;&gt;= 1;
        a = (LL)a * a % p;
    }
    return res;
}

int main()
{
    int n;
    scanf(&#34;%d&#34;, &amp;n);
    while (n -- )
    {
        int a, p;
        scanf(&#34;%d%d&#34;, &amp;a, &amp;p);
        if (a % p) printf(&#34;%d\n&#34;, qmi(a, p - 2, p));
        else puts(&#34;impossible&#34;);
    }
    return 0;
}
```

#### 扩展欧几里得求逆元
```cpp
#include &lt;cstdio&gt;

using namespace std;

typedef long long LL;

int n;

int exgcd(int a, int b, int &amp;x, int &amp;y)
{
    if (!b)
    {
        x = 1, y = 0;
        return a;
    }

    int d = exgcd(b, a % b, y, x);
    y -= a / b * x;
    return d;
}

int main()
{
    scanf(&#34;%d&#34;, &amp;n);
    while (n -- )
    {
        int a, p, x, y;
        scanf(&#34;%d%d&#34;, &amp;a, &amp;p);
        int d = exgcd(a, p, x, y);
        if (d == 1) printf(&#34;%d\n&#34;, ((LL)x &#43; p) % p); // 保证 x 是正数
        else puts(&#34;impossible&#34;);
    }
    return 0;
}
```


---

> 作者: yitao  
> URL: https://yitaonote.com/2025/c322c46/  

