# 归并排序


&lt;!--more--&gt;

### 算法思路
![IMG_20210702_174626.jpg](https://cdn.acwing.com/media/article/image/2021/07/02/94631_7fb6b17cdb-IMG_20210702_174626.jpg)

稳定 $O(n\log n)$

### 完整代码

#### C&#43;&#43;

```cpp
#include &lt;iostream&gt;

using namespace std;

const int N = 1e5 &#43; 10;

int q[N], tmp[N];

void merge_sort(int q[], int l, int r)
{
    if (l &gt;= r) return;

    int mid = l &#43; r &gt;&gt; 1;

    merge_sort(q, l, mid), merge_sort(q, mid &#43; 1, r);

    int k = 0, i = l, j = mid &#43; 1;
    while (i &lt;= mid &amp;&amp; j &lt;= r)
        if (q[i] &lt;= q[j]) tmp[k &#43;&#43; ] = q[i &#43;&#43; ];
        else tmp[k &#43;&#43; ] = q[j &#43;&#43; ];

    while (i &lt;= mid) tmp[k &#43;&#43; ] = q[i &#43;&#43; ];
    while (j &lt;= r) tmp[k &#43;&#43; ] = q[j &#43;&#43; ];

    for (i = l, j = 0; i &lt;= r; i &#43;&#43; , j &#43;&#43; ) q[i] = tmp[j];
}

int main()
{
    int n;
    cin &gt;&gt; n;
    for (int i = 0; i &lt; n; i &#43;&#43; ) scanf(&#34;%d&#34;, &amp;q[i]);

    merge_sort(q, 0, n - 1);

    for (int i = 0; i &lt; n; i &#43;&#43; ) printf(&#34;%d &#34;, q[i]);
    return 0;
}
```
#### Python3
```python3
def merge_sort(q, l, r):
    if l &gt;= r:
        return

    mid = l &#43; r &gt;&gt; 1
    merge_sort(q, l, mid)
    merge_sort(q, mid &#43; 1, r)

    i, j, k = l, mid &#43; 1, 0
    while i &lt;= mid and j &lt;= r:
        if q[i] &lt; q[j]:
            tmp[k] = q[i]
            i &#43;= 1
        else:
            tmp[k] = q[j]
            j &#43;= 1
        k &#43;= 1

    while i &lt;= mid:
        tmp[k] = q[i]
        i &#43;= 1
        k &#43;= 1
    while j &lt;= r:
        tmp[k] = q[j]
        j &#43;= 1
        k &#43;= 1

    j = 0
    for i in range(l, r &#43; 1):
        q[i] = tmp[j]
        j &#43;= 1

if __name__ == &#34;__main__&#34;:
    n = int(input())
    q = list(map(int, input().split()))
    tmp = [0] * n
    merge_sort(q, 0, n - 1)
    for i in range(len(q)):
        print(q[i], end = &#39; &#39;)
```


---

> 作者: yitao  
> URL: https://yitaonote.com/2025/f751a18/  

