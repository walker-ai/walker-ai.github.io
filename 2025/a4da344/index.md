# 《HOT100》


《HOT100》做题记录

&lt;!--more--&gt;


### 5.最长回文子串

&gt; [🔗](https://leetcode.cn/problems/longest-palindromic-substring/description/?envType=study-plan-v2&amp;envId=top-100-liked)
&gt; 给你一个字符串 `s`，找到 `s` 中最长的 回文 子串。

```cpp
class Solution {
    static const int N = 1e3 &#43; 10;
    static const int p[N], b[N];
public:
    string longestPalindrome(string s) {
        // 用 manacher 算法求解
        int k = 0, n = 0;

        auto init = [&amp;]() {
            b[k &#43;&#43; ] = &#39;$&#39;, b[k &#43;&#43; ] = &#39;#&#39;;
            for (auto c : s) {
                b[k &#43;&#43; ] = c;
                b[k &#43;&#43; ] = &#39;#&#39;;
            }
            b[k &#43;&#43; ] = &#39;^&#39;;
            n = k;
        };

        init();

        auto manacher = [$]() {
            int mr = 0, mid = 0;
            for (int i = 1; i &lt; n; i &#43;&#43; ) {
                if (i &lt; mr) p[i] = max(2 * mid - i, mr - i);
                else p[i] = 1;

                while (b[i - p[i]] == b[i &#43; p[i]]) p[i] &#43;&#43; ;
                if (i &#43; p[i] &gt; mr) {
                    mr = i &#43; p[i];
                    mid = i;
                }
            }
        };

        manacher();

        // 如果只是求最长回文子串的长度，返回res即可
        int res = 0, max_i = 0;
        for (int i = 0; i &lt; n; i &#43;&#43; ) {
            if (p[i] &gt; res) {
                res = p[i];
                max_i = i;
            }
        }

        // 如果是求具体的回文子串，需要根据最长长度截取
        int start = max_i - (res - 1);
        int end = max_i &#43; (res - 1);
        for (int i = start; i &lt;= end; i &#43;&#43; ) {
            if (i == 0 || i == n - 1) continue;
            if (i % 2 == 0) ans.push_back(b[i]);
        }
        return ans;
    }
};
```


---

> 作者: yitao  
> URL: https://yitaonote.com/2025/a4da344/  

