# 《面试经典150题》


《面试经典150题》做题记录，[原题单链接](https://leetcode.cn/studyplan/top-interview-150/)

&lt;!--more--&gt;

## 数组/字符串

### 88.合并两个有序数组

&gt; 给你两个按 非递减顺序 排列的整数数组 `nums1` 和 `nums2`，另有两个整数 `m` 和 `n` ，分别表示 `nums1` 和 `nums2` 中的元素数目。
&gt; 请你 合并 `nums2` 到 `nums1` 中，使合并后的数组同样按 非递减顺序 排列。
&gt; 注意：最终，合并后数组不应由函数返回，而是存储在数组 `nums1` 中。为了应对这种情况，`nums1` 的初始长度为 `m &#43; n`，其中前 `m` 个元素表示应合并的元素，后 `n` 个元素为 0 ，应忽略。`nums2` 的长度为 n 。

&gt; [!TIP] 腾讯PCG-青云 一、二面手撕

```cpp
class Solution {
public:
    void merge(vector&lt;int&gt;&amp; nums1, int m, vector&lt;int&gt;&amp; nums2, int n) {
        int p1 = m - 1, p2 = n - 1, p = m &#43; n - 1;
        while (p2 &gt;= 0) {
            if (p1 &gt;= 0 &amp;&amp; nums1[p1] &gt; nums2[p2]) {
                nums1[p -- ] = nums1[p1 -- ];
            } else {
                nums1[p -- ] = nums2[p2 -- ];
            }
        }
    }
};
```

### 27.移除元素

&gt;给你一个数组 `nums` 和一个值 `val`，你需要 原地 移除所有数值等于 `val` 的元素。元素的顺序可能发生改变。然后返回 `nums` 中与 `val` 不同的元素的数量。
&gt;假设 `nums` 中不等于 `val` 的元素数量为 `k`，要通过此题，您需要执行以下操作：
&gt;更改 `nums` 数组，使 `nums` 的前 `k` 个元素包含不等于 `val` 的元素。`nums` 的其余元素和 `nums` 的大小并不重要。
返回 `k`。

```cpp
// 用栈存储去除后的元素
class Solution {
public:
    int removeElement(vector&lt;int&gt;&amp; nums, int val) {
        int top = 0;
        for (int x : nums) {
            if (x != val) {
                nums[top &#43;&#43; ] = x;
            }
        }
        return top;
    }
};
```

### 26.删除有序数组中的重复项

&gt;给你一个 非严格递增排列 的数组 `nums` ，请你 原地 删除重复出现的元素，使每个元素 只出现一次 ，返回删除后数组的新长度。元素的 相对顺序 应该保持 一致 。然后返回 `nums` 中唯一元素的个数。
&gt;考虑 `nums` 的唯一元素的数量为 `k` ，你需要做以下事情确保你的题解可以被通过：
&gt;更改数组 `nums` ，使 `nums` 的前 `k` 个元素包含唯一元素，并按照它们最初在 `nums` 中出现的顺序排列。`nums` 的其余元素与 `nums` 的大小不重要。
返回 `k` 。

```cpp
class Solution {
public:
    int removeDuplicates(vector&lt;int&gt;&amp; nums) {
        int siz = nums.size();
        int top = 0;

        for (int i = 0; i &lt; siz; i &#43;&#43; ) {
            int x = nums[i];
            if (i &amp;&amp; x == nums[i - 1]) continue; // 跳过重复数字
            nums[top &#43;&#43; ] = x;
        }
        return top;
    }
};
```

### 80.删除有序数组中的重复项 II

&gt;给你一个有序数组 `nums` ，请你 **原地** 删除重复出现的元素，使得出现次数超过两次的元素只出现两次 ，返回删除后数组的新长度。
&gt;不要使用额外的数组空间，你必须在 **原地** 修改输入数组 并在使用 $O(1)$ 额外空间的条件下完成。

```cpp
// 用栈模拟, top表示栈顶, 栈存储全部不重复元素
class Solution {
public:
    int removeDuplicates(vector&lt;int&gt;&amp; nums) {
        int top = 2;
        int siz = nums.size();

        for (int i = 2; i &lt; siz; i &#43;&#43; ) {
            if (nums[i] != nums[top - 2]) nums[top &#43;&#43; ] = nums[i];
        }
        return min(top, siz);
    }
};
```

### 169.多数元素

&gt;给定一个大小为 `n` 的数组 `nums` ，返回其中的多数元素。多数元素是指在数组中出现次数 大于 `⌊ n/2 ⌋`` 的元素。
&gt;你可以假设数组是非空的，并且给定的数组总是存在多数元素。

```cpp
// 摩尔投票法
class Solution {
public:
    int majorityElement(vector&lt;int&gt;&amp; nums) {    
        int x = 0, votes = 0;
        for (int num: nums) {
            if (votes == 0) x = num;
            votes &#43;= num == x ? 1 : -1;
        }
        return x;
    }
};
```

### 189.轮转数组

&gt; 给定一个整数数组 `nums`，将数组中的元素向右轮转 `k` 个位置，其中 `k` 是非负数。

&gt;示例 1: &lt;br&gt;
输入: nums = [1,2,3,4,5,6,7], k = 3 &lt;br&gt;
输出: [5,6,7,1,2,3,4] &lt;br&gt;
解释: &lt;br&gt;
向右轮转 1 步: [7,1,2,3,4,5,6] &lt;br&gt;
向右轮转 2 步: [6,7,1,2,3,4,5] &lt;br&gt;
向右轮转 3 步: [5,6,7,1,2,3,4] &lt;br&gt;

&gt;示例 2: &lt;br&gt;
输入：nums = [-1,-100,3,99], k = 2 &lt;br&gt;
输出：[3,99,-1,-100] &lt;br&gt;
解释: &lt;br&gt;
向右轮转 1 步: [99,-1,-100,3] &lt;br&gt;
向右轮转 2 步: [3,99,-1,-100]

```cpp
class Solution {
public:
    void rotate(vector&lt;int&gt;&amp; nums, int k) {
        // 根据 k 计算出每个元素轮转后的位置，然后填入新的 vector 中
        int n = nums.size();
        k %= n;

        std::reverse(nums.begin(), nums.end());
        std::reverse(nums.begin(), nums.begin() &#43; k);
        std::reverse(nums.begin() &#43; k, nums.end());
    }
};
```

### 121.买卖股票的最佳时机

&gt; 给定一个数组 `prices` ，它的第 `i` 个元素 `prices[i]` 表示一支给定股票第 `i` 天的价格。&lt;br&gt;
&gt; 你只能选择 某一天 买入这只股票，并选择在 未来的某一个不同的日子 卖出该股票。设计一个算法来计算你所能获取的最大利润。&lt;br&gt;
&gt; 返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 `0` 。

&gt; [!TIP] 状态机模型

```cpp
class Solution {
public:
    int maxProfit(vector&lt;int&gt;&amp; prices) {
        // 买入-&gt;卖出，求区间端点差值的最大值
        int ans = 0;
        int min_price = prices[0];
        for (int p : prices) {
            ans = max(ans, p - min_price);
            min_price = min(min_price, p);
        }
        return ans;
    }
};
```

### 122.买卖股票的最佳时机II

&gt; 给你一个整数数组 `prices` ，其中 `prices[i]` 表示某支股票第 `i` 天的价格。&lt;br&gt;
&gt; 在每一天，你可以决定是否购买和/或出售股票。你在任何时候 最多 只能持有 一股 股票。你也可以先购买，然后在 同一天 出售。&lt;br&gt;
&gt; 返回 你能获得的 最大 利润 。

&gt; [!TIP] [状态划分参考](https://www.acwing.com/file_system/file/content/whole/index/content/12264398/)


```cpp
class Solution {
public:
    int maxProfit(vector&lt;int&gt;&amp; prices) {
        // 状态机模型  
        // f[i][j]表示所有考虑前 i 个步骤，且第 i 个状态是 j(0未持股, 1持股)的集合，属性是最大值
        // 对于f[i][j]
        // 如果i-1步是0，0-&gt;0（未持股且不买入）；0-&gt;1（未持股且买入）；
        // 如果i-1步是1，1-&gt;0（持股且卖出）；1-&gt;1（持股且不卖出）        

        int n = prices.size();
        int INF = 0x3f3f3f3f;

        vector&lt;vector&lt;int&gt;&gt; f(n &#43; 1, vector&lt;int&gt;(2, 0));  // f[n][2]
        prices.insert(prices.begin(), 0);

        f[0][0] = 0, f[0][1] = -INF;
        for (int i = 1; i &lt;= n; i &#43;&#43; ) {
            f[i][0] = max(f[i - 1][0], f[i - 1][1] &#43; prices[i]);
            f[i][1] = max(f[i - 1][1], f[i - 1][0] - prices[i]);
        }

        return max(f[n][0], f[n][1]);
    }
};
```

### 55.跳跃游戏

&gt; 给你一个非负整数数组 `nums` ，你最初位于数组的 第一个下标 。数组中的每个元素代表你在该位置可以跳跃的最大长度。&lt;br&gt;
&gt; 判断你是否能够到达最后一个下标，如果可以，返回 `true` ；否则，返回 `false` 。

```cpp
class Solution {
public:
    bool canJump(vector&lt;int&gt;&amp; nums) {
        // 只要跳到了不为 0 的格子上，就一直可以往后跳
        // 转为合并区间问题
        int mx = 0;
        for (int i = 0; i &lt; nums.size(); i &#43;&#43; ) {
            if (i &gt; mx) return false;
            mx = max(mx, i &#43; nums[i]);
        }
        return true;
    }
};
```

### 45.跳跃游戏II

&gt; 给定一个长度为 `n` 的 `0` 索引整数数组 `nums`。初始位置为 `nums[0]`。&lt;br&gt;
&gt; 每个元素 `nums[i]` 表示从索引 `i` 向后跳转的最大长度。换句话说，如果你在 `nums[i]` 处，你可以跳转到任意 `nums[i &#43; j]` 处: &lt;br&gt;
&gt; - `0 &lt;= j &lt;= nums[i]` &lt;br&gt;
&gt; - `i &#43; j &lt; n` &lt;br&gt;
&gt; 返回到达 `nums[n - 1]` 的最小跳跃次数。生成的测试用例可以到达 `nums[n - 1]`。

```cpp
class Solution {
public:
    int jump(vector&lt;int&gt;&amp; nums) {
        int ans = 0;
        int cur_right = 0; // 已建造的桥的右端点
        int next_right = 0; // 下一座桥的右端点的最大值

        for (int i = 0; i &#43; 1 &lt; nums.size(); i &#43;&#43; ) {
            // 遍历的过程中，记录下一座桥的最远点
            next_right = max(next_right, i &#43; nums[i]);
            if (i == cur_right) {  // 无路可走，必须建桥
                cur_right = next_right;  // 建桥后，最远可以到达 next_right
                ans &#43;&#43; ;
            }
        }
        return ans;
    }
};
```

### 274.H指数

&gt;给你一个整数数组 `citations` ，其中 `citations[i]` 表示研究者的第 `i` 篇论文被引用的次数。计算并返回该研究者的 `h` 指数。
&gt;根据维基百科上 `h` 指数的定义：`h` 代表“高引用次数” ，一名科研人员的 `h` 指数 是指他（她）至少发表了 `h` 篇论文，并且 至少 有 `h` 篇论文被引用次数大于等于 `h` 。如果 `h` 有多种可能的值，`h` 指数 是其中最大的那个。

&gt; 即：给你一个数组，求一个最大的 $h$，使得数组中有至少 $h$ 个数都大于等于 $h$。

```cpp
class Solution {
public:

    int hIndex(vector&lt;int&gt;&amp; citations) {
        int n = citations.size();

        auto check = [&amp;](int mid) -&gt; bool {
            int n = citations.size();

            int res = 0;
            for (int i = 0; i &lt; n; i &#43;&#43; ) {
                if (citations[i] &lt; mid) res &#43;&#43; ;
            }
            return (n - res) &gt;= mid;
        };

        int l = -1, r = n &#43; 1;
        while (l &#43; 1 &lt; r) {
            int mid = l &#43; (r - l) / 2;
            if (check(mid)) l = mid;
            else r = mid;
        }
        return l;
    }
};
```

### 380. O(1) 时间插入、删除和获取随机元素

&gt;实现RandomizedSet 类：

&gt;- `RandomizedSet()` 初始化 `RandomizedSet` 对象
&gt;- `bool insert(int val)` 当元素 `val` 不存在时，向集合中插入该项，并返回 `true` ；否则，返回 `false` 。
&gt;- `bool remove(int val)` 当元素 `val` 存在时，从集合中移除该项，并返回 `true` ；否则，返回 `false` 。
&gt;- `int getRandom()` 随机返回现有集合中的一项（测试用例保证调用此方法时集合中至少存在一个元素）。每个元素应该有 相同的概率 被返回。
你必须实现类的所有函数，并满足每个函数的 平均 时间复杂度为 `O(1)` 。

```cpp
// 哈希表 &#43; 变长数组
class RandomizedSet {
public:
    RandomizedSet() {
        srand((unsigned)time(NULL));
    }

    bool insert(int val) {
        if (indices.count(val)) {
            return false;
        }
        int index = nums.size();
        nums.emplace_back(val);
        indices[val] = index;
        return true;
    }

   // 主要是这里，删除默认让尾部的值覆盖要删除的元素，然后erase掉指定的值
    bool remove(int val) {
        if (!indices.count(val)) return false;
        int index = indices[val];
        int last = nums.back();
        nums[index] = last;
        indices[last] = index;
        nums.pop_back();
        indices.erase(val);
        return true;
    }

    int getRandom() {
        int randomIndex = rand() % nums.size();
        return nums[randomIndex];
    }
private:
    vector&lt;int&gt; nums;
    unordered_map&lt;int, int&gt; indices;
};

/**
 * Your RandomizedSet object will be instantiated and called as such:
 * RandomizedSet* obj = new RandomizedSet();
 * bool param_1 = obj-&gt;insert(val);
 * bool param_2 = obj-&gt;remove(val);
 * int param_3 = obj-&gt;getRandom();
 */
```

### 238.除自身以外数组的乘积

&gt; 给你一个整数数组 `nums`，返回 数组 `answer` ，其中 `answer[i]` 等于 `nums` 中除 `nums[i]` 之外其余各元素的乘积 。题目数据 保证 数组 `nums` 之中任意元素的全部前缀元素和后缀的乘积都在 32位 整数范围内。请不要使用除法，且在 `O(n)` 时间复杂度内完成此题。

```cpp
class Solution {
public:
    vector&lt;int&gt; productExceptSelf(vector&lt;int&gt;&amp; nums) {

        // 用“前缀数组”和“后缀数组”完成
        int n = nums.size();
        vector&lt;int&gt; pre(n &#43; 1, 1);
        vector&lt;int&gt; suf(n &#43; 1, 1);

        vector&lt;int&gt; ans(n);

        for (int i = 1; i &lt;= n; i &#43;&#43; ) {
            pre[i] = pre[i - 1] * nums[i - 1];
        }

        for (int i = n - 1; i &gt;= 1; i -- ) {
            suf[i] = suf[i &#43; 1] * nums[i];
        }

        for (int i = 0; i &lt; n; i &#43;&#43; ) {
            ans[i] = pre[i] * suf[i &#43; 1];
        }
        return ans;
    }
};
```

### 134.加油站

&gt; 在一条环路上有 `n` 个加油站，其中第 `i` 个加油站有汽油 `gas[i]` 升。 &lt;br&gt;
&gt; 你有一辆油箱容量无限的的汽车，从第 `i` 个加油站开往第 `i&#43;1` 个加油站需要消耗汽油 `cost[i]` 升。你从其中的一个加油站出发，开始时油箱为空。&lt;br&gt;
&gt; 给定两个整数数组 `gas` 和 `cost` ，如果你可以按顺序绕环路行驶一周，则返回出发时加油站的编号，否则返回 `-1` 。如果存在解，则 保证 它是 唯一 的。

&gt; [!TIP] 本题是特殊做法，通用做法是==单调队列==，详情见：[AcWing.1088](https://www.acwing.com/problem/content/1090/)

```cpp
class Solution {
public:
    int canCompleteCircuit(vector&lt;int&gt;&amp; gas, vector&lt;int&gt;&amp; cost) {
        int n = gas.size();

        for (int i = 0, j = 0; i &lt; n;) {  // 枚举起点
            int left = 0;
            for (j = 0; j &lt; n; j &#43;&#43; ) {  // 枚举走了几步
                int k = (i &#43; j) % n;
                left &#43;= gas[k] - cost[k];
                if (left &lt; 0) break;  // 如果剩余油量不够，则退出枚举，这里有个贪心思想，i~j 之间不用枚举
            }
            if (j == n) return i;
            i = i &#43; j &#43; 1;
        }
        return -1;
    }
};
```

### 135.分发糖果

&gt;n 个孩子站成一排。给你一个整数数组 ratings 表示每个孩子的评分。&lt;br&gt;
&gt;你需要按照以下要求，给这些孩子分发糖果：
&gt;- 每个孩子至少分配到 1 个糖果。
&gt;- 相邻两个孩子中，评分更高的那个会获得更多的糖果。
&gt;请你给每个孩子分发糖果，计算并返回需要准备的 最少糖果数目 。

```cpp
class Solution {
public:
    int candy(vector&lt;int&gt;&amp; ratings) {
        int n = ratings.size();
        int ans = n;  // 每个孩子至少1个糖果
        for (int i = 0; i &lt; n; i &#43;&#43; ) {
            // 找起始点，满足递增才可以作为起始点
            int start = i &gt; 0 &amp;&amp; ratings[i - 1] &lt; ratings[i] ? i - 1 : i;

            // 找严格递增段
            while (i &#43; 1 &lt; n &amp;&amp; ratings[i] &lt; ratings[i &#43; 1]) {
                i &#43;&#43; ;
            }
            // 循环结束时，i 为峰顶
            int top = i;

            // 找严格递减段
            while (i &#43; 1 &lt; n &amp;&amp; ratings[i] &gt; ratings[i &#43; 1]) {
                i &#43;&#43; ;
            }
            // 循环结束时，i 为谷底
            int inc = top - start;  // start 到 top 严格递增
            int dec = i - top;  // top 到 i 严格递减

            ans &#43;= (inc * (inc - 1) &#43; dec * (dec - 1)) / 2 &#43; max(inc, dec);  // 等差数列公式，由于求最少糖果数，所以公差为1
        }
        return ans;
    }
};
```

### 42.接雨水

&gt; 给定 `n` 个非负整数表示每个宽度为 `1` 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。

![123](https://cdn.ipfsscan.io/weibo/large/005wRZF3ly1i428m5hjijj30bg04hmx7.jpg)

```cpp
class Solution {
public:
    int trap(vector&lt;int&gt;&amp; height) {
        int n = height.size(), pre_max = 0, suf_max = 0;  // pre_max之前最高的柱子高度，suf_max之后最高的柱子高度

        // 注意到下标 i 处能接的雨水量由 pre_max[i] 和 suf_max[i] 中的最小值决定。
        int left = 0, right = n - 1, res = 0;
        while (left &lt; right) {
            pre_max = max(pre_max, height[left]);   // 维护pre_max
            suf_max = max(suf_max, height[right]);  // 维护suf_max

            if (pre_max &lt; suf_max) {
                res &#43;= pre_max - height[left];
                left &#43;&#43; ;
            } else {
                res &#43;= suf_max - height[right];
                right -- ;
            }
        }
        return res;
    }
};
```

### 13.罗马数字转整数

&gt;给你一个罗马数字，将其转换为整数

```cpp
unordered_map&lt;char, int&gt; ROMAN = {
    {&#39;I&#39;, 1},
    {&#39;V&#39;, 5},
    {&#39;X&#39;, 10},
    {&#39;L&#39;, 50},
    {&#39;C&#39;, 100},
    {&#39;D&#39;, 500},
    {&#39;M&#39;, 1000},
};

class Solution {
public:
    int romanToInt(string s) {
        int ans = 0;
        for (int i = 0; i &#43; 1 &lt; s.size(); i &#43;&#43; ) {
            int x = ROMAN[s[i]], y = ROMAN[s[i &#43; 1]];
            ans &#43;= x &lt; y ? -x : x;
        }
        return ans &#43; ROMAN[s.back()];
    }
};
```

### 12.整数转罗马数字

&gt;给你一个整数，将其转为罗马数字

```cpp
class Solution {
    static constexpr string R[4][10] = {
        {&#34;&#34;, &#34;I&#34;, &#34;II&#34;, &#34;III&#34;, &#34;IV&#34;, &#34;V&#34;, &#34;VI&#34;, &#34;VII&#34;, &#34;VIII&#34;, &#34;IX&#34;}, // 个位
        {&#34;&#34;, &#34;X&#34;, &#34;XX&#34;, &#34;XXX&#34;, &#34;XL&#34;, &#34;L&#34;, &#34;LX&#34;, &#34;LXX&#34;, &#34;LXXX&#34;, &#34;XC&#34;}, // 十位
        {&#34;&#34;, &#34;C&#34;, &#34;CC&#34;, &#34;CCC&#34;, &#34;CD&#34;, &#34;D&#34;, &#34;DC&#34;, &#34;DCC&#34;, &#34;DCCC&#34;, &#34;CM&#34;}, // 百位
        {&#34;&#34;, &#34;M&#34;, &#34;MM&#34;, &#34;MMM&#34;}, // 千位
    };

public:
    string intToRoman(int num) {
        return R[3][num / 1000] &#43; R[2][num / 100 % 10] &#43; R[1][num / 10 % 10] &#43; R[0][num % 10];
    }
};
```

### 58.最后一个单词的长度

&gt;给你一个字符串 s，由若干单词组成，单词前后用一些空格字符隔开。返回字符串中 最后一个 单词的长度。
单词 是指仅由字母组成、不包含任何空格字符的最大子字符串。

```cpp
class Solution {
public:
    int lengthOfLastWord(string s) {
        int i = s.length() - 1;
        while (s[i] == &#39; &#39; &amp;&amp; i &gt; 0) i -- ;
        int j = i - 1;
        while (j &gt;= 0 &amp;&amp; s[j] != &#39; &#39;) j -- ;
        return i - j;
    }
};
```

### 14.最长公共前缀

&gt; 编写一个函数来查找字符串数组中的最长公共前缀。&lt;br&gt;
&gt; 如果不存在公共前缀，返回空字符串 `&#34;&#34;`。

```cpp
class Solution {
public:
    string longestCommonPrefix(vector&lt;string&gt;&amp; strs) {
        string&amp; s0 = strs[0];
        for (int j = 0; j &lt; s0.size(); j &#43;&#43; ) {
            for (string&amp; s : strs) {
                if (j == s.size() || s[j] != s0[j]) {
                    return s0.substr(0, j);
                }
            }
        }
        return s0;
    }
};
```

### 151.反转字符串中的单词

&gt; 给你一个字符串 `s` ，请你反转字符串中 单词 的顺序。&lt;br&gt;
&gt; 单词 是由非空格字符组成的字符串。`s` 中使用至少一个空格将字符串中的 单词 分隔开。&lt;br&gt;
&gt; 返回 单词 顺序颠倒且 单词 之间用单个空格连接的结果字符串。&lt;br&gt;
&gt; 注意：输入字符串 `s` 中可能会存在前导空格、尾随空格或者单词间的多个空格。返回的结果字符串中，单词间应当仅用单个空格分隔，且不包含任何额外的空格。

```java
class Solution {
    public String reverseWords(String s) {
        s = s.trim();                                    // 删除首尾空格
        int j = s.length() - 1, i = j;
        StringBuilder res = new StringBuilder();
        while (i &gt;= 0) {
            while (i &gt;= 0 &amp;&amp; s.charAt(i) != &#39; &#39;) i--;     // 搜索首个空格
            res.append(s.substring(i &#43; 1, j &#43; 1) &#43; &#34; &#34;); // 添加单词
            while (i &gt;= 0 &amp;&amp; s.charAt(i) == &#39; &#39;) i--;     // 跳过单词间空格
            j = i;                                       // j 指向下个单词的尾字符
        }
        return res.toString().trim();                    // 转化为字符串并返回
    }
}
```

### 28.找出字符串中第一个匹配项的下标

&gt; 给你两个字符串 `haystack` 和 `needle` ，请你在 `haystack` 字符串中找出 `needle` 字符串的第一个匹配项的下标（下标从 `0` 开始）。如果 `needle` 不是 `haystack` 的一部分，则返回  `-1` 。

```cpp
class Solution {
public:
    int strStr(string haystack, string needle) {

        int m = haystack.size(), n = needle.size();
        for (int i = 0; i &lt; haystack.size(); i &#43;&#43; ) {
            if (i &#43; n &gt; m) return -1;
            // 判断两个区间的值是否相同
            if (haystack.substr(i, n) == needle) return i;
        }
        return -1;
    }
};
```

## 双指针

### 11.盛水最多的容器

&gt; 给定一个长度为 `n` 的整数数组 `height` 。有 `n` 条垂线，第 `i` 条线的两个端点是 `(i, 0)` 和 `(i, height[i])` 。
&gt; 找出其中的两条线，使得它们与 `x` 轴共同构成的容器可以容纳最多的水。
&gt; 返回容器可以储存的最大水量。
&gt; 说明：你不能倾斜容器。

![123](https://cdn.ipfsscan.io/weibo/large/005wRZF3ly1i428wopgwqj30m90an3yr.jpg)



```cpp
// 思路和接雨水类似
class Solution {
public:
    int maxArea(vector&lt;int&gt;&amp; height) {
        int n = height.size();
        int left = 0, right = n - 1;

        int ans = 0;
        while (left &lt; right) {
            ans = max(ans, min(height[left], height[right]) * (right - left));
            if (height[left] &lt; height[right]) {
                left &#43;&#43; ;
            } else {
                right -- ;
            }
        }
        return ans;
    }
};
```

### 15.三数之和

&gt; [🔗](https://leetcode.cn/problems/3sum?envType=study-plan-v2&amp;envId=top-interview-150)
&gt; 给你一个整数数组 `nums` ，判断是否存在三元组 `[nums[i], nums[j]`, `nums[k]]` 满足 `i != j`、`i != k` 且 `j != k` ，同时还满足 `nums[i] &#43; nums[j] &#43; nums[k] == 0` 。请你返回所有和为 `0` 且不重复的三元组。&lt;br&gt;
&gt; 注意：答案中不可以包含重复的三元组。

```cpp
class Solution {
public:
    vector&lt;vector&lt;int&gt;&gt; threeSum(vector&lt;int&gt;&amp; nums) {
        int n = nums.size();
        vector&lt;vector&lt;int&gt;&gt; ans;

        sort(nums.begin(), nums.end());

        // 本质上是过滤所有不可能的情况

        for (int i = 0; i &lt; n; i &#43;&#43; ) {
            if (i &amp;&amp; nums[i] == nums[i - 1]) continue;
            for (int j = i &#43; 1, k = n - 1; j &lt; k; j &#43;&#43; ) {
                if (j &gt; i &#43; 1 &amp;&amp; nums[j] == nums[j - 1]) continue;
                while (j &lt; k - 1 &amp;&amp; nums[i] &#43; nums[j] &#43; nums[k - 1] &gt;= 0) k -- ;
                if (nums[i] &#43; nums[j] &#43; nums[k] == 0) ans.push_back({nums[i], nums[j], nums[k]});
            }
        }
        return ans;
    }
};
```

## 滑动窗口

### 209.长度最小的子数组

&gt; 给定一个含有 `n` 个正整数的数组和一个正整数 `target` 。
&gt; 找出该数组中满足其总和大于等于 `target` 的长度最小的 子数组 `[nums_l, nums_l&#43;1, ..., nums_r-1, nums_r]` ，并返回其长度。如果不存在符合条件的子数组，返回 `0` 。

```cpp
class Solution {
public:
    int minSubArrayLen(int target, vector&lt;int&gt;&amp; nums) {
        // 预处理前缀和
        int n = nums.size();
        vector&lt;int&gt; s(n &#43; 1, 0);
        for (int i = 1; i &lt;= n; i &#43;&#43; ) s[i] = s[i - 1] &#43; nums[i - 1];

        // 枚举右指针，然后移动左指针
        int l = 1;  // l 不用回头
        int ans = n &#43; 1;
        for (int r = 1; r &lt;= n; r &#43;&#43; ) {          
            while ((s[r] - s[l - 1]) &gt;= target) {
                ans = min(ans, r - l &#43; 1);
                l &#43;&#43; ;
            }
        }
        return ans &lt;= n ? ans : 0;
    }
};
```

### 3.无重复字符的最长子串

&gt; 给定一个字符串 `s` ，请你找出其中不含有重复字符的 最长 子串 的长度。

```cpp
class Solution {
public:
    int lengthOfLongestSubstring(string s) {  
        // 滑动窗口,
        unordered_map&lt;char, int&gt; heap;  // 记录每个字符出现过多少次

        int res = 0;
        int left = 0;
        for (int i = 0; i &lt; s.size(); i &#43;&#43; ) {
            heap[s[i]] &#43;&#43; ;
            while (heap[s[i]] &gt; 1) {
                heap[s[left]] -- ;
                left &#43;&#43; ;
            }
            res = max(res, i - left &#43; 1);
        }
        return res;
    }
};
```

### 76.最小覆盖子串

&gt; 给你一个字符串 s 、一个字符串 t 。返回 s 中涵盖 t 所有字符的最小子串。如果 s 中不存在涵盖 t 所有字符的子串，则返回空字符串 &#34;&#34; 。

```cpp
class Solution {
    bool is_covered(int cnt_s[], int cnt_t[]) {
        for (int i = &#39;A&#39;; i &lt;= &#39;Z&#39;; i &#43;&#43; ) {
            if (cnt_s[i] &lt; cnt_t[i]) {
                return false;
            }
        }
        for (int i = &#39;a&#39;; i &lt;= &#39;z&#39;; i &#43;&#43; ) {
            if (cnt_s[i] &lt; cnt_t[i]) {
                return false;
            }
        }
        return true;
    }
public:
    string minWindow(string s, string t) {
        // 不定长滑动窗口
        int cnt_s[128]{};
        int cnt_t[128]{};
        int min_left = -1;
        int min_right = s.size();
        int ans = s.size();

        for (int i = 0; i &lt; t.size(); i &#43;&#43; ) cnt_t[t[i]] &#43;&#43; ;
        for (int i = 0, left = 0; i &lt; s.size(); i &#43;&#43; ) {
            cnt_s[s[i]] &#43;&#43; ;
            // 已经全覆盖了，右移左端点
            while (is_covered(cnt_s, cnt_t)) {
                if (i - left &lt; min_right - min_left) {
                    min_left = left;
                    min_right = i;
                }
                cnt_s[s[left]] -- ;
                left &#43;&#43; ;
            }
        }

        if (min_left &gt;= 0) {
            return s.substr(min_left, min_right - min_left &#43; 1);
        }
        return &#34;&#34;;
    }
};
```

## 矩阵

### 54.螺旋矩阵

&gt; 给你一个 `m` 行 `n` 列的矩阵 `matrix` ，请按照 顺时针螺旋顺序 ，返回矩阵中的所有元素。

![image](https://cdn.ipfsscan.io/weibo/large/005wRZF3ly1i429ce4gbyj306q06qweh.jpg)

```cpp
class Solution {
public:
    vector&lt;int&gt; spiralOrder(vector&lt;vector&lt;int&gt;&gt;&amp; matrix) {
        vector&lt;int&gt; res;

        int m = matrix.size();
        int n = matrix[0].size();

        vector&lt;vector&lt;int&gt;&gt; st(m &#43; 10, vector&lt;int&gt;(n &#43; 10, 0));

        // 用坐标偏移法模拟
        int dx[] = {0, 1, 0, -1}, dy[] = {1, 0, -1, 0};  // 右，下，左，上
        int step = 0;  // 0, 1, 2, 3

        // m 行，n 列
        st[0][0] = 1;
        res.push_back(matrix[0][0]);
        int cnt = m * n - 1;

        int i = 0, j = 0;
        while (cnt -- ) {
            int x = i &#43; dx[step];
            int y = j &#43; dy[step];

            // 判断将要走的点有没有越过边界
            if (x &gt;= m || x &lt; 0 || y &lt; 0 || y &gt;= n || st[x][y] == 1) {
                step = (step &#43; 1) % 4;    
                x = i &#43; dx[step];
                y = j &#43; dy[step];
            }

            res.push_back(matrix[x][y]);

            i = x;
            j = y;
        }


        return res;
    }
};
```

### 48.旋转图像

&gt; 给定一个 `n × n` 的二维矩阵 `matrix` 表示一个图像。请你将图像顺时针旋转 `90` 度。&lt;br&gt;
&gt; 你必须在 原地 旋转图像，这意味着你需要直接修改输入的二维矩阵。请不要 使用另一个矩阵来旋转图像。

![image](https://cdn.ipfsscan.io/weibo/005wRZF3ly1i429d0n6zyj30hu06qaa8.jpg)

```cpp
class Solution {
public:
    void rotate(vector&lt;vector&lt;int&gt;&gt;&amp; matrix) {
        // 先上下颠倒，再矩阵转置

        int m = matrix.size();
        int n = matrix[0].size();

        for (int i = 0; i &lt; m / 2; i &#43;&#43; ) {
            for (int j = 0; j &lt; n; j &#43;&#43; ) {
                swap(matrix[i][j], matrix[m - 1 - i][j]);
            }
        }

        // 矩阵转置通用公式
        for (int i = 0; i &lt; m; i &#43;&#43; ) {
            for (int j = i &#43; 1; j &lt; n; j &#43;&#43; ) {
                swap(matrix[i][j], matrix[j][i]);
            }
        }
    }
};
```

## 哈希表

### 128.最长连续序列

&gt; 给定一个未排序的整数数组 `nums` ，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。请你设计并实现时间复杂度为 `O(n)` 的算法解决此问题。

```cpp
class Solution {
public:
    int longestConsecutive(vector&lt;int&gt;&amp; nums) {
        int ans = 0;
        unordered_set&lt;int&gt; st(nums.begin(), nums.end());  // 把 nums 转为哈希集合
        for (int x : st) {  // 遍历哈希集合
            if (st.contains(x - 1)) {
                continue;  // 如果 x-1 在哈希集合中，则不以 x 为起点，因为 x-1 为起点计算出来的连续序列一定更长
            }
            // x 是序列的起点
            int y = x &#43; 1;
            while (st.contains(y)) {  // 不断查找下一个数是否在哈希集合中
                y &#43;&#43; ;
            }
            ans = max(ans, y - x);  // 从 x 到 y - 1 一共 y - x 个数
        }
        return ans;
    }
};
```

## 链表

### 92.反转链表 II

&gt; 给你单链表的头指针 `head` 和两个整数 `left` 和 `right` ，其中 `left &lt;= right` 。请你反转从位置 `left` 到位置 `right` 的链表节点，返回 反转后的链表 。

![image](https://cdn.ipfsscan.io/weibo/005wRZF3ly1i429nsf7nij30f2066dg3.jpg)

```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* reverseBetween(ListNode* head, int left, int right) {
        ListNode dummy{0, head};

        ListNode* p0 = &amp;dummy;

        // 首先维护p0指针，p0指针是待处理段的前一个指针（哨兵节点）
        for (int i = 0; i &lt; left - 1; i &#43;&#43; ) {
            p0 = p0-&gt;next;
        }

        ListNode* pre = nullptr;
        ListNode* cur = p0-&gt;next;

        for (int i = 0; i &lt; right - left &#43; 1; i &#43;&#43; ) {
            ListNode* nxt = cur-&gt;next;
            cur-&gt;next = pre;
            pre = cur;
            cur = nxt;
        }

        p0-&gt;next-&gt;next = cur;
        p0-&gt;next = pre;

        return dummy.next;
    }
};
```

### 25.K 个一组翻转链表

&gt; [🔗](https://leetcode.cn/problems/reverse-nodes-in-k-group/description/?envType=study-plan-v2&amp;envId=top-interview-150)
&gt; 给你链表的头节点 `head` ，每 `k` 个节点一组进行翻转，请你返回修改后的链表。&lt;br&gt;
&gt; `k` 是一个正整数，它的值小于或等于链表的长度。如果节点总数不是 `k` 的整数倍，那么请将最后剩余的节点保持原有顺序。&lt;br&gt;
&gt; 你不能只是单纯的改变节点内部的值，而是需要实际进行节点交换。

![image](https://assets.leetcode.com/uploads/2020/10/03/reverse_ex1.jpg)

```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* reverseKGroup(ListNode* head, int k) {
        // 统计节点个数
        int n = 0;
        for (ListNode* cur = head; cur; cur = cur-&gt;next) n &#43;&#43; ;

        ListNode dummy(0, head);  // 哨兵节点，哨兵的下一个节点是head
        ListNode* p0 = &amp;dummy;

        ListNode* pre = nullptr;
        ListNode* cur = head;

        // k 个一组进行处理
        for (; n &gt;= k; n -= k) {
            // 每组内部就是 反转链表II
            for (int i = 0; i &lt; k; i &#43;&#43; ) {
                ListNode* nxt = cur-&gt;next;  // cur不断往右走的同时，维护pre指针和nxt指针
                cur-&gt;next = pre;
                pre = cur;
                cur = nxt;
            }

            // 处理p0指针，p0指针主要是指向每一段被处理链表的哨兵节点（前一个节点）
            ListNode* nxt = p0-&gt;next;
            p0-&gt;next-&gt;next = cur;
            p0-&gt;next = pre;
            p0 = nxt;
        }
        return dummy.next;
    }
};
```

### 19.删除链表的倒数第 N 个结点

&gt; [🔗](https://leetcode.cn/problems/remove-nth-node-from-end-of-list/description/?envType=study-plan-v2&amp;envId=top-interview-150)
&gt; 给你一个链表，删除链表的倒数第 `n` 个结点，并且返回链表的头结点。

![image](https://assets.leetcode.com/uploads/2020/10/03/remove_ex1.jpg)

```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        ListNode dummy{0, head};
        ListNode* left = &amp;dummy;
        ListNode* right = &amp;dummy;

        // 左右指针都先往右走n步
        while (n -- ) {
            left = left-&gt;next;
            right = right-&gt;next;
        }

        // 再同时走一段距离，让右指针指向最后一个节点
        while (right-&gt;next) {
            left = left-&gt;next;
            right = right-&gt;next;
        }

        // 此时 left 下一个节点就是倒数第 n 个节点
        ListNode* nxt = left-&gt;next;
        left-&gt;next = left-&gt;next-&gt;next;

        delete nxt;
        return dummy.next;
    }
};
```

### 82.删除排序链表中的重复元素II

&gt; [🔗](https://leetcode.cn/problems/remove-duplicates-from-sorted-list-ii/description/?envType=study-plan-v2&amp;envId=top-interview-150)
&gt; 给定一个已排序的链表的头 `head` ， 删除原始链表中所有重复数字的节点，只留下不同的数字 。返回 已排序的链表

![image](https://assets.leetcode.com/uploads/2021/01/04/linkedlist1.jpg)


```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* deleteDuplicates(ListNode* head) {
        if (head == nullptr) return nullptr;

        ListNode dummy{0, head};
        ListNode* cur = &amp;dummy;

        while (cur-&gt;next &amp;&amp; cur-&gt;next-&gt;next) {
            if (cur-&gt;next-&gt;val == cur-&gt;next-&gt;next-&gt;val) {
                int x = cur-&gt;next-&gt;val;
                while (cur-&gt;next &amp;&amp; cur-&gt;next-&gt;val == x) {
                    cur-&gt;next = cur-&gt;next-&gt;next;
                }
            } else {
                cur = cur-&gt;next;
            }
        }
        return dummy.next;
    }
};
```

### 146.LRU 缓存

&gt; [🔗](https://leetcode.cn/problems/lru-cache/description/?envType=study-plan-v2&amp;envId=top-interview-150)
&gt; 请你设计并实现一个满足 `LRU` (最近最少使用) 缓存 约束的数据结构。&lt;br&gt;
&gt; 实现 `LRUCache` 类：&lt;br&gt;
&gt; - `LRUCache(int capacity)` 以 正整数 作为容量 `capacity` 初始化 `LRU` 缓存 &lt;br&gt;
&gt; - `int get(int key)` 如果关键字 `key` 存在于缓存中，则返回关键字的值，否则返回 `-1` 。&lt;br&gt;
&gt; - `void put(int key, int value)` 如果关键字 `key` 已经存在，则变更其数据值 `value` ；如果不存在，则向缓存中插入该组 `key-value` 。如果插入操作导致关键字数量超过 `capacity` ，则应该 逐出 最久未使用的关键字。

&gt; 函数 `get` 和 `put` 必须以 `O(1)` 的平均时间复杂度运行。

&gt; [!TIP] 用图巧记
&gt; ![image](https://pic.leetcode.cn/1696039105-PSyHej-146-3-c.png)

```cpp
struct Node {
    int key;
    int value;
    Node* prev;
    Node* next;

    Node(int k = 0, int v = 0) : key(k), value(v) {}
};

class LRUCache {
private:
    int capacity;
    Node* dummy;
    unordered_map&lt;int, Node*&gt; key_to_node;

    void remove(Node* x) {
        x-&gt;prev-&gt;next = x-&gt;next;
        x-&gt;next-&gt;prev = x-&gt;prev;
    }

    void push_front(Node* x) {
        x-&gt;next = dummy-&gt;next;
        x-&gt;prev = dummy;
        x-&gt;prev-&gt;next = x;
        x-&gt;next-&gt;prev = x;
    }

    Node* get_node(int key) {
        auto it = key_to_node.find(key);
        if (it == key_to_node.end()) return nullptr;
        Node* node = key_to_node[key];
        remove(node);
        push_front(node);
        return node;
    }

public:
    LRUCache(int capacity) : capacity(capacity), dummy(new Node()) {
        dummy-&gt;prev = dummy;
        dummy-&gt;next = dummy;
    }

    int get(int key) {
        Node* node = get_node(key);
        return node ? node-&gt;value : -1;
    }

    void put(int key, int value) {
        Node* node = get_node(key);
        if (node) {
            node-&gt;value = value;
            return;
        }
        node = new Node(key, value);
        key_to_node[key] = node;
        push_front(node);

        if (key_to_node.size() &gt; capacity) {
            Node* back_node = dummy-&gt;prev;
            key_to_node.erase(back_node-&gt;key);
            remove(back_node);
            delete back_node;
        }
    }
};
```

## 堆

### 215.数组中的第K个最大元素

&gt; [🔗](https://leetcode.cn/problems/kth-largest-element-in-an-array/description/?envType=study-plan-v2&amp;envId=top-interview-150)
&gt; 给定整数数组 `nums` 和整数 `k`，请返回数组中第 `k` 个最大的元素。&lt;br&gt;
&gt; 请注意，你需要找的是数组排序后的第 `k` 个最大的元素，而不是第 `k` 个不同的元素。&lt;br&gt;
&gt; 你必须设计并实现时间复杂度为 `O(n)` 的算法解决此问题。

```cpp
class Solution {
public:
    int findKthLargest(vector&lt;int&gt;&amp; nums, int target) {
        // 第 K 个最大元素
        // 1 2 3 4，n=4, K=2，则
        auto quick_select = [&amp;](this auto&amp;&amp;quick_select, int l, int r, int k) {
            if (l &gt;= r) return nums[l];

            int x = nums[(l &#43; r) / 2], i = l - 1, j = r &#43; 1;
            while (i &lt; j) {
                do i &#43;&#43; ; while (nums[i] &lt; x);
                do j -- ; while (nums[j] &gt; x);
                if (i &lt; j) swap(nums[i], nums[j]);
            }
            int sl = j - l &#43; 1;
            if (k &lt;= sl) return quick_select(l, j, k);
            return quick_select(j &#43; 1, r, k - sl);
        };

        int n = nums.size();

        return quick_select(0, n - 1, n - target &#43; 1);
    }
};
```

### 295.数据流的中位数

&gt; [🔗](https://leetcode.cn/problems/find-median-from-data-stream/description/?envType=study-plan-v2&amp;envId=top-interview-150)
&gt; 中位数是有序整数列表中的中间值。如果列表的大小是偶数，则没有中间值，中位数是两个中间值的平均值。&lt;br&gt;
&gt; - 例如 `arr = [2,3,4]` 的中位数是 `3` 。&lt;br&gt;
&gt; - 例如 `arr = [2,3]` 的中位数是 `(2 &#43; 3) / 2 = 2.5` 。&lt;br&gt;
&gt; 实现 `MedianFinder` 类: &lt;br&gt;
&gt; - `MedianFinder()` 初始化 `MedianFinder` 对象。&lt;br&gt;
&gt; - `void addNum(int num)` 将数据流中的整数 `num` 添加到数据结构中。&lt;br&gt;
&gt; `double findMedian()` 返回到目前为止所有元素的中位数。与实际答案相差 `10^-5` 以内的答案将被接受。

```cpp
class MedianFinder {
    priority_queue&lt;int, vector&lt;int&gt;, greater&lt;int&gt;&gt; up;  // 小根堆
    priority_queue&lt;int&gt; down;  // 大根堆
    int siz;
public:
    MedianFinder() {
        siz = 0;  // 记录对顶堆中元素大小
    }

    // 对顶堆维护动态中位数

    void addNum(int num) {
        if (down.empty() || num &lt;= down.top()) down.push(num);
        else up.push(num);

        siz &#43;&#43; ;

        if (down.size() &gt; up.size() &#43; 1) up.push(down.top()), down.pop();
        if (up.size() &gt; down.size()) down.push(up.top()), up.pop();
    }

    double findMedian() {
        if (siz % 2) return down.top();
        return (up.top() &#43; down.top()) / 2.0;
    }
};

/**
 * Your MedianFinder object will be instantiated and called as such:
 * MedianFinder* obj = new MedianFinder();
 * obj-&gt;addNum(num);
 * double param_2 = obj-&gt;findMedian();
 */
```



## 一维动态规划

### 139.单词拆分

&gt; [🔗](https://leetcode.cn/problems/word-break/description/?envType=study-plan-v2&amp;envId=top-interview-150)
&gt; 给你一个字符串 `s` 和一个字符串列表 `wordDict` 作为字典。如果可以利用字典中出现的一个或多个单词拼接出 `s` 则返回 `true`。&lt;br&gt;
&gt; 注意：不要求字典中出现的单词全部都使用，并且字典中的单词可以重复使用。

&gt; [!TIP] 这里用到了[==字符串哈希==](https://www.acwing.com/activity/content/code/content/1410741/)来优化
&gt; 这里有一个隐藏的细节是 “秦九韶算法”，即哈希值的维护。如果不这样写则需要维护一个 `p` 数组，来进行 `a*P^3 &#43; b*P^2`


```cpp
class Solution {
public:
    bool wordBreak(string s, vector&lt;string&gt;&amp; wordDict) {
        typedef unsigned long long ULL;  // 用ULL表示是因为为了对 2^64 取模
        unordered_set&lt;ULL&gt; hash;
        const int P = 131;  // P进制的经验值，也可以取 13331，可以认为99%的概率不会哈希冲突

        for (auto&amp; s : wordDict) {
            ULL h = 0;
            for (auto c : s) {
                h = h * P &#43; c;  // 将词表中的每个词映射至 P 进制，秦九韶算法写法；
            }
            hash.insert(h);
        }

        int n = s.size();
        vector&lt;bool&gt; f(n &#43; 1);

        s = &#34; &#34; &#43; s;
        f[0] = true;  // f[i]表示单词 s 的前 i 个字符能否由 wordDict 中的单词组成，其中边界 f[0] = true
        for (int i = 0; i &lt; n; i &#43;&#43; ) {
            if (f[i]) {  // 如果 f[i] = true 并且 s[i &#43; 1:j] 也在 wordDict 中，则 f[j] = true
                ULL h = 0;
                for (int j = i &#43; 1; j &lt;= n; j &#43;&#43; ) {  // 查询 s[i &#43; 1:j] 中所有的字符串是否在 wordDict 中出现过
                    h = h * P &#43; s[j];
                    if (hash.count(h)) f[j] = true;  
                }
            }
        }
        return f[n];
    }
};
```

## 多维动态规划

### 120. 三角形最小路径和

&gt; [🔗](https://leetcode.cn/problems/triangle/description/?envType=study-plan-v2&amp;envId=top-interview-150)
&gt; 给定一个三角形 `triangle` ，找出自顶向下的最小路径和。&lt;br&gt;
&gt; 每一步只能移动到下一行中相邻的结点上。相邻的结点 在这里指的是 下标 与 上一层结点下标 相同或者等于 上一层结点下标 &#43; 1 的两个结点。也就是说，如果正位于当前行的下标 `i` ，那么下一步可以移动到下一行的下标 `i` 或 `i &#43; 1` 。

```cpp
class Solution {
public:
    int minimumTotal(vector&lt;vector&lt;int&gt;&gt;&amp; grid) {
        int n = grid.size();
        const int INF = 0x3f3f3f3f;

        vector&lt;vector&lt;int&gt;&gt; f(n &#43; 1, vector&lt;int&gt;(n &#43; 1, -1));

        auto dfs = [&amp;](this auto&amp;&amp; dfs, int x, int y) {
            int n = grid.size();

            if (x == n - 1) return f[x][y] = grid[x][y]; // 不能继续走，f[x][y]的值就是当前点的值
            if (f[x][y] != -1) return f[x][y];


            int res = INF;
            if (x &gt;= 0 &amp;&amp; x &lt; n &amp;&amp; y &gt;= 0 &amp;&amp; y &lt;= x) {

                res = min(dfs(x &#43; 1, y), dfs(x &#43; 1, y &#43; 1)) &#43; grid[x][y];
            }

            return f[x][y] = res;
        };
        return dfs(0, 0);
    }
};
```



### 221.最大正方形

&gt; [🔗](https://leetcode.cn/problems/maximal-square/description/?envType=study-plan-v2&amp;envId=top-interview-150)
&gt; 在一个由 &#39;0&#39; 和 &#39;1&#39; 组成的二维矩阵内，找到只包含 &#39;1&#39; 的最大正方形，并返回其面积。

```cpp
class Solution {
    static const int N = 310;
    int f[N][N];
public:
    int maximalSquare(vector&lt;vector&lt;char&gt;&gt;&amp; matrix) {
        int m = matrix.size(), n = matrix[0].size();

        memset(f, 0, sizeof f);

        int a = 0;
        for (int i = 0; i &lt; m; i &#43;&#43; ) {
            for (int j = 0; j &lt; n; j &#43;&#43; ) {
                if (matrix[i][j] == &#39;0&#39;) continue;
                if (i == 0 || j == 0) f[i][j] = 1;  // 边长最大只能为1
                else {
                    // f[i][j]表示以 (i,j) 为右下角的，最大正方形的边长
                    f[i][j] = min(min(f[i][j - 1], f[i - 1][j]), f[i - 1][j - 1]) &#43; 1;
                }
                a = max(a, f[i][j]);
            }
        }
        return a * a;
    }
};
```


---

> 作者: yitao  
> URL: https://yitaonote.com/2025/be012dd/  

