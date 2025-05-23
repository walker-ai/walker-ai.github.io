---
title: 构造 RandX()
date: 2025-03-22T23:15:08+08:00
tags: [构造]
categories: [算法]
slug: b44918a
author:
  name: "yitao"


# See details front matter: https://fixit.lruihao.cn/documentation/content-management/introduction/#front-matter
---

由 `randA()` 实现 `randB()`：[万能构造法](https://leetcode.cn/problems/implement-rand10-using-rand7/solutions/979495/mo-neng-gou-zao-fa-du-li-sui-ji-shi-jian-9xpz/)

<!--more-->

> `randA()` 构造 `randB()` 时，需要找一个最大质因子不超过 A 的数 n (n>=B），然后对 n 分解质因子就能找到每个采样需要取多少种结果。实际到具体数字时，可以把部分质因子合并成不超过 A 的数，从而减少采样次数。

举个具体例子，如何用 `rand7()` 来构造 `rand10()`

## 确定采样参数

- 步骤 1：选择一个合适的数 n

我们选择 `n = 30`，因为它大于或等于 10，并且它的最大质因子是 5，不超过 7。

- 步骤 2：对 n 进行质因子分解

将 30 分解为质因子的乘积形式：

$$ 30 = 2^1 \times 3^1 \times 5^1 $$

- 步骤 3：确定采样次数和结果数量

根据质因子分解的结果，我们需要进行 3 次采样，每次采样的结果数量分别为 2、3 和 5。（分别对应三个质因子）

## 进行采样

我们将进行 3 次采样，每次采样使用 `rand7()` 函数来生成一个介于 1 和 7 之间的随机数。

1.  第一次采样
    - 使用 `rand7()` 来生成一个随机数，如果结果在 [1, 2] 范围内，则将其作为第一次采样的结果。否则，重新采样。
    ```c++
    int first;
    while (true) {
        first = rand7();
        if (first <= 2) break;
    }
    ```
2.  第二次采样
    - 使用 `rand7()` 来生成一个随机数，如果结果在 [1, 3] 范围内，则将其作为第一次采样的结果。否则，重新采样。
    ```c++
    int second;
    while (true) {
        second = rand7();
        if (second <= 3) break;
    }
    ```
3.  第三次采样
    - 使用 `rand7()` 来生成一个随机数，如果结果在 [1, 5] 范围内，则将其作为第一次采样的结果。否则，重新采样。
    ```c++
    int third;
    while (true) {
        third = rand7();
        if (third <= 5) break;
    }
    ```

## 组合结果

将每次采样的结果组合起来，得到一个长度为 30 的序列。具体来说：

- first 的取值范围是 [1, 2]，总共有 2 种可能。
- second 的取值范围是 [1, 3]，总共有 3 种可能。
- third 的取值范围是 [1, 5]，总共有 5 种可能。

我们通过三次采样得到三个值：first、second 和 third。我们的目标是将这三个值组合成一个唯一的索引，这个索引应该对应于一个长度为 30 的序列中的一个位置。

这三个值的所有可能组合数是：$ 2\times 3\times 5 = 30 $， 这正好等于我们预定义的序列长度。

## 映射

为了将这三个值组合成一个唯一的索引，我们需要为每个值分配一个权重，使得它们的组合能够覆盖从 1 到 30 的所有整数。

公式为：

$$ index = (first - 1)\times 3 \times 5 + (second - 1)\times 5 + (third - 1) + 1 $$

代表：

$$[0, 1]\times 3\times 5 \rightarrow [0, 15]$$
$$[0, 2]\times 5 \rightarrow [0, 10]$$
$$[0, 4] + 1 \rightarrow [1, 5]$$

## 完整代码

```c++
int rand10() {
    int first, second, third;
    while (true) {
        first = rand7(); // 第一次采样
        if (first <= 2) break; // 如果结果在 [1, 2] 范围内，则退出循环
    }
    while (true) {
        second = rand7(); // 第二次采样
        if (second <= 3) break; // 如果结果在 [1, 3] 范围内，则退出循环
    }
    while (true) {
        third = rand7(); // 第三次采样
        if (third <= 5) break; // 如果结果在 [1, 5] 范围内，则退出循环
    }
    // 将结果组合并映射到 [1, 10] 范围内
    int index = (first - 1) * 3 * 5 + (second - 1) * 5 + (third - 1) + 1;
    if (index <= 10) return index;
    else return rand10(); // 如果结果超出范围，则重新采样
}
```
