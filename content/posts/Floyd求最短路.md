---
title: Floyd求最短路
subtitle:
date: 2025-08-16T17:33:47+08:00
slug: 6278ffa
draft: false
author:
  name: "yitao"
  link:
  email:
  avatar:
description:
keywords:
license:
comment: false
weight: 0
tags:
  - draft
categories: [算法]
hiddenFromHomePage: false
hiddenFromSearch: false
hiddenFromRelated: false
hiddenFromFeed: false
summary:
resources:
  - name: featured-image
    src: featured-image.jpg
  - name: featured-image-preview
    src: featured-image-preview.jpg
toc: true
math: true
lightgallery: false
password:
message:
repost:
  enable: true
  url:

# See details front matter: https://fixit.lruihao.cn/documentation/content-management/introduction/#front-matter
---

<!--more-->

#  Floyd 算法

### 算法框架

![IMG_20210628_164616_edit_183458334133463.jpg](https://cdn.acwing.com/media/article/image/2021/06/28/94631_603f3714d7-IMG_20210628_164616_edit_183458334133463.jpg)

### 适用场景

多源汇最短路，可以有负权边，但是不能有负权回路。

### 算法原理概述

基于动态规划。

闫氏DP分析法：

状态表示

> $d(k,i,j)$：从点 $i$ 出发，只经过 $1\sim k$ 这些中间点到达 $j$ 的最短距离。

状态计算

> $d(k,i,j) = d(k-1,i,k) + d(k-1,k,j) \Rightarrow d(i,j) = d(i,k) + d(k,j)$

### 时间复杂度分析

三重循环，故复杂度为 $O(n^{3})$。

### 完整代码
注：一定要先循环```k```，```i```和```j```的顺序可以任意颠倒。
```cpp
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;
const int N = 210, INF = 1e9;

int n, m, Q;
int d[N][N];// 邻接矩阵，也是floyd算法处理的距离

void floyd()
{
    for (int k = 1; k <= n; k ++ )
        for (int i = 1; i <= n; i ++ )
            for (int j = 1; j <= n; j ++ )
                d[i][j] = min(d[i][j], d[i][k] + d[k][j]);
}

int main()
{
    scanf("%d%d%d", &n, &m, &Q);
    for (int i = 1; i <= n; i ++ )
        for (int j = 1; j <= n; j ++ )
            if (i == j) d[i][j] = 0; // 去掉自环
            else d[i][j] = INF;
    while (m -- )
    {
        int a, b, w;
        scanf("%d%d%d", &a, &b, &w);
        d[a][b] = min(d[a][b], w); // 若有重边，则只保留最短的边
    }
    floyd();
    while (Q -- )
    {
        int a, b;
        scanf("%d%d", &a, &b);
        // 和bellman ford类似，即使终点与起点不连通，也还是可能会被负权邻边更新，所以适当放宽条件
        if (d[a][b] > INF / 2) puts("impossible");
        else printf("%d\n", d[a][b]);
    }
    return 0;
}
```
