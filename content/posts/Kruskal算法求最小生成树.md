---
title: Kruskal算法求最小生成树
subtitle:
date: 2025-08-16T17:33:59+08:00
slug: acc2381
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

> Kruskal 算法原理及证明见[提高课](https://www.acwing.com/activity/content/code/content/1745411/)。

## Kruskal算法求最小生成树 $O(m\log m)$
![IMG_20210629_141909_edit_223503600585686.jpg](https://cdn.acwing.com/media/article/image/2021/06/29/94631_ff56a31cd8-IMG_20210629_141909_edit_223503600585686.jpg)

**这里重载运算符的方式也可以通过写cmp()来作为sort的第三个参数**

```cpp
bool cmp(Edge a, Edge b)
{
    return a.w < b.w;
}
```


### 完整代码
```cpp
#include <iostream>
#include <algorithm>
#include <cstring>
using namespace std;

const int N = 100010, M = 200010, INF = 0x3f3f3f3f;
int p[N];
int n, m;

struct Edge
{
    int a, b, w;
    bool operator< (const Edge &W)const
    {
        return w < W.w;
    }
}edges[M];


int find(int x)
{
    if (p[x] != x)  p[x] = find(p[x]);
    return p[x];
}

int kruskal()
{
    sort(edges, edges + m);

    for (int i = 1; i <= n; i ++ )  p[i] = i;

    int res = 0, cnt = 0;
    for (int i = 0; i < m; i ++ )
    {
        int a = edges[i].a, b = edges[i].b, w = edges[i].w;

        a = find(a), b = find(b);
        if (a != b)
        {
            p[a] = b;
            res += w;
            cnt ++;
        }
    }
    if (cnt < n - 1)    return INF;
    return res;
}

int main()
{
    cin >> n >> m;

    for (int i = 0; i < m; i ++ )
    {
        int a, b, w;
        scanf("%d%d%d", &a, &b, &w);
        edges[i] = {a, b, w};
    }

    int t = kruskal();

    if (t == INF)   puts("impossible");
    else cout << t << endl;
    return 0;
}
```
