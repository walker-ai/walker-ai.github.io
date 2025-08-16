---
title: Prim算法求最小生成树
subtitle:
date: 2025-08-16T17:33:53+08:00
slug: 1f7d467
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

> Prim 算法原理及证明见[提高课](https://www.acwing.com/activity/content/code/content/1745411/)。

Prim算法求最小生成树 $O(n^{2})$，跟Dijkstra很像

![IMG_20210628_204051_edit_188738021723283.jpg](https://cdn.acwing.com/media/article/image/2021/06/28/94631_240b2fb5d8-IMG_20210628_204051_edit_188738021723283.jpg)

### 完整代码
```
#include <iostream>
#include <algorithm>
#include <cstring>

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
    for (int i = 0; i < n; i ++ ) // 每次找到集合外的，距集合距离最小的点
    {   
        int t = -1;     // t = -1 表示当前还没还有找到任何一个点
        for (int j = 1; j <= n; j ++ )
            if (!st[j] && (t == -1 || dist[t] > dist[j]))
                t = j;
        // 当前图是不连通的，不存在最小生成树
        if (i && dist[t] == INF)    return INF;
        if (i) res += dist[t];     // dist[t] 表示一条树边，加到生成树中去

        for (int j = 1; j <= n; j ++ )  dist[j] = min(dist[j], g[t][j]);
        st[t] = true;
    }
    return res;
}

int main()
{
    scanf("%d%d", &n, &m);
    memset(g, 0x3f, sizeof g);

    while (m -- )
    {
        int a, b, c;
        scanf("%d%d%d", &a, &b, &c);
        g[a][b] = g[b][a] = min(g[a][b], c);    // 无向图，处理重边
    }
    int t = prim();
    if (t == INF) puts("impossible");
    else printf("%d\n", t);
    return 0;
}
```
