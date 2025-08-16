---
title: Dijkstra求最短路I
subtitle:
date: 2025-08-16T17:33:23+08:00
slug: c2f4d7e
draft: false
author:
  name: "yitao"
  link:
  email:
  avatar:
description:
keywords:
license:
comment: true
weight: 0
tags:
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
  enable: false
  url:

# See details front matter: https://fixit.lruihao.cn/documentation/content-management/introduction/#front-matter
---

<!--more-->

# 朴素 Dijkstra 算法


### 算法框架


![IMG_20210627_211108_edit_157472230052533.jpg](https://cdn.acwing.com/media/article/image/2021/06/27/94631_35d5affad7-IMG_20210627_211108_edit_157472230052533.jpg)


### 算法步骤

适用于稠密图，故用邻接矩阵存储图

```st[]```存储所有当前已经更新过其他点的点。

1、遍历 $n$ 次，找到当前不在```st```集合中的，距离起点最近的点，赋给```t```

2、用```t```去更新其他所有能够直连的点的距离：```dist[x] > dist[t] + w```若真，就更新```dist[x]```。

### 证明

基于贪心，无需掌握，过程略。


> [!TIP]
> 每一轮迭代，在还没有更新过其他点的所有点中，找到距离起点最近的点```t```，然后用```t```去更新其他所有点到起点的距离。
> ```cpp
> int t = -1; // 为了方便找这样一个距离起点最近的点，先让 t = -1，以便加入起点循环查找
> for (int j = 1; j <= n; j ++ )
>    if (!st[j] && (t == -1 || dist[t] > dist[j]))
>        t = j;
>```

### 完整代码
```cpp
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 510;
int n, m;
int g[N][N];
int dist[N];        // 当前点到初始点的最短距离
bool st[N];         // st数组更确切的含义是某个点是否已经更新过其他点，而不是它的最短距离是否已经确定

int dijkstra()
{
    memset(dist, 0x3f, sizeof dist);
    dist[1] = 0;

    for (int i = 0; i < n; i ++ )
    {
        int t = -1;
        for (int j = 1; j <= n; j ++ )
            if (!st[j] && (t == -1 || dist[t] > dist[j]))
                t = j;

        st[t] = true;
        // 实际上，这里也只用t去更新了其他与t相邻的点的距离，只更新了相邻节点，因为不是相邻的话
        // g[t][j] 为 INF，相当于没有更新。
        for (int j = 1; j <= n; j ++ )
            dist[j] = min(dist[j], dist[t] + g[t][j]);  // 用t去更新其他点的距离
    }
    if (dist[n] == 0x3f3f3f3f)  return -1;
    return dist[n];
}

int main()
{
    cin >> n >> m;
    memset(g, 0x3f, sizeof g);

    while (m -- )
    {
        int a, b, c;
        scanf("%d%d%d", &a, &b, &c);
        g[a][b] = min(g[a][b], c);      // 处理自环和重边
    }

    int t = dijkstra();
    cout << t << endl;
    return 0;
}
```

#### Python3
```python3
N, M = 510, int(1e5 + 10)
dist = [0 for _ in range(N)]
st = [False for _ in range(N)]
g = [[0] * N for _ in range(N)]

def dijkstra():
    dist = [int(1e9) for _ in range(N)]
    dist[1] = 0

    for i in range(n):
        t = -1
        for j in range(1, n + 1):
            if not st[j] and (t == -1 or dist[t] > dist[j]):
                t = j
        st[t] = True
        for j in range(1, n + 1):
            dist[j] = min(dist[j], dist[t] + g[t][j])
    if dist[n] == int(1e9): return -1
    return dist[n]

if __name__ == '__main__':
    n, m = map(int, input().split())
    g = [[int(1e9)] * N for _ in range(N)]

    while m:
        m -= 1
        a, b, c = map(int, input().split())
        g[a][b] = min(g[a][b], c)

    print(dijkstra())
```
