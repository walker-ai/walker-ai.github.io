---
title: Dijkstra求最短路II
subtitle:
date: 2025-08-16T17:33:25+08:00
slug: 652842d
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

# 堆优化 Dijkstra

### 算法框架 & 步骤

![IMG_20210627_213210_edit_157871044713930.jpg](https://cdn.acwing.com/media/article/image/2021/06/27/94631_e0a41162d7-IMG_20210627_213210_edit_157871044713930.jpg)

> [!TIP] 堆优化版 Dijkstra 适用于稀疏图，故用邻接表存储图。

### 优化思路

1、朴素 Dijkstra 中，最慢的一步是每轮迭代中，寻找所有未更新过其他点的点中，距离起点最近的点，我们可以用小根堆来维护这个所有未更新过其他点的点构成的集合，这样每次查找最小值，就是 $O(1)$ 的时间复杂度，一共迭代 $n$ 次，所以是 $O(n)$。

2、这样用这个最小点去更新其他所有点的距离的时间复杂度，就由 $O(m)$ 变为了 $O(m\log n)$，因为在堆中修改一个数的时间复杂度是 $O(\log n)$ 的。故总时间复杂度是 $O(m\log n)$。

3、另外由于 STL 内的优先队列写法，堆中不支持修改任意元素，修改体现在往堆中添加一个新元素，这样会造成堆中元素冗余，堆中元素可能是 $m$ 个，这样时间复杂度就会退化至 $O(m\log m)$，但由于一般 $m \leq n^{2}$，故 $\log m \leq \log(n^{2}) = \log m \leq 2\log n$，时间复杂度接近，所以一般不用手写堆，直接用 STL 内的优先队列即可。

另外，Dijkstra 可以算是 BFS 的升级版，就是说如果求最短路径，当图从无权值变成有权值时，BFS 不再适用了，于是我们用 Dijkstra 方法。换句话说，对于无权值图，Dijkstra 方法跟 BFS 是一致的。你可以画个无权图，用 Dijkstra 走一遍，发现其实这就是 BFS。

> [!TIP] 此处邻接表中不需要特殊考虑重边，因为算法保证了一定能够选择最短的边。

### 完整代码

#### C++
```cpp
#include <iostream>
#include <cstring>
#include <algorithm>
#include <queue>

using namespace std;
typedef pair<int, int> PII;

const int N = 200010; // 可能存在重边，需要开大一点

int n, m;
int e[N], h[N], w[N], ne[N], idx; // w[i]表示当前这个结点所连的下一条边权
int dist[N];   // 当前点到初始点的最短距离
bool st[N]; // 标记当前点的最短距离是否已经确定

void add(int a, int b, int c)
{
    e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx++;
}

int dijkstra()
{
    memset(dist, 0x3f, sizeof dist);
    dist[1] = 0;

    priority_queue<PII, vector<PII>, greater<PII>>  heap; // 小根堆的写法，不用过多深究，记住即可
    heap.push({0, 1});

    while (heap.size())
    {
        auto t = heap.top();
        heap.pop();

        int ver = t.second, distance = t.first;
        if (st[ver])    continue;   // st[ver]为真表示当前这个点是堆中备份点，已经被处理过
        st[ver] = true;

        // 遍历t的所有出边
        for (int i = h[ver]; i != -1; i = ne[i])
        {
            int j = e[i];
            if (dist[j] > distance + w[i])      // 依旧是用t更新其他所有直连点距离
            {
                dist[j] = distance + w[i];
                heap.push({dist[j], j});
            }
        }
    }
    if (dist[n] == 0x3f3f3f3f)  return -1;
    return dist[n];
}

int main()
{
    scanf("%d%d", &n, &m);
    memset(h, -1, sizeof h);

    while (m -- )
    {
        int a, b, c;
        scanf("%d%d%d", &a, &b, &c);
        add(a, b, c);       // 堆优化版Dijkstra不用特殊处理重边和自环，因为算法本身会选择最短边
    }

    int t = dijkstra();

    printf("%d\n", t);
    return 0;
}
```

#### Python3
```python3
def add(a, b, c):
    global idx
    e[idx], w[idx], ne[idx] = b, c, h[a]
    h[a] = idx
    idx += 1

def dijkstra():
    from queue import PriorityQueue
    q = PriorityQueue()
    q.put((0, 1))
    dist = [float('inf')] * (n + 10)
    dist[1] = 0

    while q.qsize() > 0:
        t = q.get()  # .get() 相当于 .front(), .pop()
        ver, distance = t[1], t[0]
        if st[ver]: continue
        st[ver] = True

        i = h[ver]
        while i != -1:
            j = e[i]
            if dist[j] > distance + w[i]:
                dist[j] = distance + w[i]
                q.put((dist[j], j))
            i = ne[i]
    if dist[n] == float('inf'): return -1
    return dist[n]

if __name__ == '__main__':
    N, M = int(2e5 + 10), int(2e5 + 10)

    n, m = map(int, input().split())
    h, w, e, ne = [-1] * N, [0] * M, [0] * M, [0] * M
    idx = 0
    st = [False] * (n + 10)

    for _ in range(m):
        a, b, c = map(int, input().split())
        add(a, b, c)

    print("%d" % dijkstra())
```
