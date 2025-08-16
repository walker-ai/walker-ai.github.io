---
title: N皇后问题
subtitle:
date: 2025-08-16T17:32:52+08:00
slug: f5a0f88
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

### 第一种写法，按行枚举
```cpp
#include <iostream>
using namespace std;


const int N = 20;

char g[N][N];
bool col[N], dg[N], udg[N];
int n;




void dfs(int u)
{
    if(u == n)
    {
        for(int i = 0; i < n; i ++ )  puts(g[i]);
        puts("");
        return;
    }

    for(int i = 0; i < n; i ++ )
    {
        if(!col[i] && !dg[u + i] && !udg[n - u + i])
        {
            g[u][i] = 'Q';
            col[i] = dg[u + i] = udg[n - u + i] = true;
            dfs(u + 1);
            col[i] = dg[u + i] = udg[n - u + i] = false;
            g[u][i]= '.';
        }
    }
}

int main()
{
    cin >> n;
    for(int i = 0; i < n; i ++ )
        for(int j = 0; j < n; j ++ )
            g[i][j] = '.';

    dfs(0);
    return 0;
}
```

### 第二种写法，一格一格枚举
```cpp
#include <iostream>
using namespace std;

const int N = 20;
bool row[N], col[N], dg[N], udg[N];
char g[N][N];
int n;


void dfs(int x, int y, int s)
{
    if(y == n)  y = 0, x ++;

    if(x == n)
    {
        if(s == n)
        {
            for(int i = 0; i < n; i ++) puts(g[i]);
            puts("");
        }
        return;
    }

    dfs(x, y + 1, s);

    if(!row[x] && !col[y] && !dg[x + y] && !udg[x - y + n])
    {
        g[x][y] = 'Q';
        row[x] = col[y] = dg[x + y] = udg[x - y + n] = true;
        dfs(x, y + 1, s + 1);
        row[x] = col[y] = dg[x + y] = udg[x - y + n] = false;
        g[x][y] = '.';
    }
}

int main()
{
    cin >> n;

    for(int i = 0; i < n; i ++ )
        for(int j = 0; j < n; j ++ )
            g[i][j] = '.';

    dfs(0, 0, 0);
    return 0;
}
```
