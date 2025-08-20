---
title: DeepSeek相关优化技术
subtitle:
date: 2025-08-14T00:15:18+08:00
slug: d01413e
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
categories: [推理优化]
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


## MLA

> [参考图解视频](https://www.bilibili.com/video/BV1jDLBzoE1e/?spm_id_from=333.337.search-card.all.click&vd_source=d436f8a78656c9132eae4a84f939d219)

### 朴素实现
![image](https://cdn.ipfsscan.io/weibo/large/005wRZF3ly1i4kr9o3jfcj31c40u0wjj.jpg)


### 矩阵吸收
![image](https://cdn.ipfsscan.io/weibo/large/005wRZF3ly1i4kr9jozmfj320o0qhn2j.jpg)

> [!TIP] 上面图解版本不包括**旋转位置编码**，但我认为用来理解MLA基本原理还是非常好的



### 朴素实现（包含rope）

> 参考文献：[https://zhuanlan.zhihu.com/p/1901704483446187870](https://zhuanlan.zhihu.com/p/1901704483446187870)

![image](https://cdn.ipfsscan.io/weibo/large/005wRZF3ly1i4kr76q5blj31400iywg9.jpg)

> [!TIP] $t$ 代表 t-th token


首先输入 $h_t$ 经过 $W_{DQ}$ 得到 $c_t^Q$ 再分别经过 $W^{QR}$, $W^{UQ}$ 得到用于 rope计算的 $q_t^R$ 和 nope 的 $q_t^C$：

$$
\begin{aligned}
h_{t}W_{DQ} &= c_{t}^{Q} \\
c_{t}^{Q}W^{QR} &= q_{t}^{R} \\
c_{t}^{Q}W^{UQ} &= q_{t}^{C} \\
\end{aligned}
$$

其次来看 KV 部分，输入 $h_t$ 经过 $W_{DKV}$ 得到 $c_t^{KV}$，得到联合低秩压缩的 KV，$c_t^{KV}$ 与历史的 c cache拼接后，得到完整的 $c^{KV}$，
再分别经过 $W_{UK}$，$W_{UV}$ 进行升维，得到 $k^C$，$v^C$：


$$
\begin{aligned}
XW_{DKV} &= c_t^{KV} \\
c^{KV}W_{UK} &= k^C\\
c^{KV}W_{UV} &= v^C \\
\end{aligned}
$$

再回到输入部分，现在来计算 qk 的rope部分，输入 $h_t$ 经过 $W_{KR}$ 得到 $k_t^R$，同样与历史的 $k_pe$ cache拼接后，得到完整的 $k^R$，然后与 $q_t^R$ 进行 rope计算：

$$
\begin{aligned}
h_tW^{KR} &= k_t^R \\
q_t^R(k^R)^T &= \text{attn}^R
\end{aligned}
$$


得到 $\text{attn}^R$ 后，我们再回到之前的计算过程，我们之前计算得到了 $q_t^C$ 和完整的 $k^C$，因此可以计算出 attn 的nope部分：

$$
\begin{aligned}
q_t^C(k^C)^T &= \text{attn}^C
\end{aligned}
$$

故：

$$
Softmax(\dfrac{QK^T}{\sqrt{d}}) = Softmax(\dfrac{\text{attn}^C + \text{attn}^R}{\sqrt{d}})
$$

然后再与v进行相乘得到最终的注意力输出：

$$
\begin{aligned}
PV &= Softmax(\dfrac{\text{attn}^C + \text{attn}^R}{\sqrt{d}})v^C \\
O &=  Softmax(\dfrac{\text{attn}^C + \text{attn}^R}{\sqrt{d}})v^C W_o
\end{aligned}
$$


### 矩阵吸收（包含rope）


![image](https://cdn.ipfsscan.io/weibo/large/005wRZF3ly1i4krhbv6mcj31400iymyx.jpg)

主要是将 $c_{t}^{Q}W^{UQ} = q_{t}^{C}$ 和 $c^{KV}W_{UK} = k^C$ 这两步中的 $W^{UQ}$ 和 $W^{UK}$ 吸收到了 $W^{UQK}$，使得
$\text{attn}^C$ 的计算可以直接由 $q_t^C(c^{KV})^T = \text{attn}^C$ 得到，而不需要先对低秩联合压缩的 $c^{KV}$ 先做升维得到 $k^C$，即解压操作。

另外是将 $c^{KV}W^{UV} = v^C$ 和 $O =  Softmax(\dfrac{\text{attn}^C + \text{attn}^R}{\sqrt{d}})v^C W_o$ 这两步中的 $W^{UV}$ 和 $W_o$ 吸收到了 $W^{UVO}$，使得注意力输出的 $O$ 的计算可以直接由 $Softmax(\dfrac{\text{attn}^C + \text{attn}^R}{\sqrt{d}})W^{UVO}$ 得到，而不需要先对低秩联合压缩的 $c^{KV}$ 先做升维得到 $v^C$，即解压操作。


**总结：**
这样做可以直接在潜在空间（即压缩后的维度）中进行注意力计算，而不需要先解压 KV-Cache。并且能够减少计算量：通过将两个矩阵相乘的操作合并成一个，减少了所需的计算步骤，从而提高了推理速度。简单来说，矩阵吸收就像是把一个两步走的任务（先解压，再计算）变成了一步走的任务（直接用“吸收”后的矩阵进行计算），从而实现了内存和计算效率的双重优化。


## MoE

## MTP


## DeepSeek开源周

### FlashMLA

### DeepEP

### DeepGEMM

### DualPipe & EPLB

### 3FS
