---
title: 量化
subtitle:
date: 2025-08-07T22:17:38+08:00
slug: a793bc7
draft: false
author:
  name: "yitao"

---

<!--more-->

## 基本概念

> 参考链接：[https://zhuanlan.zhihu.com/p/1912965417447686461](https://zhuanlan.zhihu.com/p/1912965417447686461)

- 缩放系数：$s=\dfrac{\text{max}(|\text{min}_ {val}|, |\text{max}_{val}|)}{2^{N-1}}$

- 零点：$z=round(q_{min}-\dfrac{r_{min}}{s})=round(-2^{N-1}-\dfrac{r_{min}}{s})$

```python
quantized:
x_float = (x_int8 - zero_point) * scale

dequantized:
x_int8 = round(x_float/scale) + zero_point
```

这里有个前置知识就是，在神经网络中一般权重的形状
都是 $W\in R^{O\times I}$，其中 $O$ 表示输出通道(output channel), $I$ 表示输入通道(input channel)，对于矩阵乘法，即 $y = Wx$, $x\in R^{I}, y\in R^{O}$

## 量化粒度 - 三种形式

- per-tensor

这是最粗粒度的量化方式。对整个 tensor 计算出一个统一的缩放因子（scale factor），然后将张量中的所有元素都使用这个缩放因子量化到低精度格式，例如 INT8。例如，一个 INT8 的 per-tensor 动态量化器会找到整个张量的最大绝对值，以此计算出一个缩放因子，然后将所有元素缩放到 INT8 的表示范围内 [-127, +127] 并进行取整1 。

- per-token

这种量化方式的粒度比 per-tensor 更细。对于张量的每一个 token（通常指的是矩阵的每一行），都单独计算并应用一个缩放因子。

- per-block

这种量化方式的粒度比 per-token 更细。它将张量在 token 维度上划分为若干个块（block）。对于每一个块内的所有 token（行），计算出一个统一的缩放因子并应用，。

- per-channel

这种量化方式和 per-token 类似，但是量化维度不一样，对于张量的每一个通道（通常指 hidden dim 维度的一列），都单独计算并应用一个缩放因子。

## 离群值 (Outliers)

离群值的存在是模型量化时精度下降的主要原因。这是因为量化将模型中的连续数值映射到有限的离散数值范围（例如 INT4 的范围是 [-7, +7]）。如果数据中存在少数数值远超其他数据的离群值，为了表示这些极端的数值，量化的步长（resolution）就需要增大。

这样做的直接后果是：

大多数正常的、幅度较小的数值在量化后会变得非常接近甚至等于零。例如，如果一个数值比组内的最大值小很多倍，它可能会被量化为零，导致大量信息的丢失。

有限的量化比特无法精确表示这些大部分的正常数值，从而降低了整体的量化精度。

为了解决这个问题，需要采用平滑（smoothing）技术来减小激活或权重中离群值的影响，使得数值的幅度分布更加均匀。量化方法通过观察任务的 outliner 特点，来针对性地设计量化方法。

比较有代表性的是SmoothQuant ，它观察到在LLM的推理过程中，激活值（activations）中往往比权重值（weights）更容易出现显著的离群值。SmoothQuant 通过一种数学上等价的Per-channel缩放（channel-wise scaling）操作，将模型量化的难度从激活转移到权重。具体来说，它降低了激活中异常大的数值，使得激活值更容易被量化到低比特（例如 INT8），从而在保持模型精度的前提下，实现更高效的量化推理。

## 对称 / 非对称量化区别

| 项目             | 对称量化               | 非对称量化            |
| -------------- | ------------------ | ---------------- |
| 是否有 zero point | ❌ 固定为 0            | ✅ 存在 zero\_point |
| 适用范围           | 权重量化（特别是 centered） | 激活量化（值范围变化大）     |
| 运算简化           | ✅ 快速，适合矩阵乘法        | ❌ 多了减法，加快复杂度     |
| 表现力            | ❌ 表达负偏移有限          | ✅ 支持偏移，精度更高      |
| 效果稳定性          | ✅ 稳定               | ✅ 更灵活，适应动态变化     |

## 量化对象 - Linear (Dense) / Attention

> 参考链接：[https://zhuanlan.zhihu.com/p/1895945361824122028](https://zhuanlan.zhihu.com/p/1895945361824122028)

### Linear 量化

指对模型的 Linear 层进行量化，Linear 层主要分布于：

1. Attention 中的 Q/K/V/O projection
2. MLP（FFN) 中的 gate/up/down
3. Embedding (`[vocab_size, hidden_size]`) 和 LM Head (`[hidden_size, vocab_size]`)
4. MoE 中的 expert

### Attention量化，以 SageAttention 为例

矩阵乘法中，对于每个矩阵你只能沿着公共维度进行量化（下图右边）。根据这个简单的原则，Attention 中四个矩阵可以量化的组合如下。注意能做 per-token，就能做 per-block 量化。其中 P 代表 $softmax(QK^T/\sqrt{d})$

![image](https://cdn.ipfsscan.io/weibo/large/005wRZF3ly1i45oak307uj312m0l8dgo.jpg)

这是因为在进行矩阵乘法 $QK^{T}$ 后，得到的结果矩阵的维度是 N × N（Q 和 K 的维度都是 N × d）。如果我们对 K 进行了per-channel 量化（下图左边，总共 d 个channel，每个 channel 包含 N 个元素），每个通道都有一个独立的scale factor，总共是 d 个 scale factor。在反量化（dequantization）时，我们需要将量化后的结果乘以对应的scale factor，而QK^T 的结果矩阵的维度是 NxN，根本没有 d 的通道维度不直接对应，因此无法使用 K 的通道维度的缩放因子进行正确的反量化。

| | Q | K | P | V |
| :-: | :-: | :-: | :-: | :-: |
| per-channel | ❌ | ❌ | ❌ | ✅ |
| per-token | ✅ | ✅ | ✅ | ❌ |
| per-block | ✅ | ✅ | ✅ | ❌ |

## 量化类型 - weight-only / weight-activation / KV cache


![image](https://cdn.ipfsscan.io/weibo/large/005wRZF3ly1i4bjajqt85j30yt0kvad4.jpg)

INT 量化范式：

1. FP32作为基准，提供了最大的数值范围和零精度损失，但存储开销最大。
2. 如果用户不太关心效率，那么INT16格式是最佳选择。INT16格式是最精确的，如果是转换FP32，INT16甚至比FP16更精确。
3. 对于对实时性要求高的服务，建议采用INT8量化方案，可以在保持较高精度的同时获得显著的性能提升。如果你的网络中某些层需要更高的精度，可以使用W8A16来解决这个问题。
4. 在资源受限但对精度要求相对较低的场景，则可以采用INT4方案以获得最大的资源效益。

### weight-only

#### GPTQ

> $\text{Frobenius}$ 范数，简称 F-范数，$||\cdot||^{2}_{2}$ 表示对矩阵的 Frobenius 范数的平方（即所有元素平方和）
Frobenius 范数可以用来衡量矩阵整体的大小，比如在误差分析中，可以用来评估两个矩阵之间的差异程度

GPTQ是一种训练后权重量化方法，使用基于二阶信息的逐层量化，成功将每个权重量化至 3-4 位，几乎没有精度损失。GPTQ 对某个 block 内的所有参数逐个量化，每个参数量化后，需要适当调整这个 block 内其他未量化的参数，以弥补量化造成的精度损失。 GPTQ 量化需要准备校准数据集。

OBS/OBQ/GPTQ等一系列工作的核心就是：

不直接最小化权重误差，而是：$\min_{q(w)}||(q(w)-w)X||^2_{2}$

给定权重矩阵W，有以下步骤：
1. 收集激活样本（校准数据集）
2. 计算输入协方差矩阵 $H = X^{T}X$
3. 逐列量化 W 的每一列
  - 每列权重找最优 int4 表示
  - 误差反馈，用 Hessian 更新下列的残差
4. 保存量化结果，然后在推理阶段使用 int4 ✖️ float16 / int8的高效矩阵乘法

![image](https://cdn.ipfsscan.io/weibo/large/005wRZF3ly1i4cllfsriej30zk0mrgoa.jpg)

![image](https://cdn.ipfsscan.io/weibo/large/005wRZF3ly1i4cllm5vvmj30zk0irac5.jpg)

![image](https://cdn.ipfsscan.io/weibo/large/005wRZF3ly1i4cllp0h02j30zk0ckwfn.jpg)

#### AWQ

> 参考：[https://www.zhihu.com/search?type=content&q=AWQ%20%E9%87%8F%E5%8C%96](https://www.zhihu.com/search?type=content&q=AWQ%20%E9%87%8F%E5%8C%96)

AWQ观察到权重通道对性能的重要性各不相同，通过保留1%的显著权重可以大大减少量化误差。基于此观察，AWQ采用了激活感知权重量化来量化LLM，具体会专注于激活值较大的权重通道，并通过每通道缩放实现最佳量化效果。

**误区：AWQ量化=W4A16量化**

AWQ是一种对模型权重进行低比特量化的方法，使用该方法可以将模型权重(Weight)量化为4bit，并在计算激活值(Activation)时反量化为FP16，即W4A16。也可以基于AWQ方法将权重量化为3bit/8bit，并在计算时是使用4bit/8bit/16bit，由此衍生出W4A4、W4A8等一系列方法。
作者在原文中指出，W4A16可以在精度损失较小的情况下，大幅降低内存占用，且提升模型推理速度，是最常用的方法，因此AWQ和W4A16同镜率较高。

**显著权重（权重并不同等重要，仅有部分显著权重对结果影响较大）**

权重矩阵中显著权重位于哪个通道，找到这个通道，将这个通道内的部分保留原来的精度(fp16)，然后其他部分量化为低 bit。

步骤：在计算时，首先将激活值对每一列求绝对值的平均值，然后把平均值较大的一列对应的通道视作显著通道，保留FP16精度。对其他通道进行低比特量化，如下图：

![image](https://cdn.ipfsscan.io/weibo/large/005wRZF3ly1i4clh2pwwnj30wu0b6whh.jpg)

这里详细解释一下“将激活值对每一列求绝对值的平均值，然后把平均值较大的一列对应的通道视作显著通道”

对于 $y=xW^{T}$，其中 $y\in R^{B\times O}, W\in R^{O\times I}, x\in R^{B\times I}$

（这里 chatgpt 的解释是，在数学上 $y = Wx$，实现中记为 $y=xW^{T}$）

这张图中的 $X\in R^{I\times B}, W\in R^{O\times I}$，因此 $y=Wx \in R^{O\times B}$

所以不管数学上的表达还是实际上的代码实现，y 的输出通道始终由权重的输出通道决定，即 y 的输出通道是否是显著值，就看对应的激活值那一列，所以这里是求的每一列的绝对值的平均值，把平均值较大的一列视为显著通道。

但是如果这样做，权重矩阵中有的元素需要用 FP16，而其他元素需要用 INT8，不好写 kernel。因此就引入了 Scaling 方法

**Scaling（量化时对显著权重进行放大可以降低量化误差）**

量化误差主要来源于对权重的量化，AWQ的目标是通过缩放显著权重，减少量化误差

核心思想：对显著权重按比例放大，然后在计算时相应地缩小输入，这样在量化过程中显著权重的相对误差被降低。

量化函数（这里量化函数不是表示量化后的整数值，而是指 反量化之后的近似权重值，是直接给出最终用于推理的值）：

![image](https://cdn.ipfsscan.io/weibo/large/005wRZF3ly1i4clrb0kfvj30eb02e0sn.jpg)

其中 $N$ 是量化后的比特数，$\Delta$ 是量化因子(scaler)，$\Delta= \dfrac{max(|w|)}{2^{N-1}}$
- $w'=Round(\frac{w}{\Delta})$ 是量化过程，
- $\Delta\cdot w'$ 是反量化过程
- $w, \Delta, x$ 都是 fp16 格式，不会带来精度损失，精度损失全部来源于 round 函数

对于权重 $\text{w}$ 中的单个元素 $w$，引入一个缩放因子 $s>1$，量化过程将 $w$ 与该因子相乘，写作：$\text{w}'=Round(\dfrac{ws}{\Delta'})$，相应将反量化过程写作 $\dfrac{\Delta' \cdot w'}{s}$，对 x 进行逆缩放，则：

![image](https://cdn.ipfsscan.io/weibo/large/005wRZF3ly1i4cltauu51j312504ydfz.jpg)

其原始量化误差为：

![image](https://cdn.ipfsscan.io/weibo/large/005wRZF3ly1i4cltpuin4j30qh046q31.jpg)

RoundErr：四舍五入误差，为±0.5

$\Delta$：量化比例因子，决定误差绝对值大小

缩放后的量化误差：

![image](https://cdn.ipfsscan.io/weibo/large/005wRZF3ly1i4cltkxy43j30us050aa8.jpg)

所以误差比值可以描述为 $\dfrac{\Delta'}{\Delta}\cdot \dfrac{1}{s}$，我们认为 $\Delta'\approx \Delta$，加上 $s>1$，所以作者认为量化时对显著权重进行放大，可以降低量化误差

>从量化函数来看，AWQ 属于对称量化。这里量化因子 q_scale: $\Delta' = \dfrac{max(|w|)}{2^{N-1}}$

**自动计算缩放系数**

按照上文的分析，我们需要找到权重矩阵每个通道的缩放系数，使得量化误差最小，即最小化公式4：

![image](https://cdn.ipfsscan.io/weibo/large/005wRZF3ly1i4clwdh8dcj30fq02xgll.jpg)

按照作者的观点，激活值越大，对应通道越显著，就应该分配更大的缩放系数降低其量化误差。这里为了简单记忆，作者统计了各通道的平均激活值（计算输入矩阵各列绝对值的平均值），并直接将此作为各通道的缩放系数。

#### SmoothQuant

是一个 W8A8 算法，本质还是跟 AWQ 有点像，主要是将 激活量化 的难度转移到权重上，简单来说就是 除以一个值，然后权重乘以一个值：

$$ Y=(X*\text{diag}(s)^{-1})*(\text{diag}(s)*W) $$

也就是对activate (也就是X)进行缩放，并把相反的缩放系数应用到对应的weight(也就是W)上，得到数学上等价的结果。它的核心观察在于：
- 由于activation outlier的存在，activation的分布非常不规则；
- weight分布均匀
这样，通过上面的操作，试图把 activation 变得更均匀，而把 weightde 均匀分布变得没有那么均匀，也就是把activation 量化de 难度部分平摊到weight 上

并且该论文主要实现了三种量化方法：per-tensor、per-token、per-channel。同时也进行了static 和 dynamic量化的区分：

1. static：离线使用标定数据计算好缩放系数
2. dynamic：在线运行的时候统计缩放系数

### weight-activation

- LLM.int8() 发现激活中的异常值集中在一小部分通道中。基于这一点，LLM.int8() 根据输入通道内的离群值分布将激活和权重分成两个不同的部分。包含激活值和权重的异常数据的通道以FP16格式存储，其他通道则以INT8格式存储。

## 支撑量化的一些算子和库

### Marlin

支持混合精度运算，例如 FP16 * INT4 运算，FP8 * INT4运算

一种支持 W4A16的 GEMM kernel（一定程度上kernel 实现和量化算法是独立的），因此 marlin kernel 也支持 AWQ 量化模型执行。原始的Marlin Kernel只支持W4A16计算模式，而 QQQ 在 Marlin kernel 的基础上，支持了 W4A8 的计算模式。
