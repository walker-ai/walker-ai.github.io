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
  enable: false
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

```python
class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        return self.net(x)

class MoE(nn.Module):
    def __init__(self, input_dim, num_experts, top_k, expert_capacity, hidden_dim, output_dim):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_capacity = expert_capacity

        # 路由网络
        self.gate = nn.Linear(input_dim, num_experts)

        # 专家集合
        self.experts = nn.ModuleList(
            [Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)])

    def forward(self, x):
        batch_size, input_dim = x.shape
        device = x.device

        # 路由计算
        logits = self.gate(x)
        probs = torch.softmax(logits, dim=-1)
        print("probs: ", probs)
        topk_probs, topk_indices = torch.topk(probs, self.top_k, dim=-1)
        print("topk_probs: ", topk_probs)
        print("topk_indices: ", topk_indices)
        # 辅助损失计算
        if self.training:
            # 重要性损失（专家利用率均衡）
            importance = probs.sum(0)
            importance_loss = torch.var(importance) / (self.num_experts ** 2)

            # 负载均衡损失（样本分配均衡）
            mask = torch.zeros_like(probs, dtype=torch.bool)
            mask.scatter_(1, topk_indices, True)
            routing_probs = probs * mask
            expert_usage = mask.float().mean(0)
            routing_weights = routing_probs.mean(0)
            load_balance_loss = self.num_experts * (expert_usage * routing_weights).sum()

            aux_loss = importance_loss + load_balance_loss
        else:
            aux_loss = 0.0

        # 专家分配逻辑
        flat_indices = topk_indices.view(-1)
        flat_probs = topk_probs.view(-1)
        sample_indices = torch.arange(batch_size, device=device)[:, None]\
                            .expand(-1, self.top_k).flatten()
        print("sample_indices: ", sample_indices)

        # 初始化输出
        outputs = torch.zeros(batch_size, self.experts[0].net[-1].out_features,
                            device=device)

        # 处理每个专家
        for expert_idx in range(self.num_experts):
            print("expert_idx: ", expert_idx)
            # 获取分配给当前专家的样本
            expert_mask = flat_indices == expert_idx
            print("expert_mask: ", expert_mask)
            expert_samples = sample_indices[expert_mask]
            print("expert_samples: ", expert_samples)
            expert_weights = flat_probs[expert_mask]
            print("expert_weights: ", expert_weights)

            # 容量控制
            if len(expert_samples) > self.expert_capacity:
                expert_samples = expert_samples[:self.expert_capacity]
                expert_weights = expert_weights[:self.expert_capacity]

            if len(expert_samples) == 0:
                continue

            # 处理专家计算
            expert_input = x[expert_samples]
            print("expert_input: ", expert_input)
            expert_output = self.experts[expert_idx](expert_input)
            weighted_output = expert_output * expert_weights.unsqueeze(-1)

            # 累加输出
            outputs.index_add_(0, expert_samples, weighted_output)

        return outputs, aux_loss

# 测试示例
if __name__ == "__main__":
    input_dim = 5
    output_dim = 10
    num_experts = 8
    top_k = 3
    expert_capacity = 32
    hidden_dim = 512
    batch_size = 10

    # add
    device = torch.device("npu:4" if torch.npu.is_available() else "cpu")
    moe = MoE(input_dim, num_experts, top_k, expert_capacity, hidden_dim, output_dim).to(device)
    x = torch.randn(batch_size, input_dim).to(device)
    moe.eval()
    output, _ = moe(x)
    print(f"Eval output shape: {output.shape}") # torch.Size([64, 256])
```


gate 就是一个线性层，形状为 `(hidden_state, n_experts)`

输入 x (num_tokens, hidden_state) 经过 gate 得到 router_logits (num_tokens, n_experts)

然后会经过 topk 来将每个token对应的topk个激活专家选出来，这里可以用python代码简单介绍这一过程：

```python
logits = self.gate(x)
probs = torch.softmax(logits, dim=-1)
```

假设这里的 `num_tokens=10, num_experts=8`，故 probs 是一个10行8列的矩阵

```python
probs:  
tensor([[0.1710, 0.1348, 0.0746, 0.1714, 0.0594, 0.2695, 0.0251, 0.0940],
        [0.1556, 0.0776, 0.1658, 0.1489, 0.1152, 0.1679, 0.0565, 0.1124],
        [0.1077, 0.1154, 0.1564, 0.1317, 0.0630, 0.2026, 0.0518, 0.1715],
        [0.0681, 0.0680, 0.1236, 0.1030, 0.1707, 0.2827, 0.0627, 0.1211],
        [0.0453, 0.0648, 0.2313, 0.0781, 0.1026, 0.1304, 0.1326, 0.2149],
        [0.1394, 0.2278, 0.0625, 0.1832, 0.0395, 0.1512, 0.0691, 0.1274],
        [0.1096, 0.1462, 0.1302, 0.1397, 0.0607, 0.1898, 0.0639, 0.1598],
        [0.1200, 0.1952, 0.0970, 0.1648, 0.0360, 0.1072, 0.1018, 0.1779],
        [0.0650, 0.0501, 0.1463, 0.1025, 0.2219, 0.1446, 0.1439, 0.1257],
        [0.0641, 0.0813, 0.0579, 0.1348, 0.1170, 0.0631, 0.3554, 0.1264]],
)
```

接着，再用topk算子把每个token的激活专家选出来：

```python
topk_probs, topk_indices = torch.topk(probs, self.top_k, dim=-1)
```

> [!TIP] 由此可见 top-k 算子也是非常重要的，实现过程可以看 [CUDA常用算子案例](https://yitaonote.com/2025/ebaa040/)

`topk_probs`和`topk_indices` 的打印结果如下，因为我们设置的top_k=3，所以每个token都把排名前三的概率选出来了，同时`topk_indices`把这些概率对应的专家编号也选出来了，比如第0个token，激活了5号专家、3号专家、0号专家。

```python
topk_probs:  tensor([[0.2695, 0.1714, 0.1710],
        [0.1679, 0.1658, 0.1556],
        [0.2026, 0.1715, 0.1564],
        [0.2827, 0.1707, 0.1236],
        [0.2313, 0.2149, 0.1326],
        [0.2278, 0.1832, 0.1512],
        [0.1898, 0.1598, 0.1462],
        [0.1952, 0.1779, 0.1648],
        [0.2219, 0.1463, 0.1446],
        [0.3554, 0.1348, 0.1264]])
topk_indices:  tensor([[5, 3, 0],
        [5, 2, 0],
        [5, 7, 2],
        [5, 4, 2],
        [2, 7, 6],
        [1, 3, 5],
        [5, 7, 1],
        [1, 7, 3],
        [4, 2, 5],
        [6, 3, 7]])
```

选择好专家后，就要开始计算了。计算规则是，对于每个token，假如它选择的专家是e1、e2、e3，概率分别是p1、p2、p3，那么这个token的计算结果就是 `p1*e1_out+p2*e2_out+p3*e3_out`。

> [!TIPS] 这里实际的 prob 应该还要进行归一化

由于计算个体是每个专家，所以代码中用for循环遍历每个专家。我们以第0个专家为例，看看它的计算过程是怎样的。首先需要确定0号专家的输入。由于不是每个token都选择了0号专家，所以不能把x直接作为输入，而是要确定一个下标向量idxes，把x[idxes]作为0号专家的输入，idxes的值就是激活了0号专家的所有token编号，那么怎么得到idxes呢？

首先计算一个mask：

```python
expert_mask = flat_indices == expert_idx

# 结果：
expert_mask:  tensor([False, False,  True, False, False,  True, False, False, False, False,
        False, False, False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False, False])
```

`flat_indices`是`topk_indices`平铺之后的向量。通过对比，可以看到`expert_mask`中True的位置和`topk_indices`中0的位置铺平之后是一致的，代表第0个专家被第0个token和第1个token激活了。

而且`expert_mask`代表的含义是：只要它的第0-2的位置是True的话，就代表被第0个token激活了，只要它的第3-5的位置是True的话，就代表被第1个token激活了，以此类推，我们可以声明一个`sample_indices`向量：

```python
sample_indices:  tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7,
        8, 8, 8, 9, 9, 9])
```

再通过下面的代码就可以把idxes、概率权重、输入都取出来了：

```python
expert_samples = sample_indices[expert_mask]
expert_weights = flat_probs[expert_mask]
expert_input = x[expert_samples]
```

再进行专家计算，并把计算结果叠加到对应的token上面去：：

```python
expert_output = self.experts[expert_idx](expert_input)
weighted_output = expert_output * expert_weights.unsqueeze(-1)

outputs.index_add_(0, expert_samples, weighted_output)
```



然后如果有配置共享专家，则会先经过共享专家，共享专家也是基本的MLP层，共享专家是一直激活的

> [!TIP] 在 DeepSeek-V3中，MoE层一般总共有256个路由专家，1个共享专家

MLP层一般会将 gate(w_1, 即经过激活函数的linear) 和 up(w_3)权重进行融合，形成 gate_up_proj(w_13)，而 down(w_2) 权重


DeepSeek MoE 架构的公式形式：

![image](https://cdn.ipfsscan.io/weibo/large/005wRZF3gy1i4lbc6xdmkj314009amxt.jpg)

这里的公式其实结合上面代码理解非常简单，首先 $\mathbf{h}_t^l$ 是整个MoE部分的输出，其中 $\mathbf{u}_t^l$ 是经过MoE部分之前的输入，这里也是因为残差连接直接进行加和，而 $g_{i,t}$ 这一项是所有路由专家的加权计算结果，$g_{i,t}$ 表示每个选中的路由专家(top-k个)的**用于加和**的权重（并非本身FFN层的权重），而 $\text{FFN}_i(\mathbf{u}_t^l)$ 表示每个路由专家的计算结果。

那么这里的 $s_{i,t} = Softmax_i(\mathbf{u}_t^{l^T}e_i^l)$ 其实对应的代码就是：

```python
logits = self.gate(x)
probs = torch.softmax(logits, dim=-1)
```

这里 $e_i^l$ 论文中被称为可学习的参数，其实我理解就是 gate 这个线性层权重。

那么对于 DeepSeek-V3，gate部分有略微的改动：

![image](https://cdn.ipfsscan.io/weibo/large/005wRZF3gy1i4lbkkt9l1j30jz06imxb.jpg)



了解了MoE的计算过程，接下来看看专家并行：

> 参考文献：
> 1. [https://zhuanlan.zhihu.com/p/1918753864556974722](https://zhuanlan.zhihu.com/p/1918753864556974722)
> 2. https://zhuanlan.zhihu.com/p/681154742

![image](https://cdn.ipfsscan.io/weibo/large/005wRZF3gy1i4lfix454vj31400z7ju5.jpg)

专家并行的目标是将一个 MoE 层中的众多专家分布到不同的设备上，每个设备负责一部分专家。如果某个设备上的计算需要其他设备的专家，可以通过All2All通信实现。

专家并行思想来源论文：《GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding》

具体来说，MoE模型通常使用 Gating 模块来决定每个输入数据样本应该由哪些专家来处理。假设有一个输入数据样本位于设备 A 上，而 Gating 模块决定该样本应该由设备 B 和设备 C 上的专家来处理，那么就需要将该数据样本从设备 A 传输到设备 B 和设备 C。

标准 All-to-All

> 在一个由 N 个节点组成的群体中，每一个节点都需要向其他 所有 N-1 个节点发送一份不同的数据，同时也需要从其他 所有 N-1 个节点接收一份不同的数据。

![image](https://cdn.ipfsscan.io/weibo/large/005wRZF3gy1i4lc9g945kj31400gd0u0.jpg)

非标准 All-to-All

简单来说就是有可能发送到不同设备的数据量不同，从不同设备接收的数据量也可能不同。

![image](https://cdn.ipfsscan.io/weibo/large/005wRZF3gy1i4lcgj6b17j30nd0f2gms.jpg)

上述非标准 All2All 中有个问题：有些时候当前设备只知道要向其他设备发送多少数据，而并不知道需要从其他设备接收多少数据。

这个问题可以通过 2 次 all2all 来解决：

- 第一次 all2all 交换要传输的数据量信息，这是一个标准的 all2all 操作。
- 第二次 all2all 根据上述获取的数据量信息来执行真正的数据传输，此时是一个非标准 all2all 操作。

## MTP


## DeepSeek开源周

### FlashMLA

### DeepEP

### DeepGEMM

### EPLB

#### 背景动机

专家并行时，如何决定将那个专家放到哪张卡上。

考虑DeepSeek的EP，总共256个路由专家，1个共享专家.

prefill时,EP32, 256/32 = 8，每张卡放8个路由专家， 共享专家在所有卡上都复制一份，单卡总共9个专家。

decode时，EP144, 每张卡只放2个路由专家和一个共享专家，总共有 144 * 2-256 = 32个来放冗余路由专家。

需要解决的问题：

1. 怎么决定对哪些专家进行冗余？

2. 冗余多少份？

3. 对于任意一个专家，应该放在哪张卡上？

EPLB 就是在解决上述问题。

逻辑专家：指模型中的256路由专家 + 1共享专家

物理专家：指经过冗余后实际部署到GPU上的专家, 数量大于 256 + 1


[DeepSeek 推理系统概览](https://zhuanlan.zhihu.com/p/27181462601)

> **Prefill**：路由专家 EP32、MLA 和共享专家 DP32，一个部署单元是 4 节点，32 个冗余路由专家，每张卡 9 个路由专家和 1 个共享专家

> **Decode**：路由专家 EP144、MLA 和共享专家 DP144，一个部署单元是 18 节点，32 个冗余路由专家，每张卡 2 个路由专家和 1 个共享专家


官网例子：
```python
import torch
import eplb

# 这里的weight是记录每一个专家历史工作负载，来评估每个专家的“热门”程度
weight = torch.tensor([[ 90, 132,  40,  61, 104, 165,  39,   4,  73,  56, 183,  86],
                       [ 20, 107, 104,  64,  19, 197, 187, 157, 172,  86,  16,  27]])

num_replicas = 16  # 实际可以放置的总物理专家数量
num_groups = 4     # 对总卡数进行分组
num_nodes = 2      # 节点总数
num_gpus = 8       # 卡总数

phy2log, log2phy, logcnt = eplb.rebalance_experts(weight, num_replicas, num_groups, num_nodes, num_gpus)
print(phy2log)

# 最后输出负载均衡后的推荐放置方案
# Output:
# tensor([[ 5,  6,  5,  7,  8,  4,  3,  4, 10,  9, 10,  2,  0,  1, 11,  1],
#         [ 7, 10,  6,  8,  6, 11,  8,  9,  2,  4,  5,  1,  5,  0,  3,  1]])
```

该示例展示了一个两层的 MoE 模型，每层包含 12 个专家。每层引入 4 个冗余专家，总共 16 个副本被放置在 2 个节点上，每个节点包含4个 GPU。输出结果展示了专家复制和放置的计划。

![image](https://cdn.ipfsscan.io/weibo/large/005wRZF3gy1i4lphi3qmgj314006pt9t.jpg)


#### EPLB核心函数
```python
def balanced_packing(weight: torch.Tensor, num_packs: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pack n weighted objects to m packs, such that each bin contains exactly n/m objects and the weights of all packs
    are as balanced as possible.

    Parameters:
        weight: [X, n], the weight of each item
        num_packs: number of packs

    Returns:
        pack_index: [X, n], the pack index of each item
        rank_in_pack: [X, n], the rank of the item in the pack
    """
    num_layers, num_groups = weight.shape
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs

    if groups_per_pack == 1:
        pack_index = torch.arange(weight.size(-1), dtype=torch.int64, device=weight.device).expand(weight.shape)
        rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
        return pack_index, rank_in_pack

    indices = weight.float().sort(-1, descending=True).indices.cpu()
    pack_index = torch.full_like(weight, fill_value=-1, dtype=torch.int64, device='cpu')
    rank_in_pack = torch.full_like(pack_index, fill_value=-1)
    for i in range(num_layers):
        pack_weights = [0] * num_packs
        pack_items = [0] * num_packs
        for group in indices[i]:
            pack = min((i for i in range(num_packs) if pack_items[i] < groups_per_pack),
                       key=pack_weights.__getitem__)
            assert pack_items[pack] < groups_per_pack
            pack_index[i, group] = pack
            rank_in_pack[i, group] = pack_items[pack]
            pack_weights[pack] += weight[i, group]
            pack_items[pack] += 1
    return pack_index, rank_in_pack


def replicate_experts(weight: torch.Tensor, num_phy: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    """
    Replicate `num_log` experts to `num_phy` replicas, such that the maximum load of all replicas is minimized.

    Parameters:
        weight: [X, num_log]
        num_phy: total number of experts after replication

    Returns:
        phy2log: [X, num_phy], logical expert id of each physical expert
        rank: [X, num_phy], the replica rank
        logcnt: [X, num_log], number of replicas for each logical expert
    """
    n, num_log = weight.shape
    num_redundant = num_phy - num_log
    assert num_redundant >= 0
    device = weight.device
    phy2log = torch.arange(num_phy, dtype=torch.int64, device=device).repeat(n, 1)
    rank = torch.zeros(n, num_phy, dtype=torch.int64, device=device)
    logcnt = torch.ones(n, num_log, dtype=torch.int64, device=device)
    arangen = torch.arange(n, dtype=torch.int64, device=device)
    for i in range(num_log, num_phy):
        redundant_indices = (weight / logcnt).max(dim=-1).indices
        phy2log[:, i] = redundant_indices
        rank[:, i] = logcnt[arangen, redundant_indices]
        logcnt[arangen, redundant_indices] += 1
    return phy2log, rank, logcnt
```

#### 核心API
```python
# 分层均衡
def rebalance_experts_hierarchical(weight: torch.Tensor, num_physical_experts: int,
                      num_groups: int, num_nodes: int, num_gpus: int):
    """
    Parameters:
        weight: [num_moe_layers, num_logical_experts]
        num_physical_experts: number of physical experts after replication
        num_groups: number of expert groups
        num_nodes: number of server nodes, where the intra-node network (e.g, NVLink) is faster
        num_gpus: number of GPUs, must be a multiple of `num_nodes`

    Returns:
        physical_to_logical_map: [num_moe_layers, num_physical_experts]
        logical_to_physical_map: [num_moe_layers, num_logical_experts, X]
        logical_count: [num_moe_layers, num_logical_experts]
    """

    num_layers, num_logical_experts = weight.shape
    assert num_logical_experts % num_groups == 0
    group_size = num_logical_experts // num_groups
    assert num_groups % num_nodes == 0
    groups_per_node = num_groups // num_nodes
    assert num_gpus % num_nodes == 0
    assert num_physical_experts % num_gpus == 0
    phy_experts_per_gpu = num_physical_experts // num_gpus

    def inverse(perm: torch.Tensor) -> torch.Tensor:
        inv = torch.empty_like(perm)
        inv.scatter_(1, perm, torch.arange(perm.size(1), dtype=torch.int64, device=perm.device).expand(perm.shape))
        return inv

    # Step 1: 将专家组均匀分配到各个节点，确保不同节点的负载平衡
    # 将权重矩阵按组进行展开并计算每组的总负载
    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    # 使用 balanced_packing 函数将专家组打包到节点上，
    # 得到每个组所在的节点索引和在该节点内的排名
    group_pack_index, group_rank_in_pack = balanced_packing(tokens_per_group, num_nodes)
    # 计算逻辑专家到中间逻辑专家的映射
    log2mlog = (((group_pack_index * groups_per_node + group_rank_in_pack) * group_size).unsqueeze(-1) +
                torch.arange(group_size, dtype=torch.int64, device=group_pack_index.device)).flatten(-2)
        # 计算中间逻辑专家到逻辑专家的逆映射
    mlog2log = inverse(log2mlog)

    # Step 2: 在每个节点内复制专家，以最小化所有副本的最大负载。
    # [num_layers * num_nodes, num_logical_experts // num_nodes]
    # 根据中间逻辑专家到逻辑专家的映射，重新排列权重矩阵，并按节点进行分组
    tokens_per_mlog = weight.gather(-1, mlog2log).view(-1, num_logical_experts // num_nodes)
    # 使用 replicate_experts 函数在每个节点内复制专家，
    # 得到物理专家到中间逻辑专家的映射、物理专家的排名和每个中间逻辑专家的副本数
    phy2mlog, phyrank, mlogcnt = replicate_experts(tokens_per_mlog, num_physical_experts // num_nodes)    

    # Step 3: 将复制后的专家分配到各个 GPU 上，确保不同 GPU 的负载平衡。
    # [num_layers * num_nodes, num_physical_experts // num_nodes]
    # 计算每个物理专家的负载
    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
    # 使用 balanced_packing 函数将物理专家打包到每个节点内的 GPU 上，
    # 得到每个物理专家所在的 GPU 索引和在该 GPU 内的排名
    pack_index, rank_in_pack = balanced_packing(tokens_per_phy, num_gpus // num_nodes)
    # 计算物理专家到最终物理专家的映射
    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    # 计算最终物理专家到物理专家的逆映射
    pphy2phy = inverse(phy2pphy)

        # 根据最终物理专家到物理专家的映射，重新排列物理专家到中间逻辑专家的映射
    pphy2mlog = phy2mlog.gather(-1, pphy2phy) # [num_layers * num_nodes, num_log_per_nodes]
    # 调整 pphy2mlog 的形状，使其包含所有节点的信息
    pphy2mlog = (pphy2mlog.view(num_layers, num_nodes, -1) +
                 torch.arange(0, num_logical_experts, num_logical_experts // num_nodes).view(1, -1, 1)).flatten(-2)
    # 根据中间逻辑专家到逻辑专家的映射，将 pphy2mlog 转换为最终物理专家到逻辑专家的映射  
    pphy2log = mlog2log.gather(-1, pphy2mlog)
    # 根据最终物理专家到物理专家的映射，重新排列物理专家的排名
    pphyrank = phyrank.gather(-1, pphy2phy).view(num_layers, -1)
    # 根据逻辑专家到中间逻辑专家的映射，计算每个逻辑专家的副本数
    logcnt = mlogcnt.view(num_layers, -1).gather(-1, log2mlog)
    return pphy2log, pphyrank, logcnt

# 全局均衡（适用于推理时更高的专家并行度）
def rebalance_experts(weight: torch.Tensor, num_replicas: int, num_groups: int,
                      num_nodes: int, num_gpus: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Entry point for expert-parallelism load balancer.

    Parameters:
        weight: [layers, num_logical_experts], the load statistics for all logical experts
        num_replicas: number of physical experts, must be a multiple of `num_gpus`
        num_groups: number of expert groups
        num_nodes: number of server nodes, where the intra-node network (e.g, NVLink) is faster
        num_gpus: number of GPUs, must be a multiple of `num_nodes`

    Returns:
        physical_to_logical_map: [layers, num_replicas], the expert index of each replica
        logical_to_physical_map: [layers, num_logical_experts, X], the replica indices for each expert
        expert_count: [layers, num_logical_experts], number of physical replicas for each logical expert
    """
    num_layers, num_logical_experts = weight.shape
    weight = weight.float().cpu()
    if num_groups % num_nodes == 0:
        # use hierarchical load-balance policy
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(weight, num_replicas,
                                                                  num_groups, num_nodes, num_gpus)
    else:
        # use global load-balance policy
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(weight, num_replicas, 1, 1, num_gpus)
    maxlogcnt = logcnt.max().item()
    log2phy: torch.Tensor = torch.full((num_layers, num_logical_experts, maxlogcnt),
                                       -1, dtype=torch.int64, device=logcnt.device)
    log2phy.view(num_layers, -1).scatter_(-1, phy2log * maxlogcnt + phyrank,
            torch.arange(num_replicas, dtype=torch.int64, device=log2phy.device).expand(num_layers, -1))
    return phy2log, log2phy, logcnt
```
