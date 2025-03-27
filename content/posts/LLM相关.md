---
title: LLM相关
subtitle:
date: 2025-03-27T12:51:11+08:00
slug: a69a1e8
draft: true
author:
  name: yitao
---

记录一些 LLM 基础知识

<!--more-->


### 手撕MHA

#### `numpy` 实现
```python
import numpy as np

def compute_qkv(X, W_q, W_k, W_v):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v

    return Q, K, V

def self_attention(Q, K, V, is_causal):
    d_k = Q.shape[1]
    score = np.matmul(Q, K.T) / np.sqrt(d_k)

    # mask
    if is_causal:
        q_len, k_len = scores.shape
        mask = np.tril(np.ones((q_len, k_len)), k_len - q_len)
        scores = np.where(mask, scores, -np.inf)

    score_max = np.max(score, axis=1, keepdims=True)
    
    score_output = np.exp(score - score_max) / np.sum(np.exp(score - score_max), axis=1, keepdims=True)

    score_output = np.matmul(score_output, V)

    return score_output

def multi_head_attention(Q, K, V, n_heads):
    d_model = Q.shape[1]
    assert d_model % n_heads == 0

    d_k = d_model // n_heads

    # 拼接历史缓存
    if past_key_value is not None:
        K = np.concatenate([past_key_value[0], K], axis=0)
        V = np.concatenate([past_key_value[1], V], axis=0)
    new_cache = (K, V)

    attention = []

    # Q->(seq_len, d_model), reshape to (n_heads, seq_len, d_k)
    Q_reshaped = Q.reshape(Q.shape[0], n_heads, d_k).transpose(1, 0, 2)
    K_reshaped = K.reshape(Q.shape[0], n_heads, d_k).transpose(1, 0, 2)
    V_reshaped = V.reshape(Q.shape[0], n_heads, d_k).transpose(1, 0, 2)

    for i in range(n_heads):
        attention_output = self_attention(Q_reshaped[i], K_reshaped[i], V_reshaped[i])
        attention.append(attention_output)

    attention = np.concatenate(attention, axis=-1)

    return attention, new_cache
```
