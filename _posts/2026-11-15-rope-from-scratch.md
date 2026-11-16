---
layout: post
title: RoPE from scratch
date: 2026-11-15 12:00:00
description: Rotary positional embeddings explained with one formula, one picture and a few lines of PyTorch.
tags: transformers attention positional-encoding
categories: from-scratch
---

Self-attention does not know the order of tokens: if you shuffle the input, the attention scores stay the same.
A positional encoding fixes that. In this post I implement **rotary positional embeddings** (RoPE), introduced in [RoFormer](https://arxiv.org/abs/2104.09864) and used in most modern LLMs.

## The idea

Instead of _adding_ a position vector to the token embedding, RoPE _rotates_ the query and key vectors by an angle that depends on the position.
Split a vector of size $$d$$ into $$d/2$$ pairs. The pair number $$i$$ of a token at position $$m$$ is rotated by the angle $$m\theta_i$$:

$$
\begin{pmatrix} x'_{2i} \\ x'_{2i+1} \end{pmatrix}
=
\begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{pmatrix}
\begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix},
\qquad
\theta_i = 10000^{-2i/d}
$$

Each pair has its own frequency: the first pairs rotate fast, the last ones rotate slowly.

## Why rotations

Let $$R_m$$ be the rotation applied at position $$m$$. Rotations preserve dot products, so the attention score between a query at position $$m$$ and a key at position $$n$$ is

$$
\langle R_m q, R_n k \rangle = q^\top R_m^\top R_n k = q^\top R_{n-m} k
$$

The score depends only on the **relative** position $$n - m$$, not on the absolute positions. This is exactly what we want from attention.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/rope/rope_rotation.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Left: one pair of dimensions is rotated further at each position. Right: the score between the same vector placed at two positions depends only on the distance between them.
</div>

## Implementation

A 2D rotation is a multiplication by the complex number $$e^{i m \theta}$$, so the code is short: view each pair as a complex number, multiply, and view it back as real numbers.

```python
import torch


def rope_frequencies(head_dim: int, seq_len: int, base: float = 10000.0) -> torch.Tensor:
    """Complex rotations e^{i * m * theta_i}, shape (seq_len, head_dim // 2)."""
    assert head_dim % 2 == 0, "head_dim must be even"
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(seq_len).float()
    angles = torch.outer(positions, theta)  # (seq_len, head_dim // 2)
    return torch.polar(torch.ones_like(angles), angles)


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Rotate x of shape (batch, heads, seq_len, head_dim)."""
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    x_rotated = x_complex * freqs[: x.shape[-2]]
    return torch.view_as_real(x_rotated).flatten(-2).type_as(x)
```

In an attention layer, RoPE is applied to queries and keys (not to values) right before the dot product:

```python
freqs = rope_frequencies(head_dim=64, seq_len=128)

q = torch.randn(2, 8, 128, 64)  # (batch, heads, seq_len, head_dim)
k = torch.randn(2, 8, 128, 64)

q, k = apply_rope(q, freqs), apply_rope(k, freqs)
scores = q @ k.transpose(-2, -1) / 64**0.5
```

## Takeaways

- RoPE encodes position by rotating pairs of dimensions, it adds no trainable parameters.
- The attention score depends only on the relative position of two tokens.
- It is applied inside every attention layer to queries and keys, not once at the input.
