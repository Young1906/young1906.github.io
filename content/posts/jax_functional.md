---
title: Learning JAX again 
date : 2024-05-08
tags : [learn, ml]
draft: true 
categories: [
    "Machine Learning",
    ]
---

I have been using JAX 

## Simple MLP example

```python

import jax.random as jr
import jax.numpy as jnp
from jax.nn.initializers import Initializer
from jax.nn.initializers import lecun_normal

default_initializer = lecun_normal()

def linear(
    in_features  : int,
    out_features : int,
    width        : int,
    depth        : int,
    initializer  : Initializer = default_initializer):

    def init_fn(key):
        kernel_key, bias_key = jr.split(key, 2)
        kernel = initializer(kernel_key, (in_features, out_features))
        bias   = initializer(bias_key, (out_features, 1))
        bias   = jnp.squeeze(bias)

        return (kernel, bias)

    @jax.jit
    def apply_fn(params, inputs):
        kernel, bias = params
        return jnp.einsum("p q, ... p -> ... q", kernel, inputs) + bias

    return init_fn, apply_fn
```



## Single-controller JAX: one nodes, multiple GPUs 

## Multiple-controllers JAX: one nodes, multiple GPUs 
