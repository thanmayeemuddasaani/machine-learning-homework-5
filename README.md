# Machine Learning – Homework 5  
## Transformer Attention & Encoder Block Implementation
## Overview

This project implements two core components of transformer architectures:

1. Scaled Dot-Product Attention using NumPy
2. A Simplified Transformer Encoder Block using PyTorch

Both components were executed and verified with correct outputs and expected tensor shapes.
## 1. Scaled Dot-Product Attention

### Description

The implementation computes:

- Attention scores using QK^T  
- Scaled scores using sqrt(d_k)  
- Softmax-normalized attention weights  
- Context vectors produced from V  

This follows the standard transformer attention mechanism.

### Code: Scaled Dot-Product Attention (NumPy)

```python
import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def scaled_dot_product_attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = np.dot(Q, K.T)
    scaled_scores = scores / np.sqrt(d_k)
    attention_weights = softmax(scaled_scores)
    context = np.dot(attention_weights, V)
    return attention_weights, context
```

---
## 1. Scaled Dot-Product Attention Output

### Attention Weights:

Each row represents how much a query token attends to all other tokens.  
Softmax ensures each row sums to 1.  
Higher values indicate stronger attention toward specific tokens.

### Context Vector:

Computed as the weighted sum of the value matrix.  
Represents the updated embedding for each token after attention is applied.  
Encodes information gathered from all other tokens in the sequence.

### Interpretation

Attention weights determine which tokens influence each other.  
Context vectors contain the resulting combined information, forming the final token representations after attention.

## 2. Transformer Encoder Block Output

### Input Shape:

`(32, 10, 64)` → 32 samples, 10 tokens each, embedding size 64.

### Output Shape:

`(32, 10, 64)` → Same shape because:

- Multi-head attention preserves embedding size  
- Feed-forward layer expands → compresses back to 64  
- Residual connections require matching dimensions  
- LayerNorm does not change shape  

### Interpretation

The model changes the representation, not the shape, confirming the encoder block works correctly.



