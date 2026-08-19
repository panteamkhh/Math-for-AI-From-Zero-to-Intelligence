# Layer 8 — Day 70: Activation Functions

**Keywords:** Activation Functions

---

### 🎯 Intuition
Activation functions add non-linearity to a network — without them, any deep network (no matter how large) would ultimately collapse into a single simple linear transformation.

### 💡 ML/AI Application
The choice of activation (ReLU, GELU, sigmoid, tanh) directly affects training speed, gradient stability, and the network's ability to learn complex patterns; ReLU is the most widely used choice in modern deep networks due to its simplicity and resistance to vanishing gradients.

### 📝 Mental Exercise
Why can't even a 1,000-layer network learn more than a simple linear model without a non-linear activation function?

### 🧪 Hands-on
```python
import numpy as np

np.random.seed(0)
x = np.random.randn(1, 3)

def stack_linear_layers(x, n_layers, activation=None):
    h = x
    for _ in range(n_layers):
        W = np.random.randn(3, 3) * 0.5
        h = h @ W
        if activation is not None:
            h = activation(h)
    return h

def relu(x):
    return np.maximum(0, x)

# 10 layers, NO activation -- collapses to one big linear transformation
out_no_activation = stack_linear_layers(x, 10, activation=None)

# 10 layers WITH ReLU -- genuinely different, non-linear behavior
out_with_activation = stack_linear_layers(x, 10, activation=relu)

print("Output (no activation, 10 linear layers):", out_no_activation)
print("Output (ReLU activation, 10 layers):      ", out_with_activation)

# Mathematically: W10 @ (W9 @ (... @ (W1 @ x))) always simplifies to
# (W10 @ W9 @ ... @ W1) @ x -- ONE combined linear transformation, no matter
# how many layers you stack, UNLESS a non-linear activation breaks that chain.
```

---
⬅ [Day 69](day-69-backpropagation.md) · [Layer 8 Summary](layer-8-summary-review.md) · [Day 71 →](day-71-softmax.md)
