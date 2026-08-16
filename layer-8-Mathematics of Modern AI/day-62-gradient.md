# Layer 8 — Day 62: Gradient

**Keywords:** Gradient

---

### 🎯 Intuition
The gradient is the generalization of the derivative to multivariable functions: a vector that shows, for every parameter, the direction and magnitude of the function's steepest increase.

### 💡 ML/AI Application
Every parameter in a neural network (millions or billions of weights) has its own individual gradient; the full gradient vector defines the exact update path for every weight in a single training step.

### 📝 Mental Exercise
Why does computing the gradient for billions of parameters in an LLM require a highly efficient algorithm (backpropagation), rather than computing each derivative by hand?

### 🧪 Hands-on
```python
import numpy as np

# A toy "model" with 4 parameters and a loss that depends on all of them
def loss(w):
    return np.sum((w - np.array([1, -2, 3, 0])) ** 2)

def gradient(w):
    return 2 * (w - np.array([1, -2, 3, 0]))

w = np.array([0.0, 0.0, 0.0, 0.0])
grad_vector = gradient(w)

print("Current parameters:", w)
print("Gradient vector:   ", grad_vector)
print("Each entry tells you which DIRECTION and how STRONGLY that specific")
print("parameter should move to reduce the loss.")

# Scale this to billions of parameters and computing each one "by hand" (one
# at a time, from scratch) becomes computationally impossible -- backpropagation
# reuses shared intermediate computations (via the chain rule) to get ALL
# gradients in roughly the same cost as a single forward pass.
```

---
⬅ [Day 61](day-61-kl-divergence.md) · [Layer 8 Summary](layer-8-summary-review.md) · [Day 63 →](day-63-jacobian.md)
