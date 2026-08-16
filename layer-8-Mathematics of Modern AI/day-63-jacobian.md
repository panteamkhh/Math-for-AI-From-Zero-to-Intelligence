# Layer 8 — Day 63: Jacobian

**Keywords:** Jacobian

---

### 🎯 Intuition
The Jacobian generalizes the gradient to cases where the output is also multi-dimensional — a matrix showing how every output changes with respect to every input.

### 💡 ML/AI Application
When a neural network layer has multiple outputs (like softmax over several classes), backpropagation uses the Jacobian matrix to pass gradients between layers.

### 📝 Mental Exercise
Why does a softmax layer (which has multiple inputs and multiple outputs) require a full Jacobian matrix, rather than a single ordinary derivative?

### 🧪 Hands-on
```python
import numpy as np

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def softmax_jacobian(s):
    # J[i, j] = d(softmax_i) / d(input_j)
    n = len(s)
    J = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            J[i, j] = s[i] * (1 - s[i]) if i == j else -s[i] * s[j]
    return J

logits = np.array([2.0, 1.0, 0.1])
probs = softmax(logits)
J = softmax_jacobian(probs)

print("Softmax output (3 outputs):", np.round(probs, 3))
print("Jacobian (3x3 matrix -- every output vs every input):\n", np.round(J, 3))
# A single number can't capture "how does EACH of the 3 outputs change with
# EACH of the 3 inputs" -- that requires a full matrix of 9 partial derivatives,
# which is exactly what the Jacobian provides.
```

---
⬅ [Day 62](day-62-gradient.md) · [Layer 8 Summary](layer-8-summary-review.md) · [Day 64 →](day-64-hessian.md)
