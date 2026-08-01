# Layer 4 — Day 29: Derivative Rules & Derivatives of Key Functions

**Keywords:** Derivative Rules · Derivatives of Key Functions

---

### 🎯 Intuition
Derivative rules are tools that let us find the derivative of a complex function without computing a limit from scratch every time — like a ready-made recipe for any function shape.

### 💡 ML/AI Application
Backpropagation uses exactly these rules (especially the chain rule) to automatically and efficiently compute the derivative of the loss with respect to millions of parameters in a deep network.

### 📝 Mental Exercise
Why is having ready-made derivative rules (instead of computing a limit from scratch) essential for a network with millions of parameters?

### 🧪 Hands-on
```python
import time
import numpy as np

# Simulate the cost difference: computing a derivative via the LIMIT DEFINITION
# vs. using the known RULE (chain rule / power rule) for many parameters at once.

n_params = 1_000_000
w = np.random.randn(n_params)

def loss_from_weights(w):
    return np.sum(w ** 2)  # a toy loss over a million weights

# Naive: numerically approximate EVERY partial derivative with tiny steps (slow, one at a time)
start = time.time()
h = 1e-5
grad_numeric = np.zeros(5)  # only doing 5 of them -- doing all 1M this way would take forever
base = loss_from_weights(w)
for i in range(5):
    w_step = w.copy()
    w_step[i] += h
    grad_numeric[i] = (loss_from_weights(w_step) - base) / h
print(f"Numeric approx for 5/1,000,000 params took {time.time()-start:.4f}s")

# Using the known derivative RULE (d/dw[w^2] = 2w), we get ALL 1,000,000 gradients instantly:
start = time.time()
grad_rule = 2 * w
print(f"Rule-based gradient for ALL {n_params:,} params took {time.time()-start:.6f}s")
```

---
⬅ [Day 28](day-28-derivative-as-rate-of-change.md) · [Layer 4 Summary](layer-4-summary-review.md) · [Day 30 →](day-30-optimization-critical-points-max-min.md)
