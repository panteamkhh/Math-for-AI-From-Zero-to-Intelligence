# Layer 8 — Day 58: Convexity

**Keywords:** Convexity

---

### 🎯 Intuition
A convex function has a bowl shape with just one minimum — every local minimum is also the global minimum. A non-convex function can have several dips.

### 💡 ML/AI Application
Linear and logistic regression are convex, so gradient descent always reaches the global minimum. But deep network losses are usually non-convex, which is why the training outcome is sensitive to initialization and the training path.

### 📝 Mental Exercise
Why can training a (non-convex) deep neural network with different random seeds give slightly different results, while linear regression always gives the same answer?

### 🧪 Hands-on
```python
import numpy as np

# Convex loss: ALWAYS converges to the same point, regardless of start
convex_grad = lambda w: 2 * (w - 3)

# Non-convex loss: has multiple local minima -- start matters
nonconvex_grad = lambda w: 4*w**3 - 12*w**2 + 4*w + 4

def gd(grad_fn, start, lr=0.02, steps=300):
    w = start
    for _ in range(steps):
        w = w - lr * grad_fn(w)
    return w

for seed_start in [-2.0, 0.5, 3.5]:
    print(f"Convex,     start={seed_start:5} -> converges to {gd(convex_grad, seed_start):.3f}")
for seed_start in [-2.0, 0.5, 3.5]:
    print(f"Non-convex, start={seed_start:5} -> converges to {gd(nonconvex_grad, seed_start):.3f}")
# Notice: the convex case always lands on the SAME point no matter the start.
# The non-convex case lands on DIFFERENT points -- exactly why different random
# weight initializations (different "starting points") give deep networks
# slightly different final results.
```

---
⬅ [Day 57](day-57-gradient-descent-and-learning-rate.md) · [Layer 8 Summary](layer-8-summary-review.md) · [Day 59 →](day-59-entropy.md)
