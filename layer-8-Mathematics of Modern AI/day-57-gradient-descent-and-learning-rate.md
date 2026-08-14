# Layer 8 — Day 57: Gradient Descent & Learning Rate

**Keywords:** Gradient Descent · Learning Rate

---

### 🎯 Intuition
Gradient Descent means taking steps in the direction opposite the gradient to reach a minimum. The learning rate determines how big each step is.

### 💡 ML/AI Application
Too large a learning rate causes the model to overshoot the minimum and diverge; too small a rate makes training painfully slow. It's one of the most sensitive hyperparameters every ML engineer has to tune.

### 📝 Mental Exercise
If the loss suddenly spikes (explodes) during training instead of decreasing, what's the first thing you should check?

### 🧪 Hands-on
```python
import numpy as np

def loss(w):
    return (w - 3) ** 2

def grad(w):
    return 2 * (w - 3)

def train(lr, steps=10):
    w = 0.0
    trace = [w]
    for _ in range(steps):
        w = w - lr * grad(w)
        trace.append(w)
    return trace

print("Good learning rate (0.1):", [round(v, 2) for v in train(0.1)])
print("Too-large learning rate (1.1):", [round(v, 2) for v in train(1.1)])
# With lr=1.1, watch the values grow in magnitude each step instead of converging --
# this divergence is exactly what an exploding loss looks like. The first thing
# to check when loss explodes: is the learning rate too high?
```

---
⬅ [Day 56](day-56-loss-function-and-optimization.md) · [Layer 8 Summary](layer-8-summary-review.md) · [Day 58 →](day-58-convexity.md)
