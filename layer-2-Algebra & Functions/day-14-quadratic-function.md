# Layer 2 — Day 14: Quadratic Function

**Keywords:** Quadratic Function

---

### 🎯 Intuition
A quadratic function is a symmetric curve (a parabola) with one turning point; its rate of change isn't constant — the rate of change itself changes.

### 💡 ML/AI Application
Regularization terms like L2 (ridge) use the square of the parameters — i.e., a quadratic function of the weights. This makes the penalty for large weights grow sharply (not gently) as weights increase.

### 📝 Mental Exercise
Why do we use the square of the weights in L2 regularization instead of just their absolute value? (Hint: compare the shape of a quadratic function to absolute value.)

### 💬 Worked Answer
Because **L2 Regularization** uses squared weights to **heavily penalize large weights**, while keeping the penalty function smooth and easy to optimize.

For example:
- weight 2 → penalty = 4
- weight 10 → penalty = 100

But with absolute value (L1):
- weight 2 → penalty = 2
- weight 10 → penalty = 10

So **L2 controls very large weights much more aggressively**.

Key difference:
- **L2 (square)** → shrinks weights but usually doesn't zero them out.
- **L1 (absolute value)** → can push some weights to exactly zero, removing low-importance features.

### 🧪 Hands-on
```python
import numpy as np

weights = np.array([0.5, 2, 5, 10])

l1_penalty = np.abs(weights)
l2_penalty = weights ** 2

print("weights:   ", weights)
print("L1 penalty:", l1_penalty)
print("L2 penalty:", l2_penalty)
# Notice how L2's penalty grows MUCH faster for large weights (10 -> 100)
# compared to L1's linear growth (10 -> 10). That's the quadratic shape at work.
```

---
⬅ [Day 13](day-13-linear-function.md) · [Layer 2 Summary](layer-2-summary-review.md) · [Day 15 →](day-15-absolute-value-function.md)
