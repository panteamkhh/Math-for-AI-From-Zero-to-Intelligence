# Layer 4 — Day 26: The Concept of a Limit

**Keywords:** Limit

---

### 🎯 Intuition
A limit is what a function approaches as you get closer and closer to a specific point — even if that exact point itself is unreachable.

### 💡 ML/AI Application
The concept of a limit is the mathematical foundation of the derivative, and the derivative is the foundation of learning in neural networks. When we say "instantaneous rate of change," we're talking about a limit that infinitesimally small changes converge toward.

### 📝 Mental Exercise
Why do we need the concept of a limit to define the "instantaneous slope" of a curve (rather than a straight line)?

### 🧪 Hands-on
```python
import numpy as np

def f(x):
    return x ** 2

x0 = 3.0

# Approximate the "instantaneous slope" at x0 using smaller and smaller steps (h)
# This IS the limit definition of a derivative in action:
# f'(x0) = lim(h->0) [f(x0+h) - f(x0)] / h
for h in [1.0, 0.1, 0.01, 0.001, 0.0001]:
    slope_estimate = (f(x0 + h) - f(x0)) / h
    print(f"h={h:<8} -> slope estimate = {slope_estimate:.5f}")

print("\nTrue derivative (2*x0):", 2 * x0)
# Notice how the estimate keeps getting closer to 6.0 as h shrinks toward 0 --
# that convergence IS the limit.
```

---
⬅ [Layer 4 Summary](layer-4-summary-review.md) · [Day 27 →](day-27-continuity.md)
