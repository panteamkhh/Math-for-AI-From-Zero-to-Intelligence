# Layer 4 — Day 27: Continuity

**Keywords:** Continuity

---

### 🎯 Intuition
A continuous function is one whose graph you can draw without lifting your pen off the paper — no jumps, no gaps.

### 💡 ML/AI Application
Activation functions like ReLU are continuous at zero but not "smooth" there (they have a kink). This is exactly why the derivative at that point needs special handling. Understanding continuity helps you understand why gradients can behave strangely at certain points.

### 📝 Mental Exercise
ReLU is continuous at x=0 but not differentiable there. How do you think this contradiction gets resolved in real implementations?

### 🧪 Hands-on
```python
import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    # In practice, frameworks like PyTorch/TensorFlow just PICK a convention
    # at the exact kink (x=0), since a true derivative doesn't exist there.
    # The common convention: treat the derivative at 0 as 0 (or sometimes 1).
    return np.where(x > 0, 1.0, 0.0)

xs = np.array([-2, -0.0001, 0, 0.0001, 2])
print("ReLU(x):          ", relu(xs))
print("ReLU'(x) (chosen):", relu_derivative(xs))
# The function itself has no gap (continuous), but its slope jumps abruptly
# from 0 to 1 right at x=0 -- frameworks simply define a value there by convention
# so backprop always has *something* to work with.
```

---
⬅ [Day 26](day-26-the-concept-of-a-limit.md) · [Layer 4 Summary](layer-4-summary-review.md) · [Day 28 →](day-28-derivative-as-rate-of-change.md)
