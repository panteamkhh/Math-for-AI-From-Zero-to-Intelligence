# Layer 4 — Day 32: The Fundamental Theorem of Calculus & Gradient Intuition

**Keywords:** Fundamental Theorem of Calculus · Gradient Intuition

---

### 🎯 Intuition
The Fundamental Theorem of Calculus connects the derivative and the integral: one is the rate of change, the other is the accumulation of change. The gradient generalizes the derivative to multiple dimensions — instead of one number, it's a vector pointing in the direction of steepest increase.

### 💡 ML/AI Application
The gradient is exactly what we use in Gradient Descent: a vector that tells every parameter in the model which direction to move to reduce the loss. The entire process of deep learning rests on this one idea.

### 📝 Mental Exercise
Why in Gradient Descent do we always move in the direction *opposite* to the gradient, rather than the same direction?

### 🧪 Hands-on
```python
import numpy as np

# A 2D loss surface: loss(w1, w2) = (w1-3)^2 + (w2+2)^2
def loss(w):
    return (w[0]-3)**2 + (w[1]+2)**2

def gradient(w):
    return np.array([2*(w[0]-3), 2*(w[1]+2)])

w = np.array([0.0, 0.0])
lr = 0.1

print(f"Start: w={w}, loss={loss(w):.3f}")
for step in range(20):
    grad = gradient(w)
    w = w - lr * grad   # move OPPOSITE the gradient -- this is the key step
    if step % 5 == 0:
        print(f"Step {step}: w={np.round(w, 3)}, loss={loss(w):.4f}")

print(f"\nFinal: w={np.round(w, 3)} (target was [3, -2])")
# The gradient always points toward STEEPEST INCREASE.
# Since we want to MINIMIZE the loss, we step in the opposite direction -- downhill.
```

---
⬅ [Day 31](day-31-integral-and-area-under-curve.md) · [Layer 4 Summary](layer-4-summary-review.md)

**Next layer:** Layer 5 — Discrete Mathematics →
