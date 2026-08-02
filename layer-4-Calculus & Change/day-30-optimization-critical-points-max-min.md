# Layer 4 — Day 30: Optimization, Critical Points, Maximum & Minimum

**Keywords:** Optimization · Critical Points · Maximum & Minimum

---

### 🎯 Intuition
A critical point is where a function's slope becomes zero — where the function is momentarily "flat." These points are candidates for a maximum, a minimum, or a saddle point.

### 💡 ML/AI Application
Training a model means searching for a point where the gradient (multi-dimensional slope) is close to zero — meaning the loss no longer decreases with small changes to the weights. This is the moment we call the model "converged."

### 📝 Mental Exercise
Why doesn't reaching a gradient of zero always mean you've found the best possible model? (Hint: saddle points and local minima.)

### 🧪 Hands-on
```python
import numpy as np

# A loss surface with a local minimum, a saddle point, AND a global minimum
def loss(w):
    return w**4 - 4*w**3 + 2*w**2 + 4*w   # a quartic with multiple critical points

def grad(w):
    return 4*w**3 - 12*w**2 + 4*w + 4

def gradient_descent(start, lr=0.01, steps=200):
    w = start
    for _ in range(steps):
        w = w - lr * grad(w)
    return w, loss(w)

for start in [-2.0, 0.5, 3.5]:
    final_w, final_loss = gradient_descent(start)
    print(f"start={start:5} -> converged to w={final_w:.3f}, loss={final_loss:.3f}, gradient~0: {abs(grad(final_w)) < 0.01}")

# Notice: different starting points can converge to DIFFERENT critical points --
# the gradient is ~0 at all of them, but they are not all equally good (some are
# local minima, not the true global minimum).
```

---
⬅ [Day 29](day-29-derivative-rules.md) · [Layer 4 Summary](layer-4-summary-review.md) · [Day 31 →](day-31-integral-and-area-under-curve.md)
