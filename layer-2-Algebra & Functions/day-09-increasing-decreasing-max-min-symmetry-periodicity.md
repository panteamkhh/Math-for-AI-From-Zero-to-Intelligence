# Layer 2 — Day 9: Increasing/Decreasing, Max/Min, Symmetry, Periodicity

**Keywords:** Increasing & Decreasing · Max/Min (intuitive) · Symmetry · Periodicity

---

### 🎯 Intuition
A function is increasing if a bigger input always gives a bigger output. A max/min is the highest/lowest point of a function over some interval — the place where the function changes direction.

### 💡 ML/AI Application
Optimization in machine learning is literally about finding the minimum of the loss function. Gradient Descent moves using exactly this "increasing/decreasing" idea: it always moves in the direction where the function is decreasing, to reach the minimum.

### 📝 Mental Exercise
Why is finding the **global minimum** harder in a loss function that has several local minima?

### 💬 Worked Answer
Because Gradient Descent only looks at the **neighborhood of the current point**, and it can get stuck in a **local minimum**.

Meanwhile the **global minimum** is the single best point with the lowest loss across the *entire* model space, and finding the path to it is much harder. 🎯

So:
- One minimum → simpler path
- Several minima → risk of getting stuck at the wrong answer

### 🧪 Hands-on
```python
import numpy as np

def gradient_descent(loss_fn, grad_fn, start, lr=0.1, steps=50):
    w = start
    for _ in range(steps):
        w = w - lr * grad_fn(w)
    return w, loss_fn(w)

# Convex loss: one minimum, easy to find
convex_loss = lambda w: (w - 3) ** 2
convex_grad = lambda w: 2 * (w - 3)

# Non-convex loss: multiple local minima
nonconvex_loss = lambda w: (w - 3) ** 2 + 3 * np.sin(3 * w)
nonconvex_grad = lambda w: 2 * (w - 3) + 9 * np.cos(3 * w)

print("Convex, start=-10:", gradient_descent(convex_loss, convex_grad, start=-10.0))
print("Non-convex, start=-10:", gradient_descent(nonconvex_loss, nonconvex_grad, start=-10.0))
print("Non-convex, start=6:  ", gradient_descent(nonconvex_loss, nonconvex_grad, start=6.0))
# Notice: the non-convex case can land on a different (worse) minimum depending on the starting point.
```

---
⬅ [Day 8](day-08-function-representation-and-graphs.md) · [Layer 2 Summary](layer-2-summary-review.md) · [Day 10 →](day-10-linear-equations.md)
