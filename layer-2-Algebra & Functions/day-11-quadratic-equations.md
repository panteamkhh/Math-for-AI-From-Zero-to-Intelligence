# Layer 2 — Day 11: Quadratic Equations

**Keywords:** Quadratic Equations

---

### 🎯 Intuition
A quadratic equation is a relationship where the input is raised to the power of 2, and its graph is bowl-shaped (a parabola) — meaning it has exactly one minimum or maximum point.

### 💡 ML/AI Application
Many simple loss functions (like Mean Squared Error) are quadratic in shape, which makes them "bowl-shaped" (convex) — so finding their minimum with gradient descent is reliable and stable.

### 📝 Mental Exercise
Why does a "bowl-shaped" loss function make the optimization algorithm's job easier compared to a shape with several dips?

### 💬 Worked Answer
Because a **bowl-shaped (convex)** function has only one minimum, so Gradient Descent has a clear path down to the lowest point, and the chance of getting stuck at the wrong point is low.

But in a function with several dips, the algorithm may get stuck in a **local minimum** and never reach the global minimum.

So:
- **Convex loss → simpler path, more stable result** ✅
- **Multiple minima → risk of getting stuck without finding the best answer**

### 🧪 Hands-on
```python
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

w = np.linspace(-2, 8, 200)

mse_loss = (w - 3) ** 2                          # convex: one bowl
bumpy_loss = (w - 3) ** 2 + 4 * np.sin(2 * w)     # non-convex: several dips

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(w, mse_loss); axes[0].set_title("Convex (MSE-like) loss")
axes[1].plot(w, bumpy_loss); axes[1].set_title("Non-convex loss")
for ax in axes:
    ax.set_xlabel("weight"); ax.set_ylabel("loss")
plt.tight_layout()
plt.savefig("convex_vs_nonconvex.png")
print("Saved plot: convex_vs_nonconvex.png")
print("Convex loss has 1 minimum; the bumpy one has multiple local minima gradient descent can get stuck in.")
```

---
⬅ [Day 10](day-10-linear-equations.md) · [Layer 2 Summary](layer-2-summary-review.md) · [Day 12 →](day-12-inequalities-and-geometric-interpretation.md)
