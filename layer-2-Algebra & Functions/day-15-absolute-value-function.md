# Layer 2 — Day 15: Absolute Value Function

**Keywords:** Absolute Value Function

---

### 🎯 Intuition
The absolute value function shows distance from zero, and its graph has a "V" shape — a sharp point at zero where the derivative is undefined.

### 💡 ML/AI Application
Mean Absolute Error (MAE), used as a loss function, relies on absolute value and is more robust to outliers than MSE (since it doesn't square large errors). L1 regularization uses the same idea and leads to sparsity.

### 📝 Mental Exercise
Why does L1 regularization tend to push some weights to exactly zero, while L2 doesn't?

### 💬 Worked Answer
Because **L1 uses the absolute value of the weights**, and its shape has a sharp point near zero; this makes it easy for the optimization process to push some weights all the way to **exact zero**.

But **L2** uses squared weights, creating a smooth taper — it shrinks weights but usually doesn't drive them fully to zero.

So:
- **L1 → some weights = 0 → removes low-importance features → Sparsity**
- **L2 → all weights get smaller → prevents the model from growing too large** 🎯

In plain terms: L1 says "this feature is probably worthless, drop it," while L2 says "keep every feature a little bit in check."

### 🧪 Hands-on
```python
from sklearn.linear_model import Lasso, Ridge
import numpy as np

np.random.seed(0)
X = np.random.randn(100, 10)
true_w = np.array([5, 0, 0, 3, 0, 0, 0, -2, 0, 0])  # only 3 features actually matter
y = X @ true_w + np.random.randn(100) * 0.5

lasso = Lasso(alpha=0.5).fit(X, y)   # L1
ridge = Ridge(alpha=0.5).fit(X, y)   # L2

print("True weights:  ", true_w)
print("Lasso (L1) weights:", np.round(lasso.coef_, 2))
print("Ridge (L2) weights:", np.round(ridge.coef_, 2))
# Lasso should zero out several irrelevant features exactly.
# Ridge shrinks everything a bit but rarely hits exact zero.
```

---
⬅ [Day 14](day-14-quadratic-function.md) · [Layer 2 Summary](layer-2-summary-review.md) · [Day 16 →](day-16-exponential-function.md)
