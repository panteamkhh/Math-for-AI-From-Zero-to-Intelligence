# Layer 8 — Day 67: MAP (Maximum A Posteriori)

**Keywords:** MAP (Maximum A Posteriori)

---

### 🎯 Intuition
MAP is like MLE, but it also adds a prior belief — instead of relying only on the data, we also factor in what we already believed beforehand.

### 💡 ML/AI Application
Regularization (like L2/Ridge) is, from a Bayesian perspective, equivalent to adding a prior on the weights (that small weights are more likely) — meaning regularization is really a form of MAP estimation.

### 📝 Mental Exercise
Why can L2 regularization be interpreted as "assuming a zero-mean normal prior on the weights"?

### 🧪 Hands-on
```python
import numpy as np
from scipy.optimize import minimize

np.random.seed(0)
X = np.random.randn(30, 5)
true_w = np.array([2, -1, 0, 0.5, 0])
y = X @ true_w + np.random.normal(0, 0.5, 30)

def neg_log_likelihood(w):   # MLE objective: fit the data only
    residuals = y - X @ w
    return np.sum(residuals ** 2)

def neg_log_posterior(w, prior_strength=1.0):  # MAP objective: fit data + prior
    residuals = y - X @ w
    data_term = np.sum(residuals ** 2)
    # This L2 penalty term IS the log of a zero-mean Gaussian prior on w --
    # bigger weights become "less believable" under that prior.
    prior_term = prior_strength * np.sum(w ** 2)
    return data_term + prior_term

w0 = np.zeros(5)
mle_result = minimize(neg_log_likelihood, w0).x
map_result = minimize(neg_log_posterior, w0, args=(2.0,)).x

print("True weights:", true_w)
print("MLE estimate (no prior):", np.round(mle_result, 3))
print("MAP estimate (with L2 prior):", np.round(map_result, 3))
# The MAP estimate is pulled toward zero compared to MLE -- that pull IS the prior
# in action, and it's mathematically identical to what L2 regularization does.
```

---
⬅ [Day 66](day-66-mle.md) · [Layer 8 Summary](layer-8-summary-review.md) · [Day 68 →](day-68-chain-rule.md)
