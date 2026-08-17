# Layer 8 — Day 66: MLE (Maximum Likelihood Estimation)

**Keywords:** MLE (Maximum Likelihood Estimation)

---

### 🎯 Intuition
MLE is a method for finding the best parameters of a model: the parameters that maximize the probability of observing the real data.

### 💡 ML/AI Application
Most statistical models, and many ML models (linear regression, logistic regression), are ultimately justified via MLE — minimizing their loss is mathematically equivalent to maximizing likelihood.

### 📝 Mental Exercise
Why is minimizing MSE in linear regression mathematically equivalent to maximizing likelihood under the assumption of normally-distributed noise?

### 🧪 Hands-on
```python
import numpy as np
from scipy.stats import norm

np.random.seed(0)
true_w, true_b = 2.5, 1.0
x = np.linspace(0, 10, 50)
noise = np.random.normal(0, 1.0, 50)   # normally distributed noise, by assumption
y = true_w * x + true_b + noise

def neg_log_likelihood(params, x, y, sigma=1.0):
    w, b = params
    preds = w * x + b
    # Under a normal-noise assumption, the log-likelihood of each residual
    # is directly tied to its SQUARED error -- this is where MSE comes from.
    return -np.sum(norm.logpdf(y, loc=preds, scale=sigma))

def mse(params, x, y):
    w, b = params
    preds = w * x + b
    return np.mean((y - preds) ** 2)

test_params = [(2.5, 1.0), (2.0, 1.0), (3.0, 0.5)]
for params in test_params:
    nll = neg_log_likelihood(params, x, y)
    m = mse(params, x, y)
    print(f"params={params} -> neg-log-likelihood={nll:.2f}, MSE={m:.4f}")
# Notice: whichever params minimize MSE ALSO minimize the negative log-likelihood --
# under Gaussian noise, minimizing MSE and maximizing likelihood are the same optimization.
```

---
⬅ [Day 65](day-65-likelihood.md) · [Layer 8 Summary](layer-8-summary-review.md) · [Day 67 →](day-67-map.md)
