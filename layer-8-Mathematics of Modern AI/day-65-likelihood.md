# Layer 8 — Day 65: Likelihood

**Keywords:** Likelihood

---

### 🎯 Intuition
Likelihood asks: "how probable is it that these specific parameters produced the data we observed?" Unlike probability, which goes from parameter to data, likelihood looks from data back to parameter.

### 💡 ML/AI Application
Many loss functions (including cross-entropy) are actually derived from the "maximum likelihood" principle: we train a model to find the parameters that make the observed data "most likely."

### 📝 Mental Exercise
Why do we say that training a classification model with cross-entropy is mathematically equivalent to maximizing the likelihood of the data?

### 🧪 Hands-on
```python
import numpy as np

y_true = np.array([1, 0, 1, 1])
y_pred_probs = np.array([0.8, 0.3, 0.6, 0.9])  # model's P(class=1) for each sample

# Likelihood of the observed labels, given these predicted probabilities
likelihood_per_sample = np.where(y_true == 1, y_pred_probs, 1 - y_pred_probs)
total_likelihood = np.prod(likelihood_per_sample)

# Cross-entropy loss (negative log likelihood, averaged)
cross_entropy = -np.mean(np.log(likelihood_per_sample))

print("Per-sample likelihood:", np.round(likelihood_per_sample, 3))
print("Total likelihood (product):", round(total_likelihood, 5))
print("Cross-entropy loss (-mean log likelihood):", round(cross_entropy, 4))
# MINIMIZING cross-entropy is mathematically the SAME as MAXIMIZING likelihood --
# cross-entropy is literally the negative log of the likelihood. Taking the log
# turns the awkward product of many small probabilities into a sum, which is
# far more numerically stable and easier to optimize.
```

---
⬅ [Day 64](day-64-hessian.md) · [Layer 8 Summary](layer-8-summary-review.md) · [Day 66 →](day-66-mle.md)
