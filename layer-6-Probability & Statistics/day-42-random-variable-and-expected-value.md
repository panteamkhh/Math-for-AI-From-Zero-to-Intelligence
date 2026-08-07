# Layer 6 — Day 42: Random Variable & Expected Value

**Keywords:** Random Variable · Expected Value

---

### 🎯 Intuition
A random variable is a number whose outcome isn't certain but is determined by chance. Expected value is a "probability-weighted average" — what you'd expect to get on average over the long run.

### 💡 ML/AI Application
The loss function we minimize during training is usually the "expected error over the data distribution" — the average error across all possible samples, not just the current dataset.

### 📝 Mental Exercise
Why is "our goal is to minimize expected loss" different from "minimize loss only on the training data"?

### 🧪 Hands-on
```python
import numpy as np

np.random.seed(0)

# Simulate a random variable: prediction error, which varies by chance
# depending on which sample from the TRUE underlying distribution we happen to draw
true_distribution_errors = np.random.normal(loc=2.0, scale=1.5, size=100_000)

training_sample = true_distribution_errors[:50]  # our small observed training set

print(f"Expected loss (true distribution, 100k samples): {true_distribution_errors.mean():.3f}")
print(f"Loss measured on training sample only (50 samples): {training_sample.mean():.3f}")
# The training-only estimate is noisy and can be optimistic or pessimistic purely by
# chance -- it's an ESTIMATE of the expected loss, not the expected loss itself.
# A model that only minimizes error on its small training sample can overfit to
# that sample's particular noise, rather than the true underlying distribution.
```

---
⬅ [Day 41](day-41-conditional-probability-and-bayes-theorem.md) · [Layer 6 Summary](layer-6-summary-review.md) · [Day 43 →](day-43-variance-and-standard-deviation.md)
