# Layer 1 — Day 4: Patterns, Sequences, Abstraction

**Keywords:** Patterns · Sequences · Abstraction

---

### 🎯 Intuition
Finding a pattern means finding the rule that generated a sequence of data. Abstraction means ignoring irrelevant details and focusing on the underlying structure — exactly what a machine learning model tries to do.

### 💡 ML/AI Application
The entire goal of machine learning is this: extract a general function (pattern) from observed data (sequence) that also works on new, unseen data. Overfitting is when a model memorizes the noise in the details instead of learning the general pattern.

### 📝 Mental Exercise
How would you describe an overfit model if you had to use the language of "pattern vs. memorization"?

### 🧪 Hands-on
```python
import numpy as np

np.random.seed(0)
x = np.linspace(0, 1, 15)
true_pattern = np.sin(2 * np.pi * x)
y = true_pattern + np.random.normal(0, 0.15, size=x.shape)  # noisy observations

# A low-degree fit captures the pattern
pattern_fit = np.poly1d(np.polyfit(x, y, deg=3))

# A high-degree fit memorizes the noise (classic overfitting)
memorized_fit = np.poly1d(np.polyfit(x, y, deg=14))

x_test = np.linspace(0, 1, 200)
print("Pattern fit error on new points:",
      np.mean((pattern_fit(x_test) - np.sin(2 * np.pi * x_test)) ** 2))
print("Memorized fit error on new points:",
      np.mean((memorized_fit(x_test) - np.sin(2 * np.pi * x_test)) ** 2))
# The degree-14 curve fits the 15 training points almost perfectly,
# but does far worse on new points — it learned noise, not the pattern.
```

---
⬅ [Day 3](day-03-exponents-exponential-growth-logarithms.md) · [Layer 1 Summary](layer-1-summary-review.md) · [Day 5 →](day-05-modeling-target-feature-thinking.md)
