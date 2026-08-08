# Layer 6 — Day 44: Normal Distribution & Sampling

**Keywords:** Normal Distribution · Sampling

---

### 🎯 Intuition
The normal distribution is the famous bell shape that many natural phenomena follow — most values cluster near the mean, and extreme values are rare.

### 💡 ML/AI Application
Many normalization techniques and weight initialization methods (like Xavier/He initialization) assume the data or weights are roughly normally distributed, in order to keep training stable.

### 📝 Mental Exercise
Why does initializing all of a network's weights to a single constant value (instead of sampling randomly from a distribution) cause a learning problem?

### 🧪 Hands-on
```python
import numpy as np

np.random.seed(0)

def forward_pass(x, W):
    return np.tanh(x @ W)

x = np.random.randn(1, 4)

# BAD: every weight initialized to the exact same constant
W_constant = np.full((4, 4), 0.5)
h_constant = forward_pass(x, W_constant)
print("All-constant weights -> hidden activations:", np.round(h_constant, 4))
# Every neuron receives IDENTICAL input combinations and produces IDENTICAL outputs --
# they can never learn to specialize/differentiate (this is the "symmetry problem").

# GOOD: weights sampled from a normal distribution (e.g. He/Xavier-style init)
W_normal = np.random.normal(0, 0.5, size=(4, 4))
h_normal = forward_pass(x, W_normal)
print("Randomly-initialized weights -> hidden activations:", np.round(h_normal, 4))
# Each neuron now sees a different combination and starts out distinct,
# so gradients push each one to learn something different.
```

---
⬅ [Day 43](day-43-variance-and-standard-deviation.md) · [Layer 6 Summary](layer-6-summary-review.md) · [Day 45 →](day-45-hypothesis-testing-and-correlation.md)
