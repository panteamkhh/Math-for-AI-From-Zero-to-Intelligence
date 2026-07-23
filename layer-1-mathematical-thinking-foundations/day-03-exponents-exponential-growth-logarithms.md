# Layer 1 — Day 3: Exponents, Exponential Growth, Logarithms, Scale of Growth

**Keywords:** Exponents · Exponential Growth · Logarithms · Scale of Growth

---

### 🎯 Intuition
Exponential growth means something whose *rate* of growth also grows with it (like a virus where each infected person infects several more). The logarithm is the reverse tool — it "shrinks" large numbers down so they become comparable.

### 💡 ML/AI Application
Loss functions like Cross-Entropy use logarithms because probabilities live between 0 and 1, and the log turns them into an additive scale. Learning rate is also usually searched on a logarithmic scale (0.1, 0.01, 0.001) because its effect on training is exponential, not linear.

### 📝 Mental Exercise
Why do you think we search learning-rate values on a logarithmic scale instead of a linear one when tuning hyperparameters?

### 🧪 Hands-on
```python
import numpy as np

# Linear search: wastes almost all trials in a range that barely matters
linear_lrs = np.linspace(0.0001, 0.1, 5)
print("Linear search:", linear_lrs)

# Logarithmic search: spreads trials evenly across *orders of magnitude*
log_lrs = np.logspace(-4, -1, 5)  # 10^-4 to 10^-1
print("Log search:   ", log_lrs)

# Notice: the linear search clusters near 0.1 and barely samples 0.0001-0.001,
# even though that range often matters just as much for training stability.
```

---
⬅ [Day 2](day-02-absolute-value-distance-intervals.md) · [Layer 1 Summary](layer-1-summary-review.md) · [Day 4 →](day-04-patterns-sequences-abstraction.md)
