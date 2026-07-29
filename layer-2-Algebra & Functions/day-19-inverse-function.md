# Layer 2 — Day 19: Inverse Function

**Keywords:** Inverse Function

---

### 🎯 Intuition
An inverse function is the return trip: if the original function takes an input to an output, its inverse takes that output back to the same input.

### 💡 ML/AI Application
When normalizing data (e.g., standardization), we need to be able to reverse the operation to bring the final prediction back to its original scale. Generative models like Normalizing Flows are also built directly on invertible functions.

### 📝 Mental Exercise
If you normalized data with the formula `(x - mean) / std`, what's the inverse formula to bring a prediction back to the original scale?

### 💬 Worked Answer
Normalization formula:
```
z = (x - mean) / std
```

To bring it back to the original scale:
```
x = z × std + mean
```

That is, first multiply the normalized value by the **standard deviation**, then add the **mean**.

### 🧪 Hands-on
```python
import numpy as np

x = np.array([10, 20, 30, 40, 50])
mean, std = x.mean(), x.std()

# Forward: normalize
z = (x - mean) / std
print("Normalized:", np.round(z, 3))

# Inverse: bring back to original scale
x_recovered = z * std + mean
print("Recovered:", np.round(x_recovered, 3))
print("Matches original?", np.allclose(x, x_recovered))

# Why this matters: if your model predicts in normalized (z) space,
# you MUST apply this inverse before reporting the prediction to a human.
```

---
⬅ [Day 18](day-18-function-composition.md) · [Layer 2 Summary](layer-2-summary-review.md) · [Day 20 →](day-20-modeling-with-functions.md)
