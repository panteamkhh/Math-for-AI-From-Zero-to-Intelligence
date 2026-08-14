# Layer 8 — Day 56: Loss Function & Optimization

**Keywords:** Loss Function · Optimization

---

### 🎯 Intuition
A loss function is a single number that says how "badly" a model performed. Optimization is the process of searching for the parameters that minimize that number.

### 💡 ML/AI Application
Choosing the right loss function (MSE for regression, Cross-Entropy for classification) directly shapes how the model learns — it's the first and most important design decision in any ML project.

### 📝 Mental Exercise
Why can't you directly use MSE instead of Cross-Entropy for a classification problem?

### 🧪 Hands-on
```python
import numpy as np

y_true = np.array([1, 0, 1])          # true class labels
y_pred_probs = np.array([0.9, 0.2, 0.6])  # model's predicted probabilities

mse = np.mean((y_true - y_pred_probs) ** 2)
cross_entropy = -np.mean(y_true * np.log(y_pred_probs) + (1 - y_true) * np.log(1 - y_pred_probs))

print(f"MSE loss:           {mse:.4f}")
print(f"Cross-Entropy loss: {cross_entropy:.4f}")

# Try a confidently WRONG prediction and see how each loss reacts
y_pred_probs_bad = np.array([0.9, 0.2, 0.01])  # 3rd prediction is confidently wrong
mse_bad = np.mean((y_true - y_pred_probs_bad) ** 2)
ce_bad = -np.mean(y_true * np.log(y_pred_probs_bad) + (1 - y_true) * np.log(1 - y_pred_probs_bad))
print(f"\nWith one confidently wrong prediction:")
print(f"MSE loss:           {mse_bad:.4f}  (barely changes)")
print(f"Cross-Entropy loss: {ce_bad:.4f}  (explodes)")
# MSE treats probability errors gently and gives weak gradients near 0/1.
# Cross-Entropy punishes confident wrong answers MUCH more sharply -- exactly
# the signal you want when training a classifier.
```

---
⬅ [Layer 8 Summary](layer-8-summary-review.md) · [Day 57 →](day-57-gradient-descent-and-learning-rate.md)
