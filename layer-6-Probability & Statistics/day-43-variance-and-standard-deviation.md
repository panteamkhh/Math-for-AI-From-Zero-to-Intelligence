# Layer 6 — Day 43: Variance & Standard Deviation

**Keywords:** Variance · Standard Deviation

---

### 🎯 Intuition
Variance shows how spread out data is — far or close to the mean. Standard deviation is the same idea, but rescaled back to the data's original units (since variance is squared).

### 💡 ML/AI Application
The Bias-Variance Tradeoff is one of the most important concepts in ML: a high-variance model is sensitive to noise in the data (overfitting), while a high-bias model can't learn the pattern properly (underfitting). Understanding variance means understanding why this tradeoff exists.

### 📝 Mental Exercise
A model that performs very well on training data but poorly on test data — does it suffer more from high bias, or high variance?

### 🧪 Hands-on
```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

np.random.seed(0)
x = np.linspace(0, 1, 40).reshape(-1, 1)
y = np.sin(2 * np.pi * x).ravel() + np.random.normal(0, 0.2, 40)

x_train, x_test = x[:30], x[30:]
y_train, y_test = y[:30], y[30:]

# A very deep tree: low bias, but HIGH VARIANCE (memorizes training noise)
overfit_model = DecisionTreeRegressor(max_depth=None).fit(x_train, y_train)

train_error_overfit = np.mean((overfit_model.predict(x_train) - y_train) ** 2)
test_error_overfit = np.mean((overfit_model.predict(x_test) - y_test) ** 2)

print(f"Overfit model - train error: {train_error_overfit:.4f}, test error: {test_error_overfit:.4f}")
# A big gap (great on train, much worse on test) is the signature of HIGH VARIANCE,
# not high bias -- high bias would show poor performance on BOTH train and test.
```

---
⬅ [Day 42](day-42-random-variable-and-expected-value.md) · [Layer 6 Summary](layer-6-summary-review.md) · [Day 44 →](day-44-normal-distribution-and-sampling.md)
