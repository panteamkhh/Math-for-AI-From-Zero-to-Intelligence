# Layer 6 — Day 46: Regression

**Keywords:** Regression

---

### 🎯 Intuition
From a statistical viewpoint, regression means fitting a relationship between variables that minimizes prediction error — the same thing we saw in the linear equations lesson, but now viewed through a probabilistic, uncertainty-aware lens.

### 💡 ML/AI Application
Linear and logistic regression are two of the most widely used baseline models in industry; understanding their statistical foundations (assumptions, confidence intervals, coefficient significance) lets you know when to trust a simple model and when to reach for something more complex.

### 📝 Mental Exercise
Why is it usually recommended to try a simple regression model as a baseline before jumping straight to a complex neural network?

### 🧪 Hands-on
```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import time

np.random.seed(0)
X = np.random.randn(300, 5)
y = 3*X[:, 0] - 2*X[:, 1] + 0.5*X[:, 2] + np.random.normal(0, 0.5, 300)  # mostly linear relationship

X_train, X_test = X[:240], X[240:]
y_train, y_test = y[:240], y[240:]

start = time.time()
baseline = LinearRegression().fit(X_train, y_train)
baseline_time = time.time() - start
baseline_mse = mean_squared_error(y_test, baseline.predict(X_test))

start = time.time()
neural_net = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=2000, random_state=0).fit(X_train, y_train)
nn_time = time.time() - start
nn_mse = mean_squared_error(y_test, neural_net.predict(X_test))

print(f"Linear regression -> MSE: {baseline_mse:.3f}, train time: {baseline_time:.4f}s")
print(f"Neural network    -> MSE: {nn_mse:.3f}, train time: {nn_time:.4f}s")
# When the true relationship is mostly linear, the simple baseline often matches
# (or beats) the neural network -- while training almost instantly and staying
# fully interpretable. That's exactly why you check the baseline FIRST.
```

---
⬅ [Day 45](day-45-hypothesis-testing-and-correlation.md) · [Layer 6 Summary](layer-6-summary-review.md)

**Next layer:** Layer 7 — Linear Algebra →
