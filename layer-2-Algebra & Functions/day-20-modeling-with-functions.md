# Layer 2 — Day 20: Modeling with Functions

**Keywords:** Modeling with Functions

---

### 🎯 Intuition
This day is a wrap-up: any real-world phenomenon can be approximated by choosing the right type of function (linear, exponential, quadratic, composite); choosing the right function means correctly understanding how the data behaves.

### 💡 ML/AI Application
Choosing a model architecture (linear vs. deep network, decision tree vs. regression) is really choosing a "family of functions" that we believe best describes the true relationship between the feature and the target.

### 📝 Mental Exercise
If the relationship between a feature and the target in your data is clearly non-linear, but you use linear regression, what happens to the model's accuracy?

### 💬 Worked Answer
If the relationship between **Feature** and **Target** is non-linear, linear regression can't learn the data's true pattern.

As a result:
- The model **underfits** (it's too simple for the problem).
- Prediction error increases.
- Model accuracy drops.

For such data, you need models capable of learning more complex relationships, such as **Decision Tree**, **Random Forest**, or a **neural network**.

### 🧪 Hands-on
```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

np.random.seed(0)
x = np.linspace(-3, 3, 60).reshape(-1, 1)
y = x.ravel() ** 3 + np.random.normal(0, 1, size=x.shape[0])   # clearly non-linear relationship

linear_model = LinearRegression().fit(x, y)
tree_model = DecisionTreeRegressor(max_depth=4).fit(x, y)

linear_error = np.mean((linear_model.predict(x) - y) ** 2)
tree_error = np.mean((tree_model.predict(x) - y) ** 2)

print(f"Linear regression MSE: {linear_error:.2f}  <- underfits the cubic pattern")
print(f"Decision tree MSE:     {tree_error:.2f}  <- captures the non-linear shape much better")
```

---
⬅ [Day 19](day-19-inverse-function.md) · [Layer 2 Summary](layer-2-summary-review.md)

**Next layer:** Layer 3 — Geometry & Analytic Geometry →
