# Layer 8 — Day 78: Generalization

**Keywords:** Generalization

---

### 🎯 Intuition
Generalization is a model's ability to perform well on data it has never seen — not just memorizing the training data. This is the final tie-together of everything on this path: from bias-variance, to regularization, to scaling.

### 💡 ML/AI Application
The ultimate goal of every machine learning model isn't high accuracy on training data — it's good generalization to the real world; every concept you've seen across these 78 days (loss, regularization, bias-variance, scaling) is ultimately a tool in service of this one goal.

### 📝 Mental Exercise
Looking back across the whole 78-day path, which concept do you think will have the biggest impact on your ability to recognize and fix generalization problems in a real model?

### 🧪 Hands-on
```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge

np.random.seed(0)
x = np.linspace(0, 1, 100).reshape(-1, 1)
y = np.sin(2 * np.pi * x).ravel() + np.random.normal(0, 0.2, 100)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# Model A: memorizes training data (low bias, high variance -> poor generalization)
memorizer = DecisionTreeRegressor(max_depth=None).fit(x_train, y_train)

# Model B: regularized, simpler decision boundary (better bias-variance balance)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
generalizer = make_pipeline(PolynomialFeatures(degree=5), Ridge(alpha=1.0)).fit(x_train, y_train)

for name, model in [("Memorizing tree", memorizer), ("Regularized model", generalizer)]:
    train_error = np.mean((model.predict(x_train) - y_train) ** 2)
    test_error = np.mean((model.predict(x_test) - y_test) ** 2)
    print(f"{name:20s} -> train MSE: {train_error:.4f}, test MSE: {test_error:.4f}, "
          f"gap: {test_error - train_error:.4f}")

# The SMALLER the gap between train and test error, the BETTER the generalization.
# Every concept from this 78-day path -- bias/variance, regularization, proper
# loss functions, scaling laws -- is ultimately a lever for closing this gap.
```

---

### Generalization isn't the finish line

Everything from here connects to two more practical, modern lessons: how we actually fight overfitting in deep nets day-to-day (Dropout & Normalization), and how a base model becomes a helpful, aligned assistant (RLHF).

⬅ [Day 77](day-77-scaling-laws.md) · [Layer 8 Summary](layer-8-summary-review.md) · [Day 79 →](day-79-dropout-and-normalization.md)
