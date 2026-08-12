# Layer 7 — Day 51: Vector Space & Orthogonality

**Keywords:** Vector Space · Orthogonality

---

### 🎯 Intuition
A vector space is the set of all possible vectors under the rules of addition and scalar multiplication. Two orthogonal vectors are completely "unrelated" to each other — neither casts any shadow onto the other.

### 💡 ML/AI Application
When features are orthogonal (uncorrelated), a model can more easily learn the effect of each one separately. Multicollinearity (dependency between features) is exactly the problem of missing orthogonality, and it makes linear regression unstable.

### 📝 Mental Exercise
Why can having two nearly identical features (like "height in centimeters" and "height in meters") cause problems in linear regression?

### 🧪 Hands-on
```python
import numpy as np
from sklearn.linear_model import LinearRegression

np.random.seed(0)
n = 100
height_cm = np.random.normal(170, 10, n)
height_m = height_cm / 100          # near-perfectly correlated with height_cm!
noise_feature = np.random.normal(0, 1, n)

y = 0.5 * height_cm + np.random.normal(0, 2, n)

# Fit with both redundant features included
X_redundant = np.column_stack([height_cm, height_m, noise_feature])
model = LinearRegression().fit(X_redundant, y)
print("Coefficients with redundant features:", np.round(model.coef_, 3))

# Fit again with a TINY change in the data (simulating resampling/new data)
height_cm_v2 = height_cm + np.random.normal(0, 0.01, n)  # almost no change
height_m_v2 = height_cm_v2 / 100
X_redundant_v2 = np.column_stack([height_cm_v2, height_m_v2, noise_feature])
model_v2 = LinearRegression().fit(X_redundant_v2, y)
print("Coefficients after a tiny data change:", np.round(model_v2.coef_, 3))
# Notice how wildly the individual coefficients for height_cm and height_m can swing,
# even though the PREDICTIONS barely change -- that instability is the signature of
# multicollinearity (the opposite of orthogonal features).
```

---
⬅ [Day 50](day-50-linear-transformation.md) · [Layer 7 Summary](layer-7-summary-review.md) · [Day 52 →](day-52-basis-and-rank.md)
