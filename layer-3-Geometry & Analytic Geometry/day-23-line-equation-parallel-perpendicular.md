# Layer 3 — Day 23: Line Equation, Parallel & Perpendicular Lines

**Keywords:** Line Equation · Parallel Lines · Perpendicular Lines

---

### 🎯 Intuition
A line's equation describes the relationship between x and y. Parallel lines never meet (same slope); perpendicular lines cross at exactly 90 degrees.

### 💡 ML/AI Application
In the geometric view of linear regression, the fitted line is exactly this line equation. Perpendicularity between vectors is also a key concept in PCA and in orthogonalizing features.

### 📝 Mental Exercise
Why must the principal components in PCA be perpendicular (orthogonal) to each other?

### 🧪 Hands-on
```python
import numpy as np
from sklearn.decomposition import PCA

np.random.seed(0)
X = np.random.randn(200, 4) @ np.random.randn(4, 4)  # correlated features

pca = PCA(n_components=4).fit(X)
components = pca.components_

# Check that every pair of principal components is orthogonal (dot product ~ 0)
for i in range(4):
    for j in range(i + 1, 4):
        dot = np.dot(components[i], components[j])
        print(f"PC{i+1} . PC{j+1} = {dot:.6f}  (should be ~0)")

# Why it matters: orthogonal components each capture a DIFFERENT, non-overlapping
# direction of variance in the data. If they weren't orthogonal, two components
# could partially describe the same information -- redundant, not efficient.
```

---
⬅ [Day 22](day-22-midpoint-and-slope.md) · [Layer 3 Summary](layer-3-summary-review.md) · [Day 24 →](day-24-circle-and-parabola.md)
