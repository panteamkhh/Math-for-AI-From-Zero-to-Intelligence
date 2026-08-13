# Layer 7 — Day 54: Eigenvalue & Eigenvector

**Keywords:** Eigenvalue · Eigenvector

---

### 🎯 Intuition
An eigenvector is a special vector that, when a linear transformation is applied to it, only changes in length — never in direction. The eigenvalue is exactly that amount of length change.

### 💡 ML/AI Application
This is exactly the mathematical core of PCA: the eigenvectors of a data's covariance matrix show the "principal directions of variability" in the data, and their eigenvalues show how much information (variance) each direction holds.

### 📝 Mental Exercise
Why is the direction with the largest eigenvalue chosen in PCA as the "most important principal component"?

### 🧪 Hands-on
```python
import numpy as np

np.random.seed(0)

# Data that varies a LOT along one direction and only a little along another
data = np.random.randn(300, 2) @ np.array([[3, 0], [0, 0.3]])

cov_matrix = np.cov(data.T)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

print("Eigenvalues:", np.round(eigenvalues, 3))
print("Eigenvectors:\n", np.round(eigenvectors, 3))

# The eigenvalue tells you how much VARIANCE (information) lies along its
# matching eigenvector's direction. The largest eigenvalue's direction captures
# the most spread in the data -- so keeping that direction (and dropping the
# smaller one) loses the LEAST information possible. That's why PCA always
# ranks and keeps components by eigenvalue, largest first.
top_direction = eigenvectors[:, np.argmax(eigenvalues)]
print("\nMost important direction (top eigenvector):", np.round(top_direction, 3))
```

---
⬅ [Day 53](day-53-determinant.md) · [Layer 7 Summary](layer-7-summary-review.md) · [Day 55 →](day-55-pca-and-svd.md)
