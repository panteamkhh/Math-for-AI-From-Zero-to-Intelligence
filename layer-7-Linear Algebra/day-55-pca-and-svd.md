# Layer 7 — Day 55: PCA & SVD

**Keywords:** PCA · SVD

---

### 🎯 Intuition
PCA rotates the data space to find the directions with the most spread (information) and discards the rest. SVD is a more powerful generalization that can decompose any matrix (not just square ones).

### 💡 ML/AI Application
PCA is used for dimensionality reduction and visualizing high-dimensional datasets. SVD underlies classic recommender systems (matrix factorization), image compression, and even older NLP algorithms like LSA.

### 📝 Mental Exercise
If you want to reduce a 50-dimensional dataset down to 2 dimensions for visualization, why does PCA do this with the least possible information loss?

### 🧪 Hands-on
```python
import numpy as np
from sklearn.decomposition import PCA

np.random.seed(0)

# A 50-dimensional dataset that actually only has 3 "true" underlying sources of variation
true_sources = np.random.randn(200, 3)
mixing = np.random.randn(3, 50)
high_dim_data = true_sources @ mixing + np.random.normal(0, 0.05, (200, 50))

pca = PCA(n_components=2).fit(high_dim_data)
reduced_data = pca.transform(high_dim_data)

print("Original shape:", high_dim_data.shape)
print("Reduced shape:  ", reduced_data.shape)
print("Variance explained by each of the 2 kept components:", np.round(pca.explained_variance_ratio_, 3))
print("Total variance retained:", f"{pca.explained_variance_ratio_.sum()*100:.1f}%")

# PCA picks the 2 directions (out of all possible directions) with the LARGEST
# eigenvalues -- i.e. the directions holding the most variance/information.
# Any other choice of 2 directions would necessarily capture LESS of the
# data's total spread, so PCA's choice is mathematically the least lossy one.
```

---
⬅ [Day 54](day-54-eigenvalue-and-eigenvector.md) · [Layer 7 Summary](layer-7-summary-review.md)

**Next layer:** Layer 8 — Mathematics of Modern AI →
