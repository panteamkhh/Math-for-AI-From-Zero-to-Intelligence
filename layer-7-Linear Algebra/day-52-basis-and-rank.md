# Layer 7 — Day 52: Basis & Rank

**Keywords:** Basis · Rank

---

### 🎯 Intuition
A basis is the smallest set of vectors that can build the entire space. Rank shows how many truly "independent" dimensions of information a matrix actually has.

### 💡 ML/AI Application
A low rank in a data matrix means many features are actually redundant — this is exactly what PCA and dimensionality reduction try to discover and remove.

### 📝 Mental Exercise
If a dataset has 100 columns but the rank of its matrix is only 20, what does that tell you about redundancy in the data?

### 🧪 Hands-on
```python
import numpy as np

np.random.seed(0)

# Build a dataset with 100 columns, but only 20 truly INDEPENDENT sources of information --
# the rest are just linear combinations (redundant copies/mixes) of those 20
independent_signals = np.random.randn(500, 20)
mixing_matrix = np.random.randn(20, 100)
redundant_data = independent_signals @ mixing_matrix   # 500 x 100, but "fake" complexity

rank = np.linalg.matrix_rank(redundant_data)
print(f"Data shape: {redundant_data.shape}")
print(f"Matrix rank: {rank}")
# Even though there are 100 columns, the rank reveals there are really only
# 20 independent directions of information -- the other 80 columns are just
# recombinations of the same underlying 20 signals. This is EXACTLY the kind
# of redundancy PCA is designed to detect and compress away.
```

---
⬅ [Day 51](day-51-vector-space-and-orthogonality.md) · [Layer 7 Summary](layer-7-summary-review.md) · [Day 53 →](day-53-determinant.md)
