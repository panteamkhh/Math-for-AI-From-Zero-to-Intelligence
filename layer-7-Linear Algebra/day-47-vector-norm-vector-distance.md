# Layer 7 — Day 47: Vector, Norm, Vector Distance

**Keywords:** Vector · Norm · Vector Distance

---

### 🎯 Intuition
A norm is the "length" of a vector — a generalization of absolute value to multi-dimensional spaces. Different types of norms (L1, L2) are different ways of measuring that length.

### 💡 ML/AI Application
The L2 norm underlies ridge regularization, distance calculations in KNN, and normalizing embeddings. The L1 norm underlies lasso regularization and sparsity.

### 📝 Mental Exercise
Why does the L1 norm of a weight vector tend to produce sparsity (some weights becoming exactly zero), while the L2 norm doesn't?

### 🧪 Hands-on
```python
import numpy as np

w = np.array([3.0, -4.0, 0.1, 8.0])

l1_norm = np.sum(np.abs(w))          # sum of absolute values
l2_norm = np.sqrt(np.sum(w ** 2))    # Euclidean length

print(f"Vector: {w}")
print(f"L1 norm: {l1_norm}")
print(f"L2 norm: {l2_norm:.3f}")

# Why L1 creates sparsity and L2 doesn't:
# L1's penalty grows LINEARLY (constant "pressure" toward zero, even for tiny weights),
# so optimization can push a small weight all the way to exactly 0.
# L2's penalty grows QUADRATICALLY -- the closer a weight gets to 0, the weaker
# the pressure becomes, so it shrinks weights smoothly but rarely hits exact zero.
for weight in [2.0, 0.5, 0.05]:
    print(f"weight={weight:5} -> L1 pressure (constant): 1.0,  L2 pressure (shrinks): {2*weight:.3f}")
```

---
⬅ [Layer 7 Summary](layer-7-summary-review.md) · [Day 48 →](day-48-dot-product-and-cosine-similarity.md)
