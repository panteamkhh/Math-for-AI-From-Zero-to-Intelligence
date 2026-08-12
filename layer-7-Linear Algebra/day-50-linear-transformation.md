# Layer 7 — Day 50: Linear Transformation

**Keywords:** Linear Transformation

---

### 🎯 Intuition
A linear transformation is a function that rotates, stretches, or compresses space — but keeps straight lines straight and leaves the origin fixed in place.

### 💡 ML/AI Application
Every Dense layer in a neural network (before the activation function) is a linear transformation that takes the input space to a different space with different dimensions and orientation — this is the foundation of representation learning.

### 📝 Mental Exercise
When we say a neural network layer "transforms the feature space," what does that mean geometrically?

### 🧪 Hands-on
```python
import numpy as np

# A "Dense layer" is just: output = input @ W (a linear transformation)
W = np.array([[2, 0], [0, 0.5]])  # stretches x-axis by 2x, squishes y-axis by 0.5x

points = np.array([
    [1, 1],
    [2, 0],
    [0, 3],
])

transformed = points @ W

print("Original points:\n", points)
print("\nTransformed points (after linear transformation):\n", transformed)

# Geometrically: every point moved, but straight lines are still straight, and the
# origin [0,0] stayed exactly at [0,0]. A Dense layer does this in much higher
# dimensions -- reshaping the "cloud" of data points into a new space where
# patterns that were hard to separate before might become much easier to separate.
```

---
⬅ [Day 49](day-49-matrix-and-matrix-multiplication.md) · [Layer 7 Summary](layer-7-summary-review.md) · [Day 51 →](day-51-vector-space-and-orthogonality.md)
