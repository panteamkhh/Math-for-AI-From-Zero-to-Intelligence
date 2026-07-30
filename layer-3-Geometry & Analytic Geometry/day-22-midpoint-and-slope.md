# Layer 3 — Day 22: Midpoint & Slope

**Keywords:** Midpoint · Slope

---

### 🎯 Intuition
The midpoint is exactly halfway between two points. Slope shows how "steeply" a line rises or falls.

### 💡 ML/AI Application
Slope is the same concept that later becomes the "derivative" and the "gradient" — the rate of change of an output with respect to an input, which is the core engine of learning in gradient descent.

### 📝 Mental Exercise
If increasing one feature causes the model's prediction to change very quickly, what does that tell you about the "slope" of the model's function with respect to that feature?

### 🧪 Hands-on
```python
import numpy as np

# Two points on the model's prediction curve for a single feature
x1, y1 = 10, 50   # feature=10 -> prediction=50
x2, y2 = 11, 95   # feature=11 -> prediction=95 (a tiny feature change, huge output jump)

slope = (y2 - y1) / (x2 - x1)
midpoint = ((x1 + x2) / 2, (y1 + y2) / 2)

print(f"Slope: {slope}")
print(f"Midpoint: {midpoint}")

# A large slope like this means the model is HIGHLY sensitive to this feature --
# small real-world changes could swing the prediction a lot. This is exactly the
# intuition behind a large gradient: the model's output is very reactive to that input.
```

---
⬅ [Day 21](day-21-coordinate-system-and-distance.md) · [Layer 3 Summary](layer-3-summary-review.md) · [Day 23 →](day-23-line-equation-parallel-perpendicular.md)
