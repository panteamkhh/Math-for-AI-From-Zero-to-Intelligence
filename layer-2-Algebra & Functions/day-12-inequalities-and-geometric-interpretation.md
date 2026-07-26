# Layer 2 — Day 12: Inequalities and the Geometric Interpretation of Solutions

**Keywords:** Inequalities · Geometric Interpretation of Solutions

---

### 🎯 Intuition
Instead of one exact point, an inequality describes a "region" of possible solutions. The geometric interpretation is visualizing that region on a plane or in space.

### 💡 ML/AI Application
A decision boundary in classification is exactly an inequality: the model draws a boundary and says every point on one side of it belongs to a certain class. SVMs are built directly on this idea of geometric region-splitting.

### 📝 Mental Exercise
Picture a two-class classifier — why can its decision boundary be written as an inequality?

### 💬 Worked Answer
Because the classifier decides which class a data point belongs to by comparing a **score** against a threshold.

For example:
- if `w·x + b > 0` → class 1
- if `w·x + b < 0` → class 0

The line/plane `w·x + b = 0` is exactly the **decision boundary**, splitting the space into two regions.

### 🧪 Hands-on
```python
import numpy as np

# A simple linear classifier: score = w . x + b
w = np.array([2.0, -1.0])
b = -1.0

def classify(x):
    score = w @ x + b
    return 1 if score > 0 else 0

points = [np.array([0, 0]), np.array([2, 1]), np.array([0, 3]), np.array([5, 0])]
for p in points:
    score = w @ p + b
    print(f"point={p}, score={score:.2f} -> class {classify(p)}")

# The boundary w.x + b = 0 is exactly the inequality's dividing line between the two classes.
```

---
⬅ [Day 11](day-11-quadratic-equations.md) · [Layer 2 Summary](layer-2-summary-review.md) · [Day 13 →](day-13-linear-function.md)
