# Layer 3 — Day 24: Circle & Parabola

**Keywords:** Circle · Parabola

---

### 🎯 Intuition
A circle is the set of points that are all the same fixed distance from a center. A parabola is a bowl-like shape that comes from a degree-2 (quadratic) function.

### 💡 ML/AI Application
The contour lines (level curves) of a loss function are often seen as ellipses or circles; the shape of these contours shows how fast and in what path gradient descent reaches the minimum.

### 📝 Mental Exercise
If a loss function's contours are very stretched out (narrow ellipses) instead of circular, why does gradient descent converge more slowly and unstably?

### 🧪 Hands-on
```python
import numpy as np

def gradient_descent(grad_fn, start, lr, steps=40):
    w = np.array(start, dtype=float)
    path = [w.copy()]
    for _ in range(steps):
        w = w - lr * grad_fn(w)
        path.append(w.copy())
    return np.array(path)

# Circular contours: loss = w1^2 + w2^2  (well-conditioned)
circular_grad = lambda w: np.array([2*w[0], 2*w[1]])

# Elongated elliptical contours: loss = w1^2 + 50*w2^2 (poorly-conditioned)
stretched_grad = lambda w: np.array([2*w[0], 100*w[1]])

path_circular = gradient_descent(circular_grad, [5, 5], lr=0.1)
path_stretched = gradient_descent(stretched_grad, [5, 5], lr=0.1)

print("Circular contours - final point:", path_circular[-1], " (should be near [0,0])")
print("Stretched contours - final point:", path_stretched[-1], " (often overshoots / oscillates)")
# With stretched contours, a step size that's stable for w1 is way too large for w2 (or vice versa),
# so gradient descent zig-zags and converges much more slowly.
```

---
⬅ [Day 23](day-23-line-equation-parallel-perpendicular.md) · [Layer 3 Summary](layer-3-summary-review.md) · [Day 25 →](day-25-vector-length-angle-projection-embedding-space.md)
