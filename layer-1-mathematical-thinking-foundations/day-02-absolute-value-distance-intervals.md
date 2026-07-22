# Layer 1 — Day 2: Absolute Value, Distance, Intervals, Distance Thinking

**Keywords:** Absolute Value · Distance · Intervals · Distance Thinking

---

### 🎯 Intuition
Absolute value means "how far," regardless of direction. It's the simplest form of the concept of distance — one that later reappears in more sophisticated shapes like Euclidean or cosine distance in high-dimensional spaces.

### 💡 ML/AI Application
Almost every similarity-based algorithm (KNN, K-Means, recommenders, embedding search) ultimately computes some form of "distance" between two points. Understanding distance intuitively means understanding how a model defines "being similar."

### 📝 Mental Exercise
If you wanted to measure the similarity between two users based on age and income, how would you define the distance between them?

### 🧪 Hands-on
```python
import numpy as np

user_a = np.array([25, 45000])   # age, income
user_b = np.array([40, 52000])

# Naive distance: sum of absolute differences fails because scales differ wildly
naive_distance = np.sum(np.abs(user_a - user_b))
print("Naive (unscaled) distance:", naive_distance)

# Fix: normalize before computing distance — this is why feature scaling exists
mean = np.array([32, 48000])
std = np.array([8, 6000])

a_scaled = (user_a - mean) / std
b_scaled = (user_b - mean) / std

euclidean = np.linalg.norm(a_scaled - b_scaled)
print("Euclidean distance (scaled):", euclidean)
```

---
⬅ [Day 1](day-01-sets-number-types-categorization.md) · [Layer 1 Summary](layer-1-summary-review.md) · [Day 3 →](day-03-exponents-exponential-growth-logarithms.md)
