# Layer 3 — Day 21: Coordinate System & Distance Between Two Points

**Keywords:** Coordinate System · Distance Between Two Points

---

### 🎯 Intuition
A coordinate system is an addressing system for locating any point in space. The distance between two points is calculated by generalizing the Pythagorean theorem.

### 💡 ML/AI Application
Any multi-dimensional numerical data point (like a word embedding with 300 dimensions) is really just a point in a large coordinate space. The Euclidean distance between two embeddings shows how similar two concepts are.

### 📝 Mental Exercise
If you have a user with features (age=25, income=50) and another with (age=30, income=55), how would you calculate the Euclidean distance between them?

### 💬 Worked Answer
Treat each user as a point:
- User 1: **(25, 50)**
- User 2: **(30, 55)**

Euclidean distance formula:
```
d = √[(x₂ - x₁)² + (y₂ - y₁)²]
```

Plugging in the numbers:
```
d = √[(30-25)² + (55-50)²] = √[5² + 5²] = √50 ≈ 7.07
```

✅ **Result:** the Euclidean distance between these two users is about **7.07**. The smaller this distance, the more similar the two users' features are. 📏

### 🧪 Hands-on
```python
import numpy as np

user1 = np.array([25, 50])
user2 = np.array([30, 55])

distance = np.linalg.norm(user1 - user2)
print(f"Euclidean distance: {distance:.2f}")

# Scaling up: the exact same idea works for a 300-dim word embedding —
# numpy doesn't care whether it's 2 numbers or 300.
embedding_a = np.random.randn(300)
embedding_b = np.random.randn(300)
emb_distance = np.linalg.norm(embedding_a - embedding_b)
print(f"Distance between two 300-dim embeddings: {emb_distance:.2f}")
```

---
⬅ [Layer 3 Summary](layer-3-summary-review.md) · [Day 22 →](day-22-midpoint-and-slope.md)
