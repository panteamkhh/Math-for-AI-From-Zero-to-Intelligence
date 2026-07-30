# Layer 3 — Day 25: Vector, Vector Length, Angle, Projection, Embedding Space Intuition

**Keywords:** Vector · Vector Length · Angle · Projection · Embedding Space Intuition

---

### 🎯 Intuition
A vector is something that has both a size and a direction — like an arrow. Its length is its distance from the origin, and the angle between two vectors shows how "aligned" they are. A projection is the "shadow" one vector casts onto another vector's direction.

### 💡 ML/AI Application
This is the single most important concept for understanding embeddings: every word, image, or item in a modern model becomes a vector. The angle between two vectors (cosine similarity) is the primary measure of semantic similarity in search, recommenders, and LLMs.

### 📝 Mental Exercise
Why do we usually use the angle between two embeddings (cosine similarity) to compare their similarity, rather than just Euclidean distance?

### 🧪 Hands-on
```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Same "direction" (meaning), but very different magnitude (e.g. document length/intensity)
short_doc = np.array([1.0, 2.0, 0.5])
long_doc  = np.array([10.0, 20.0, 5.0])   # same direction, scaled up 10x
unrelated_doc = np.array([0.5, -2.0, 3.0])

print("Euclidean distance (short vs long):", np.linalg.norm(short_doc - long_doc))
print("Cosine similarity  (short vs long):", cosine_similarity(short_doc, long_doc))
print("Cosine similarity  (short vs unrelated):", cosine_similarity(short_doc, unrelated_doc))

# Notice: Euclidean distance says short_doc and long_doc are "far apart" (because of scale/length),
# but cosine similarity correctly shows they point in the SAME direction (~1.0) --
# i.e., they mean the same thing, just expressed with different intensity/length.
```

---
⬅ [Day 24](day-24-circle-and-parabola.md) · [Layer 3 Summary](layer-3-summary-review.md)

**Next layer:** Layer 4 — Calculus & Change →
