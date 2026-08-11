# Layer 7 — Day 48: Dot Product & Cosine Similarity

**Keywords:** Dot Product · Cosine Similarity

---

### 🎯 Intuition
The dot product turns two vectors into a single number that reflects both how "aligned" they are and how large they are. Cosine similarity measures only the direction, ignoring length.

### 💡 ML/AI Application
This is the single most-used operation in all of modern AI: the attention mechanism in transformers, vector search, and recommender systems are all built on the dot product between embeddings.

### 📝 Mental Exercise
Why do we usually use cosine similarity — not the raw dot product — to compare the similarity of two text documents whose lengths can differ a lot?

### 🧪 Hands-on
```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

short_doc_vector = np.array([1.0, 2.0, 0.0])
long_doc_vector = np.array([10.0, 20.0, 0.0])   # same "direction" (topic), just longer/more repetitive
different_doc_vector = np.array([0.0, 1.0, 5.0])

print("Raw dot product (short, long):     ", np.dot(short_doc_vector, long_doc_vector))
print("Raw dot product (short, different):", np.dot(short_doc_vector, different_doc_vector))
print("Cosine similarity (short, long):     ", cosine_similarity(short_doc_vector, long_doc_vector))
print("Cosine similarity (short, different):", cosine_similarity(short_doc_vector, different_doc_vector))

# The raw dot product for (short, long) is huge just because "long" has bigger numbers --
# not because the documents are more similar in TOPIC. Cosine similarity correctly shows
# they point in the same direction (~1.0), so it isn't fooled by document length.
```

---
⬅ [Day 47](day-47-vector-norm-vector-distance.md) · [Layer 7 Summary](layer-7-summary-review.md) · [Day 49 →](day-49-matrix-and-matrix-multiplication.md)
