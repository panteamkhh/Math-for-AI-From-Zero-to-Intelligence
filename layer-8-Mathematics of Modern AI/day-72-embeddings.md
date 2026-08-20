# Layer 8 — Day 72: Embeddings

**Keywords:** Embeddings

---

### 🎯 Intuition
An embedding is the process of turning something discrete and meaningful (a word, an image, a user) into a numeric vector in a continuous space, such that similar things end up as vectors that are close together.

### 💡 ML/AI Application
This is the foundation of almost all modern AI: LLMs work with word embeddings, recommender systems embed users and items, and semantic search is built on distance between embeddings.

### 📝 Mental Exercise
Why does the vector "king" minus "man" plus "woman" end up approximately equal to the vector "queen" in a good embedding space?

### 🧪 Hands-on
```python
import numpy as np

# Toy, hand-crafted "embeddings" illustrating the idea (real ones are learned from data)
embeddings = {
    "king":  np.array([0.90, 0.85, 0.10]),
    "man":   np.array([0.85, 0.15, 0.05]),
    "woman": np.array([0.15, 0.85, 0.05]),
    "queen": np.array([0.20, 1.55, 0.10]),
}

result_vector = embeddings["king"] - embeddings["man"] + embeddings["woman"]
print("king - man + woman =", np.round(result_vector, 3))
print("actual 'queen' vector =", embeddings["queen"])

distance = np.linalg.norm(result_vector - embeddings["queen"])
print(f"Distance between result and 'queen': {distance:.4f}  (small = close match)")

# In a well-trained embedding space, "man -> woman" and "king -> queen" end up
# representing a SIMILAR direction/offset (roughly "make it feminine"), because
# the model learned that this semantic relationship shows up consistently
# across many word pairs during training -- so the same vector arithmetic
# that works for one pair tends to transfer to another.
```

---
⬅ [Day 71](day-71-softmax.md) · [Layer 8 Summary](layer-8-summary-review.md) · [Day 73 →](day-73-attention.md)
