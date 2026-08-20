# Layer 8 — Day 71: Softmax

**Keywords:** Softmax

---

### 🎯 Intuition
Softmax converts a set of raw numbers (logits) into a valid probability distribution — all positive, and summing to exactly one.

### 💡 ML/AI Application
The output layer of almost every multi-class classification model (and next-token prediction in LLMs) uses softmax to turn raw scores into interpretable probabilities.

### 📝 Mental Exercise
When an LLM "picks the next token," why is it really sampling from the softmax output treated as a probability distribution?

### 🧪 Hands-on
```python
import numpy as np

def softmax(logits, temperature=1.0):
    scaled = logits / temperature
    exp_scores = np.exp(scaled - np.max(scaled))
    return exp_scores / exp_scores.sum()

vocab = ["cat", "dog", "car", "banana"]
logits = np.array([4.5, 4.2, 1.0, 0.5])  # raw scores for "next token"

probs = softmax(logits)
for token, p in zip(vocab, probs):
    print(f"{token:8s}: {p:.4f}")

print(f"\nSum of probabilities: {probs.sum():.4f}  (always 1.0)")

# Sample the "next token" the way an LLM actually does -- draw from this distribution
np.random.seed(1)
sampled_token = np.random.choice(vocab, p=probs)
print(f"\nSampled next token: '{sampled_token}'")
# This IS how an LLM picks its next word -- not by always taking the single
# highest-scoring option, but by sampling according to the full probability
# distribution that softmax produced.
```

---
⬅ [Day 70](day-70-activation-functions.md) · [Layer 8 Summary](layer-8-summary-review.md) · [Day 72 →](day-72-embeddings.md)
