# Layer 8 — Day 73: Attention

**Keywords:** Attention

---

### 🎯 Intuition
Attention is a mechanism that lets a model decide, at every moment, which parts of the input are "more important" and deserve more focus — much like how our mind unconsciously focuses more on key words while reading a sentence.

### 💡 ML/AI Application
Attention is the heart of the Transformer architecture, which underlies every modern LLM (GPT, Claude, etc.); it lets a model learn dependencies between words far apart in a text, without the limitations of older recurrent networks.

### 📝 Mental Exercise
In the sentence "The cat sat on the mat because it was tired," why must the attention mechanism be able to connect the word "it" to both "tired" and "cat"?

### 🧪 Hands-on
```python
import numpy as np

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

words = ["The", "cat", "sat", "on", "the", "mat", "because", "it", "was", "tired"]

# Toy attention scores: how much "it" (index 7) attends to every other word
# (in a real model these come from Query.Key dot products -- here we hand-craft
# them to show the pattern attention is supposed to learn)
raw_scores = np.array([0.1, 4.0, 0.3, 0.1, 0.1, 0.2, 0.2, 0.5, 0.3, 3.8])
attention_weights = softmax(raw_scores)

for word, weight in zip(words, attention_weights):
    bar = "█" * int(weight * 50)
    print(f"{word:10s} {weight:.3f} {bar}")

# Notice "cat" and "tired" both receive much higher attention weight than the
# other words -- exactly the ability attention needs: connecting "it" to BOTH
# its subject ("cat") and the reason clause ("tired") at the same time,
# regardless of how far apart they are in the sentence.
```

---
⬅ [Day 72](day-72-embeddings.md) · [Layer 8 Summary](layer-8-summary-review.md) · [Day 74 →](day-74-query-key-value.md)
