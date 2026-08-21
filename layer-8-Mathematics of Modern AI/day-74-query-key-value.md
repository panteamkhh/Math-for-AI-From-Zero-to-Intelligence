# Layer 8 — Day 74: Query / Key / Value

**Keywords:** Query · Key · Value

---

### 🎯 Intuition
In the Attention mechanism, the Query means "what am I looking for," the Key means "what does each part of the input have to offer," and the Value is "the actual content that gets passed along if there's a match" — like a search system that matches a question against labels and returns the relevant content.

### 💡 ML/AI Application
The attention score is computed from the dot product between Query and Key (exactly the concept we saw on Day 48), and these scores are then used to weight the Values — this is literally the computational engine of every Transformer layer.

### 📝 Mental Exercise
Why must the similarity score between Query and Key (computed via dot product) pass through a softmax before being applied to the Value?

### 🧪 Hands-on
```python
import numpy as np

np.random.seed(0)

d = 4  # embedding dimension
n_tokens = 3

X = np.random.randn(n_tokens, d)   # 3 token embeddings

W_q = np.random.randn(d, d) * 0.5
W_k = np.random.randn(d, d) * 0.5
W_v = np.random.randn(d, d) * 0.5

Q = X @ W_q
K = X @ W_k
V = X @ W_v

# Attention scores: dot product between every Query and every Key
raw_scores = Q @ K.T
print("Raw Query.Key scores:\n", np.round(raw_scores, 2))

def softmax_rows(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

attention_weights = softmax_rows(raw_scores)
print("\nAttention weights after softmax (each row sums to 1):\n", np.round(attention_weights, 3))

output = attention_weights @ V
print("\nFinal attention output (weighted combination of Values):\n", np.round(output, 3))

# Without softmax, raw dot-product scores could be any real number (including
# negative), which wouldn't make sense as "how much weight to give this Value."
# Softmax turns them into a clean, positive, sum-to-1 weighting -- a proper
# blend of the Values, not an unconstrained combination.
```

---
⬅ [Day 73](day-73-attention.md) · [Layer 8 Summary](layer-8-summary-review.md) · [Day 75 →](day-75-positional-encoding.md)
