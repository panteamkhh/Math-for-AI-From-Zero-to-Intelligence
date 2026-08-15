# Layer 8 — Day 59: Entropy

**Keywords:** Entropy

---

### 🎯 Intuition
Entropy measures the amount of "uncertainty" or "surprise" in a probability distribution. A distribution that's very unsure (everything equally likely) has high entropy.

### 💡 ML/AI Application
In Decision Trees, entropy is used to decide which feature is the best "splitter" — the split that produces the largest entropy reduction (i.e., the largest information gain) is chosen.

### 📝 Mental Exercise
Why does a Decision Tree node where every sample belongs to just one class have zero entropy?

### 🧪 Hands-on
```python
import numpy as np

def entropy(probs):
    probs = np.array(probs)
    probs = probs[probs > 0]  # avoid log(0)
    return -np.sum(probs * np.log2(probs))

pure_node = [1.0, 0.0]       # 100% one class -- zero uncertainty
mixed_node = [0.5, 0.5]      # maximum uncertainty for 2 classes
mostly_pure = [0.9, 0.1]

print(f"Pure node entropy:    {entropy(pure_node):.4f}")
print(f"Mixed node entropy:   {entropy(mixed_node):.4f}")
print(f"Mostly-pure entropy:  {entropy(mostly_pure):.4f}")
# A pure node has NO uncertainty about which class a sample belongs to --
# there's nothing left to "learn" from splitting it further, so entropy = 0.
# A 50/50 split is maximally uncertain, giving the highest possible entropy.
```

---
⬅ [Day 58](day-58-convexity.md) · [Layer 8 Summary](layer-8-summary-review.md) · [Day 60 →](day-60-cross-entropy.md)
