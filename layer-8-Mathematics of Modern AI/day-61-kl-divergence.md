# Layer 8 — Day 61: KL Divergence

**Keywords:** KL Divergence

---

### 🎯 Intuition
KL Divergence also measures the distance between two probability distributions, but it's asymmetric: the distance from A to B isn't necessarily equal to the distance from B to A. It shows how much "information" is lost when approximating one distribution with another.

### 💡 ML/AI Application
In Variational Autoencoders (VAEs) and some Reinforcement Learning methods (like PPO), KL Divergence is used to limit how much the model's distribution is allowed to shift at each training step, keeping learning stable.

### 📝 Mental Exercise
Why can limiting the KL Divergence between a fine-tuned LLM and the original model help prevent catastrophic forgetting?

### 🧪 Hands-on
```python
import numpy as np

def kl_divergence(p, q):
    p, q = np.array(p), np.array(q)
    return np.sum(p * np.log(p / q))

original_model_dist = np.array([0.5, 0.3, 0.2])
slightly_tuned_dist  = np.array([0.45, 0.35, 0.2])
wildly_tuned_dist    = np.array([0.05, 0.05, 0.9])

print("KL(original || slightly tuned):", round(kl_divergence(original_model_dist, slightly_tuned_dist), 4))
print("KL(original || wildly tuned):  ", round(kl_divergence(original_model_dist, wildly_tuned_dist), 4))

# Notice the KL divergence is MUCH larger for the "wildly tuned" distribution --
# a large KL penalty during fine-tuning discourages the model from straying too
# far from its original, broadly-capable distribution, which is exactly what
# helps prevent it from "forgetting" everything it knew before fine-tuning.
```

---
⬅ [Day 60](day-60-cross-entropy.md) · [Layer 8 Summary](layer-8-summary-review.md) · [Day 62 →](day-62-gradient.md)
