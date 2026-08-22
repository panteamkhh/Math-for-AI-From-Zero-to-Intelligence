# Layer 8 — Day 77: Scaling Laws

**Keywords:** Scaling Laws

---

### 🎯 Intuition
Scaling laws are empirical relationships that show how a model's performance improves in a predictable way (often following a power law) as model size, data volume, and compute increase.

### 💡 ML/AI Application
These laws guide major decisions across the AI industry: how much data to collect, how large to make a model, and how to split a compute budget between model size and data volume to get the best return.

### 📝 Mental Exercise
Why do large AI companies study scaling laws on smaller models first, before training a very large and expensive model?

### 🧪 Hands-on
```python
import numpy as np

# A simplified power-law scaling relationship: loss decreases as model size grows
def scaling_law_loss(model_size, a=100, alpha=0.3):
    return a * (model_size ** -alpha)

small_model_sizes = np.array([1e6, 1e7, 1e8])   # sizes we can cheaply experiment with
predicted_losses = scaling_law_loss(small_model_sizes)

print("Observed (small-scale) results:")
for size, loss in zip(small_model_sizes, predicted_losses):
    print(f"  model_size={size:.0e} -> predicted loss={loss:.4f}")

# Now EXTRAPOLATE the fitted power-law trend to a much bigger, expensive model
large_model_size = 1e11
predicted_large_loss = scaling_law_loss(large_model_size)
print(f"\nExtrapolated prediction for model_size={large_model_size:.0e}: loss={predicted_large_loss:.4f}")

# By fitting the power-law trend on CHEAP small-scale runs, a company can predict
# roughly how a much larger (extremely expensive) model will perform BEFORE
# spending millions of dollars actually training it -- letting them make an
# informed bet instead of a costly guess.
```

---
⬅ [Day 76](day-76-similarity-search.md) · [Layer 8 Summary](layer-8-summary-review.md) · [Day 78 →](day-78-generalization.md)
