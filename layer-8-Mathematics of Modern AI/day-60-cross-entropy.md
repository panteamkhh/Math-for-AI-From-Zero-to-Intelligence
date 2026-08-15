# Layer 8 — Day 60: Cross Entropy

**Keywords:** Cross Entropy

---

### 🎯 Intuition
Cross-Entropy measures the "distance" between two probability distributions: the true distribution (correct label) and the model's predicted distribution. The closer they are, the lower the cross-entropy.

### 💡 ML/AI Application
This is the industry-standard loss function for classification — from simple logistic regression all the way to the largest LLMs, they all train using cross-entropy (or a variant of it).

### 📝 Mental Exercise
Why does cross-entropy so heavily punish the model for confident but wrong predictions (e.g., 0.99 probability on the wrong class)?

### 🧪 Hands-on
```python
import numpy as np

def cross_entropy_loss(p_correct_class):
    return -np.log(p_correct_class)

for p in [0.99, 0.7, 0.4, 0.1, 0.01]:
    print(f"P(correct class)={p:5.2f} -> loss = {cross_entropy_loss(p):.4f}")

# A CONFIDENT correct prediction (p close to 1) gives loss close to 0.
# A CONFIDENT WRONG prediction means p (probability on the CORRECT class) is
# close to 0 -- and -log(p) explodes toward infinity as p shrinks. This sharp
# penalty is exactly what discourages a model from ever being confidently wrong.
```

---
⬅ [Day 59](day-59-entropy.md) · [Layer 8 Summary](layer-8-summary-review.md) · [Day 61 →](day-61-kl-divergence.md)
