# Layer 2 — Day 17: Logarithmic Function

**Keywords:** Logarithmic Function

---

### 🎯 Intuition
The logarithm is the inverse of the exponential function: it "compresses" very large numbers so they're easier to compare and add up.

### 💡 ML/AI Application
Cross-Entropy Loss uses the logarithm of the predicted probability; when a model confidently predicts wrong (a probability near zero for the correct class), the logarithm turns that into an extremely large penalty.

### 📝 Mental Exercise
Why is the logarithmic penalty for a model giving 0.01 probability to the correct class so much larger than if it gave 0.4?

### 💬 Worked Answer
Because the logarithm turns very small probabilities into a **much larger negative number**.

In Cross-Entropy:
```
Loss = -log(probability)
```
So:
- `-log(0.01)` → a much larger penalty
- `-log(0.4)` → a much smaller penalty

Meaning: when the model is confidently wrong, it gets punished much more heavily — this teaches it to avoid confidently wrong predictions.

### 🧪 Hands-on
```python
import numpy as np

def cross_entropy_penalty(p_correct):
    return -np.log(p_correct)

for p in [0.01, 0.1, 0.4, 0.9, 0.99]:
    print(f"P(correct class)={p:5.2f} -> loss={cross_entropy_penalty(p):.3f}")

# Notice how the penalty explodes as the probability approaches 0,
# and shrinks toward 0 as the probability approaches 1 (a confident, correct prediction).
```

---
⬅ [Day 16](day-16-exponential-function.md) · [Layer 2 Summary](layer-2-summary-review.md) · [Day 18 →](day-18-function-composition.md)
