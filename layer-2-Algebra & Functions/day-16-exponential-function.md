# Layer 2 — Day 16: Exponential Function

**Keywords:** Exponential Function

---

### 🎯 Intuition
An exponential function is always positive, and the larger the input gets, the faster and faster it grows; it never actually reaches zero, but always gets closer to it.

### 💡 ML/AI Application
Softmax (used for multi-class classification) relies on the exponential function to turn a model's raw scores into positive probabilities. Sigmoid is also built on this same function.

### 📝 Mental Exercise
Why does using the exponential function in softmax guarantee that all outputs are positive?

### 💬 Worked Answer
Because the exponential function (`e^x`) **always produces a positive value for any input** — even if the input is negative.

In Softmax, we first exponentiate the model's scores, so all values become positive. Then, by dividing by their sum, these values turn into probabilities between **0** and **1**.

So:
- `e^x > 0` always ✅
- Softmax output → a positive probability for every class

### 🧪 Hands-on
```python
import numpy as np

def softmax(scores):
    exp_scores = np.exp(scores)          # guarantees every value is positive
    return exp_scores / np.sum(exp_scores)

raw_scores = np.array([-5, 0, 2, 10])   # some scores are negative!
probs = softmax(raw_scores)

print("Raw scores:", raw_scores)
print("Softmax probabilities:", np.round(probs, 4))
print("All positive?", (probs > 0).all())
print("Sum to 1?", np.isclose(probs.sum(), 1.0))
```

---
⬅ [Day 15](day-15-absolute-value-function.md) · [Layer 2 Summary](layer-2-summary-review.md) · [Day 17 →](day-17-logarithmic-function.md)
