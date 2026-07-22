# Layer 2 — Day 7: Domain and Range

**Keywords:** Domain · Range

---

### 🎯 Intuition
The domain is the set of all allowed inputs; the range is the set of all possible outputs. If an input falls outside the domain, the function is undefined or throws an error.

### 💡 ML/AI Application
Activation functions are known by their range: sigmoid has range (0, 1) for probabilities, tanh has range (-1, 1), ReLU has range [0, ∞). Choosing the model's output layer depends entirely on the range your target needs.

### 📝 Mental Exercise
If the target is a binary classification problem (the output must be a probability), why can't you use ReLU in the final layer?

### 💬 Worked Answer
Because **ReLU doesn't bound its output to the probability range** — it can produce values greater than 1 (like 3 or 10), which don't make sense as a probability.

For binary classification, the output must be between **0** and **1**, so we use **Sigmoid** so the output can be interpreted as a probability. 🎯

- ReLU → good for hidden/intermediate layers
- Sigmoid → good for the output of binary classification

### 🧪 Hands-on
```python
import numpy as np

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

raw_scores = np.array([-3, -0.5, 0, 2, 7])

print("ReLU output: ", relu(raw_scores))      # range [0, inf) -- NOT a probability
print("Sigmoid output:", sigmoid(raw_scores)) # range (0, 1) -- valid probability

# Check the ranges explicitly
print("ReLU max can exceed 1:   ", relu(raw_scores).max() > 1)
print("Sigmoid always in (0,1):", ((sigmoid(raw_scores) > 0) & (sigmoid(raw_scores) < 1)).all())
```

---
⬅ [Day 6](day-06-function-as-a-machine.md) · [Layer 2 Summary](layer-2-summary-review.md) · [Day 8 →](day-08-function-representation-and-graphs.md)
