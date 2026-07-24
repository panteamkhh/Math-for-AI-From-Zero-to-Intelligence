# Layer 2 — Day 6: The Concept of a Function (Function as a Machine)

**Keywords:** Function as a Machine

---

### 🎯 Intuition
Think of a function as a machine: something goes in, gets processed, and something else comes out. Every specific input has exactly one specific output — never more, never less.

### 💡 ML/AI Application
At the most basic level, a machine learning model is just a very complex function: `f(features) = prediction`. Training a model means finding the parameters of this function that produce the best output for our dataset.

### 📝 Mental Exercise
Picture a neural network as a chain of functions — each layer is one function. How does this picture help you understand the forward pass?

### 💬 Worked Answer
In the **forward pass**, data moves from the first layer to the last, and each layer applies a function to the output of the previous one:

```
x → f1(x) → f2(f1(x)) → ... → prediction
```

This view helps explain how a neural network builds new features at each layer and eventually turns raw input into a prediction.

### 🧪 Hands-on
```python
import numpy as np

# Treat each "layer" as a plain function: input -> output
def layer1(x):
    return np.maximum(0, x @ W1 + b1)   # linear + ReLU

def layer2(x):
    return x @ W2 + b2                  # linear, final layer

np.random.seed(0)
W1, b1 = np.random.randn(4, 8), np.zeros(8)
W2, b2 = np.random.randn(8, 1), np.zeros(1)

x = np.random.randn(1, 4)

# Forward pass = function composition, one layer at a time
h = layer1(x)
prediction = layer2(h)

print("input:", x)
print("hidden representation:", h)
print("prediction:", prediction)
# This is literally: prediction = layer2(layer1(x))
```

---
⬅ [Layer 2 Summary](layer-2-summary-review.md) · [Day 7 →](day-07-domain-and-range.md)
