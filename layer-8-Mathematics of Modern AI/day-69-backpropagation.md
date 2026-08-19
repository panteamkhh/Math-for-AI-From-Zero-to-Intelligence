# Layer 8 — Day 69: Backpropagation

**Keywords:** Backpropagation

---

### 🎯 Intuition
Backpropagation means systematically applying the chain rule from the network's output back toward its input, to efficiently compute the gradient of every weight — without this algorithm, training deep networks would be practically impossible.

### 💡 ML/AI Application
This is the algorithm every deep learning framework (PyTorch, TensorFlow) implements automatically via "autograd"; understanding it intuitively helps you debug problems like vanishing/exploding gradients.

### 📝 Mental Exercise
In very deep networks, why can the gradients flowing back from the last layer to the first layer become extremely small (vanishing gradient)?

### 🧪 Hands-on
```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# Sigmoid's derivative maxes out at 0.25 -- and it's usually much smaller than that
x_values = np.array([-3, -1, 0, 1, 3])
derivatives = sigmoid_derivative(x_values)
print("Sigmoid derivatives at various points:", np.round(derivatives, 4))
print("Maximum possible sigmoid derivative: 0.25")

# Simulate what happens across many layers: the chain rule MULTIPLIES these
# small derivatives together, layer after layer
n_layers = 20
per_layer_derivative = 0.2  # a typical sigmoid derivative value
accumulated_gradient = per_layer_derivative ** n_layers
print(f"\nGradient after passing back through {n_layers} sigmoid layers: {accumulated_gradient:.2e}")
# Multiplying many numbers smaller than 1 together shrinks the product toward
# zero exponentially fast -- this is EXACTLY the vanishing gradient problem,
# and exactly why ReLU (derivative of 1, not <0.25) helps avoid it.
```

---
⬅ [Day 68](day-68-chain-rule.md) · [Layer 8 Summary](layer-8-summary-review.md) · [Day 70 →](day-70-activation-functions.md)
