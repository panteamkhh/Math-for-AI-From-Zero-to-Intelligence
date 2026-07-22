# Layer 2 — Day 13: Linear Function

**Keywords:** Linear Function

---

### 🎯 Intuition
A linear function is a straight line — no curvature, and its rate of change (slope) is constant everywhere.

### 💡 ML/AI Application
Linear/Dense layers without an activation function in a neural network do exactly this: a weighted linear combination of the inputs. Without a non-linear activation, stacking multiple linear layers still collapses into a single linear function — which is exactly why activation functions are essential.

### 📝 Mental Exercise
Why is a neural network made entirely of linear layers (with no non-linear activation) effectively equivalent to a single layer?

### 💬 Worked Answer
Because **combining several linear functions is still just a linear function.**

For example, with two layers:
```
Layer2(Layer1(x))
```
this can ultimately be rewritten as a single simple linear relation:
```
Wx + b
```
So stacking more layers doesn't increase the model's learning capacity.

**Activation functions** introduce non-linearity, which is what lets the network learn more complex patterns.

### 🧪 Hands-on
```python
import numpy as np

np.random.seed(0)
W1, b1 = np.random.randn(3, 3), np.random.randn(3)
W2, b2 = np.random.randn(3, 3), np.random.randn(3)

x = np.random.randn(1, 3)

# Two "linear layers" stacked with NO activation in between
stacked_output = (x @ W1 + b1) @ W2 + b2

# This is mathematically identical to ONE combined linear layer:
W_combined = W1 @ W2
b_combined = b1 @ W2 + b2
single_layer_output = x @ W_combined + b_combined

print("Stacked (2 linear layers):", stacked_output)
print("Single combined layer:    ", single_layer_output)
print("Same result?", np.allclose(stacked_output, single_layer_output))
# Proves: without non-linear activation, depth adds no extra expressive power.
```

---
⬅ [Day 12](day-12-inequalities-and-geometric-interpretation.md) · [Layer 2 Summary](layer-2-summary-review.md) · [Day 14 →](day-14-quadratic-function.md)
