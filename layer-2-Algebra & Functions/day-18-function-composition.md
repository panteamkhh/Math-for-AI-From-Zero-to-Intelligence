# Layer 2 — Day 18: Function Composition

**Keywords:** Function Composition

---

### 🎯 Intuition
Composing functions means taking the output of one function and feeding it as the input to the next — like an assembly line where each station does one small piece of processing.

### 💡 ML/AI Application
A deep neural network is exactly a long composition of functions: each layer takes the previous layer's function and produces a new one. The depth of the network is how many times this composition repeats.

### 📝 Mental Exercise
If `f` is the first layer and `g` is the second, write the network's output as `g(f(x))` and explain why this matters for understanding backpropagation.

### 💬 Worked Answer
The network's output:
```
y = g(f(x))
```
means input `x` first goes through the first layer's function `f`, and then that output goes through the second layer `g`.

This concept matters for **Backpropagation** because the algorithm needs to trace each layer's effect on the final error — it does this using the **Chain Rule**, propagating the gradient backward from the output to the earlier layers.

### 🧪 Hands-on
```python
import numpy as np

def f(x):        # layer 1
    return np.tanh(x)

def g(x):         # layer 2
    return x ** 2

x = np.array([0.5])

# Composition: y = g(f(x))
y = g(f(x))

# Chain rule for the derivative: dy/dx = g'(f(x)) * f'(x)
def f_prime(x):
    return 1 - np.tanh(x) ** 2

def g_prime(x):
    return 2 * x

dy_dx = g_prime(f(x)) * f_prime(x)

print("y = g(f(x)) =", y)
print("dy/dx via chain rule =", dy_dx)
# This is exactly what backprop does at every layer boundary, just with vectors/matrices instead of scalars.
```

---
⬅ [Day 17](day-17-logarithmic-function.md) · [Layer 2 Summary](layer-2-summary-review.md) · [Day 19 →](day-19-inverse-function.md)
