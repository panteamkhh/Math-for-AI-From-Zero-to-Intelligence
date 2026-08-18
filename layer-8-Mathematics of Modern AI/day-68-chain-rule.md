# Layer 8 — Day 68: Chain Rule

**Keywords:** Chain Rule

---

### 🎯 Intuition
The chain rule says the derivative of a composite function equals the product of the derivatives of its individual parts — exactly like tracing the effect of a small change from the start to the end of a causal chain.

### 💡 ML/AI Application
This is the heart of backpropagation: the error at the network's final layer is propagated backward, layer by layer, using the chain rule to compute the gradient of every weight at every layer.

### 📝 Mental Exercise
If a network has 50 layers, why does the chain rule let us compute the gradient of the first layer simply, without directly computing the entire complex composite function?

### 🧪 Hands-on
```python
import numpy as np

# A tiny 3-function composition: y = h(g(f(x)))
def f(x): return x ** 2
def g(x): return np.sin(x)
def h(x): return 3 * x + 1

def f_prime(x): return 2 * x
def g_prime(x): return np.cos(x)
def h_prime(x): return 3

x = 1.5

# Chain rule: dy/dx = h'(g(f(x))) * g'(f(x)) * f'(x)
fx = f(x)
gfx = g(fx)
dy_dx = h_prime(gfx) * g_prime(fx) * f_prime(x)

print(f"dy/dx via chain rule: {dy_dx:.5f}")

# Verify with a tiny numeric perturbation
eps = 1e-6
y1 = h(g(f(x)))
y2 = h(g(f(x + eps)))
print(f"dy/dx via numeric approximation: {(y2 - y1) / eps:.5f}")

# The key insight: each function only needs to know its OWN local derivative.
# Chaining 50 of these local derivatives together (multiplying step by step,
# backward from the output) is vastly cheaper than deriving one giant 50-layer
# formula from scratch -- this is exactly what makes backprop tractable.
```

---
⬅ [Day 67](day-67-map.md) · [Layer 8 Summary](layer-8-summary-review.md) · [Day 69 →](day-69-backpropagation.md)
