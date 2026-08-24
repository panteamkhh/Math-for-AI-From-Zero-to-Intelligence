# Layer 8 — Day 79: Dropout & Normalization (Batch Norm / Layer Norm)

**Keywords:** Dropout · Batch Normalization · Layer Normalization

---

### 🎯 Intuition
Dropout randomly "turns off" a fraction of neurons during each training step, forcing the network to not over-rely on any single one. Normalization (Batch/Layer Norm) rescales the values flowing through the network so they stay in a stable, well-behaved range at every layer.

### 💡 ML/AI Application
Dropout is a direct, practical way to fight the high-variance side of the bias-variance tradeoff (Day 43) — it's regularization implemented as noise instead of a penalty term. Batch/Layer Norm keeps activations from drifting into extreme ranges as they pass through dozens of layers, which is exactly what keeps very deep networks (and transformers) trainable at all.

### 📝 Mental Exercise
Why does randomly dropping neurons during training act as a form of regularization, similar in spirit to what L1/L2 penalties do?

### 🧪 Hands-on
```python
import numpy as np

np.random.seed(0)

def dropout(x, rate=0.5, training=True):
    if not training:
        return x
    mask = (np.random.rand(*x.shape) > rate).astype(float)
    # Scale up the surviving activations so the expected total stays the same
    return x * mask / (1 - rate)

def layer_norm(x, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)

activations = np.array([[2.0, 8.0, -3.0, 5.0, 1.0]])

print("Original activations:      ", activations)
print("After dropout (training):  ", np.round(dropout(activations, rate=0.4), 2))
print("After layer norm:          ", np.round(layer_norm(activations), 3))

# Dropout forces the network to spread useful information across MANY neurons
# instead of concentrating it in a few -- if any single neuron might vanish on
# a given step, the network can't afford to depend on it alone. That's exactly
# the same "don't over-rely on any one feature/weight" pressure L1/L2 penalties
# create, just implemented through random noise instead of a fixed formula.
```

---
⬅ [Layer 8 Summary](layer-8-summary-review.md) · [Day 78](day-78-generalization.md) · [Day 80 →](day-80-rlhf-and-the-math-of-alignment.md)
