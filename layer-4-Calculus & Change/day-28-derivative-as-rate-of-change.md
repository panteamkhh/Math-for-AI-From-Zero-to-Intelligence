# Layer 4 — Day 28: Derivative as Rate of Change

**Keywords:** Derivative as Rate of Change

---

### 🎯 Intuition
The derivative is exactly the slope — but at every single point of a curve instead of one straight line. It tells you: "if I change the input slightly, how much does the output change?"

### 💡 ML/AI Application
This is exactly the question we ask during model training: "if I change this weight slightly, how much does the loss change?" The answer to that question is the gradient, and it's what determines the model's learning path.

### 📝 Mental Exercise
If the derivative of the loss with respect to a weight is large and positive, should you increase or decrease that weight to reduce the loss?

### 🧪 Hands-on
```python
import numpy as np

# Toy loss: loss(w) = (w - 3)^2, so its derivative is 2*(w-3)
def loss(w):
    return (w - 3) ** 2

def d_loss_dw(w):
    return 2 * (w - 3)

w = 8.0  # start far from the minimum at w=3
gradient = d_loss_dw(w)
print(f"weight={w}, gradient={gradient}")  # large & positive

# A large POSITIVE gradient means the loss increases as w increases,
# so to REDUCE the loss we must move w in the opposite (negative) direction.
new_w = w - 0.1 * gradient
print(f"After one step: w = {new_w}, new loss = {loss(new_w):.3f} (down from {loss(w):.3f})")
```

---
⬅ [Day 27](day-27-continuity.md) · [Layer 4 Summary](layer-4-summary-review.md) · [Day 29 →](day-29-derivative-rules.md)
