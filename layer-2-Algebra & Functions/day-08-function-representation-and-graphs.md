# Layer 2 — Day 8: Function Representation and Graphs

**Keywords:** Function Representation · Function Graph

---

### 🎯 Intuition
A function can be shown as a formula, a table, or a graph. A graph lets us "see" the function's behavior without computing every point individually.

### 💡 ML/AI Application
In ML we plot constantly: loss curves across epochs, feature distribution plots, a classifier's decision boundary. Being able to read these plots means being able to "debug with your eyes."

### 📝 Mental Exercise
If a training loss curve oscillates wildly (jumps up and down), what does that suggest about the learning rate?

### 💬 Worked Answer
The **learning rate is probably too large**. 🎯

The model takes steps so big that it overshoots the optimal point and can't settle into the loss minimum, so the loss value keeps bouncing up and down.

Usual fix: **lower the learning rate** or use a **learning rate scheduler**. 🚀

### 🧪 Hands-on
```python
import numpy as np

def train_toy(learning_rate, steps=30):
    # a simple 1D quadratic loss: loss(w) = (w - 3)^2
    w = 0.0
    history = []
    for _ in range(steps):
        grad = 2 * (w - 3)
        w = w - learning_rate * grad
        history.append((w - 3) ** 2)  # the loss value
    return history

stable = train_toy(learning_rate=0.1)
unstable = train_toy(learning_rate=1.05)  # too large -> overshoots repeatedly

print("Stable LR loss trace (last 5):  ", [round(v, 3) for v in stable[-5:]])
print("Unstable LR loss trace (last 5):", [round(v, 3) for v in unstable[-5:]])
# With too-large a learning rate, the loss doesn't settle down -- it oscillates or explodes.
```

---
⬅ [Day 7](day-07-domain-and-range.md) · [Layer 2 Summary](layer-2-summary-review.md) · [Day 9 →](day-09-increasing-decreasing-max-min-symmetry-periodicity.md)
