# Layer 4 — Day 31: Integral & Area Under the Curve

**Keywords:** Integral · Area Under the Curve

---

### 🎯 Intuition
An integral means summing an infinite number of infinitesimally small pieces to arrive at a "whole" — like adding up the area under a curve by slicing it into thin strips.

### 💡 ML/AI Application
The area under the ROC curve (AUC-ROC) is one of the most important evaluation metrics for classification models, and it uses exactly this idea — the integral (area under a curve).

### 📝 Mental Exercise
Why does an AUC close to 1 indicate a good classifier, while an AUC close to 0.5 indicates a model that's basically guessing randomly?

### 🧪 Hands-on
```python
import numpy as np
from sklearn.metrics import roc_curve, auc

np.random.seed(0)
n = 500
y_true = np.random.randint(0, 2, n)

# A "good" model: scores correlate strongly with the true label
good_scores = y_true + np.random.normal(0, 0.4, n)

# A "random" model: scores have nothing to do with the true label
random_scores = np.random.normal(0, 1, n)

for name, scores in [("Good model", good_scores), ("Random model", random_scores)]:
    fpr, tpr, _ = roc_curve(y_true, scores)
    area = auc(fpr, tpr)   # literally the integral (area under the curve)
    print(f"{name}: AUC = {area:.3f}")

# A good model's ROC curve bulges toward the top-left, so the area under it approaches 1.
# A random model's ROC curve sits close to the diagonal, so its area is close to 0.5.
```

---
⬅ [Day 30](day-30-optimization-critical-points-max-min.md) · [Layer 4 Summary](layer-4-summary-review.md) · [Day 32 →](day-32-fundamental-theorem-of-calculus-and-gradient-intuition.md)
