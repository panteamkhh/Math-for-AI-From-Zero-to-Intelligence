# Layer 6 — Day 45: Hypothesis Testing & Correlation

**Keywords:** Hypothesis Testing · Correlation

---

### 🎯 Intuition
Hypothesis testing is a statistical method for deciding whether an observed difference is real or just chance. Correlation shows how much two variables tend to move "in the same direction" together.

### 💡 ML/AI Application
A/B Testing to compare two versions of a model or product is built directly on hypothesis testing. Correlation is also critical in exploratory data analysis (EDA) for finding features related to the target — though "correlation is not causation."

### 📝 Mental Exercise
Why doesn't finding a high correlation between a feature and the target necessarily mean there's a causal relationship between them?

### 🧪 Hands-on
```python
import numpy as np
from scipy import stats

np.random.seed(0)

# Two variables that are correlated only because they share a hidden common cause
hidden_cause = np.random.normal(0, 1, 200)          # e.g. "outdoor temperature"
ice_cream_sales = hidden_cause * 5 + np.random.normal(0, 1, 200)  # driven by temperature
shark_attacks = hidden_cause * 3 + np.random.normal(0, 1, 200)    # ALSO driven by temperature

corr, p_value = stats.pearsonr(ice_cream_sales, shark_attacks)
print(f"Correlation (ice cream sales vs shark attacks): {corr:.3f}, p-value: {p_value:.5f}")
# High correlation, tiny p-value -- looks "statistically significant"!
# But ice cream sales don't CAUSE shark attacks -- both are driven by a hidden
# third variable (temperature/beach attendance). This is exactly why correlation
# alone can mislead you about causation.
```

---
⬅ [Day 44](day-44-normal-distribution-and-sampling.md) · [Layer 6 Summary](layer-6-summary-review.md) · [Day 46 →](day-46-regression.md)
