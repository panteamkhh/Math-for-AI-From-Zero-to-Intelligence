# Layer 2 — Day 10: Linear Equations

**Keywords:** Linear Equations

---

### 🎯 Intuition
A linear equation is the simplest kind of relationship: a change in the input always affects the output by a fixed ratio (`y = ax + b`). There's no curvature or complexity.

### 💡 ML/AI Application
Linear regression — one of the most fundamental and widely used ML models — is exactly this equation, except instead of one input we have several features, and we learn the coefficients (`a`) from data.

### 📝 Mental Exercise
If you build a linear regression model to predict salary from years of experience, what does the coefficient `a` mean?

### 💬 Worked Answer
The coefficient **a (the weight, or slope)** shows how much the salary changes on average for **one additional unit of experience**. 🎯

For example, if:
```
Salary = 5 × Experience + b
```
that means each extra year of experience creates roughly a **5-unit increase in salary**.

So: **a = the size of experience's effect on salary**

### 🧪 Hands-on
```python
import numpy as np

# Toy dataset: years of experience -> salary (in $1000s)
experience = np.array([1, 2, 3, 5, 8, 10])
salary = np.array([42, 47, 53, 62, 78, 88])

# Fit y = a*x + b with least squares
a, b = np.polyfit(experience, salary, deg=1)
print(f"salary ≈ {a:.2f} * experience + {b:.2f}")
print(f"Interpretation: each extra year of experience adds about {a:.2f}k to salary")

# Predict for someone with 6 years of experience
pred = a * 6 + b
print(f"Predicted salary at 6 years experience: {pred:.1f}k")
```

---
⬅ [Day 9](day-09-increasing-decreasing-max-min-symmetry-periodicity.md) · [Layer 2 Summary](layer-2-summary-review.md) · [Day 11 →](day-11-quadratic-equations.md)
