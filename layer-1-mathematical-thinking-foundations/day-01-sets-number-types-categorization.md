# Layer 1 — Day 1: Sets, Number Types, Categorization

**Keywords:** Sets · Number Types · Precise Definitions · Categorization

> 📌 This isn't about memorizing formulas — the goal is mathematical intuition and seeing the direct link to ML/AI.

---

### 🎯 Intuition
A set is a "container" for grouping similar things. When you categorize data (integers, decimals, categorical labels), you're really saying what *type* each piece of data belongs to. That's the same idea behind type-checking and data schemas in software.

### 💡 ML/AI Application
In ML, every column in a dataset has a set of allowed values (categorical vs. numerical vs. boolean). Correctly identifying the data type is the first step of preprocessing, and it directly drives your choice of encoding (One-Hot, Label Encoding) and even which model family fits best.

### 📝 Mental Exercise
Imagine a hypothetical dataset (say, customers of a retail store) and classify each column by the type of set it belongs to.

### 🧪 Hands-on
```python
import pandas as pd

df = pd.DataFrame({
    "customer_id": [1, 2, 3],
    "age": [25, 41, 33],
    "country": ["Iran", "Germany", "USA"],
    "is_subscribed": [True, False, True],
    "purchase_amount": [120.5, 340.0, 89.9],
})

# This is literally "set thinking": which type-set does each column belong to?
for col in df.columns:
    print(f"{col:18s} -> {df[col].dtype}")

# Why it matters downstream:
# int/float   -> can feed directly into most models
# bool        -> often treated as 0/1
# object/str  -> needs encoding (One-Hot / Label Encoding) before most models can use it
```

---
⬅ [Layer 1 Summary](layer-1-summary-review.md) · [Day 2 →](day-02-absolute-value-distance-intervals.md)
