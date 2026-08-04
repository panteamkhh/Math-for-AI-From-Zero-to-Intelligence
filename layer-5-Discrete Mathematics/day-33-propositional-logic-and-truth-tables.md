# Layer 5 — Day 33: Propositional Logic & Truth Tables

**Keywords:** Propositional Logic · Truth Table

---

### 🎯 Intuition
Propositional logic deals with statements that can only be true or false, combined with AND, OR, NOT. A truth table shows every possible outcome of that combination.

### 💡 ML/AI Application
Filtering conditions in pandas, database queries, and designing boolean features (like `is_weekend AND is_holiday`) all use propositional logic directly.

### 📝 Mental Exercise
Write a condition using AND/OR logic to filter "users who are both active and have purchased more than once."

### 🧪 Hands-on
```python
import pandas as pd

df = pd.DataFrame({
    "user_id": [1, 2, 3, 4, 5],
    "is_active": [True, True, False, True, False],
    "purchase_count": [3, 1, 5, 0, 2],
})

# Propositional logic, directly translated into a pandas filter:
# "active AND purchased more than once"
condition = (df["is_active"]) & (df["purchase_count"] > 1)

print(df[condition])

# Build the truth table for just these two conditions to see every combination:
truth_table = pd.DataFrame({
    "is_active": [True, True, False, False],
    "purchased_more_than_once": [True, False, True, False],
})
truth_table["AND"] = truth_table["is_active"] & truth_table["purchased_more_than_once"]
truth_table["OR"] = truth_table["is_active"] | truth_table["purchased_more_than_once"]
print("\nTruth table:\n", truth_table)
```

---
⬅ [Layer 5 Summary](layer-5-summary-review.md) · [Day 34 →](day-34-logical-reasoning-and-sets.md)
