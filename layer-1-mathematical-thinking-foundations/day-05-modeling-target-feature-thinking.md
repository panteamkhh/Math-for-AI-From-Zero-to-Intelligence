# Layer 1 — Day 5: Modeling, Target, Feature Thinking

**Keywords:** Modeling · Target · Feature Thinking

---

### 🎯 Intuition
Modeling means simplifying a complex reality into a computable relationship between inputs and an output. A feature is any attribute we believe helps predict the target.

### 💡 ML/AI Application
Every ML project starts with the same three questions: What is the target? What features are available? What shape can the relationship between them take (linear, non-linear)? Feature Engineering is the art of choosing or constructing the best features for that relationship.

### 📝 Mental Exercise
To predict the price of an apartment, propose three features and explain why you think each one affects the target.

### 🧪 Hands-on
```python
import pandas as pd

raw = pd.DataFrame({
    "size_sqm": [50, 80, 120],
    "build_year": [2015, 1998, 2005],
    "num_rooms": [1, 3, 4],
    "price": [95000, 130000, 210000],
})

# Feature engineering: turning raw columns into features that carry more signal
raw["age_years"] = 2026 - raw["build_year"]
raw["price_per_sqm"] = raw["price"] / raw["size_sqm"]     # useful for EDA, not as an input feature
raw["rooms_per_sqm"] = raw["num_rooms"] / raw["size_sqm"]  # a proxy for "layout density"

print(raw[["size_sqm", "age_years", "num_rooms", "rooms_per_sqm", "price"]])
# Target: price
# Candidate features: size_sqm, age_years, num_rooms, rooms_per_sqm
```

---
⬅ [Day 4](day-04-patterns-sequences-abstraction.md) · [Layer 1 Summary](layer-1-summary-review.md)

**Next layer:** Layer 2 — Algebra & Functions →
