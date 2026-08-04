# Layer 5 — Day 37: Permutations & Combinations

**Keywords:** Permutation · Combination

---

### 🎯 Intuition
A permutation is an arrangement where order matters (A-then-B is different from B-then-A). A combination is when only the selected group matters, not the order.

### 💡 ML/AI Application
In model evaluation, choosing a subset of features for feature selection is a "combination" problem (order doesn't matter), while the order of layers in a network architecture is a "permutation" problem.

### 📝 Mental Exercise
If you have 10 features and want to pick 3 of them for your model (regardless of order), is this a permutation problem or a combination problem?

### 🧪 Hands-on
```python
from itertools import permutations, combinations

features = ["age", "income", "education", "experience", "location",
            "gender", "credit_score", "employment", "region", "tenure"]

# COMBINATION: choosing 3 features, order doesn't matter
feature_combos = list(combinations(features, 3))
print(f"Feature selection (combinations): {len(feature_combos)} possible subsets")
print("Example:", feature_combos[0])

# PERMUTATION: ordering 3 chosen layer types, order DOES matter
layer_types = ["Dense", "Dropout", "BatchNorm"]
layer_orders = list(permutations(layer_types))
print(f"\nLayer ordering (permutations): {len(layer_orders)} possible arrangements")
for order in layer_orders:
    print(" -> ".join(order))
# Choosing WHICH 3 features = combination (10 choose 3 = 120).
# Deciding the ORDER of 3 layers = permutation (3! = 6), because Dense->Dropout->BatchNorm
# behaves differently from BatchNorm->Dense->Dropout.
```

---
⬅ [Day 36](day-36-addition-and-multiplication-principles.md) · [Layer 5 Summary](layer-5-summary-review.md) · [Day 38 →](day-38-graphs-and-trees.md)
