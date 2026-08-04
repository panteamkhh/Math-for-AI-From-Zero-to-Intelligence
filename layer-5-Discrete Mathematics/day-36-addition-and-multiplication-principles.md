# Layer 5 — Day 36: The Addition Principle & The Multiplication Principle

**Keywords:** Addition Principle · Multiplication Principle

---

### 🎯 Intuition
The addition principle says: if two choices are independent alternatives (either this or that), you add up the total number of possibilities. The multiplication principle says: if choices happen one after another, you multiply the total number of possibilities.

### 💡 ML/AI Application
Calculating the hyperparameter search space in Grid Search uses exactly the multiplication principle: with 3 values for learning rate and 4 values for batch size, there are 12 combinations to test.

### 📝 Mental Exercise
If you want to test 4 values for learning rate, 3 values for network depth, and 2 values for batch size, how many total combinations do you need to test?

### 🧪 Hands-on
```python
from itertools import product

learning_rates = [0.1, 0.01, 0.001, 0.0001]
depths = [2, 4, 6]
batch_sizes = [16, 32]

combinations = list(product(learning_rates, depths, batch_sizes))

print(f"Total combinations: {len(combinations)}")
print(f"Multiplication principle check: {len(learning_rates)} x {len(depths)} x {len(batch_sizes)} = "
      f"{len(learning_rates) * len(depths) * len(batch_sizes)}")

for combo in combinations[:5]:
    print(combo)
print("...")
# This is EXACTLY why Grid Search gets expensive fast: the multiplication principle
# means adding one more hyperparameter value multiplies your total runs, not just adds to them.
```

---
⬅ [Day 35](day-35-relations-and-functions-discrete-view.md) · [Layer 5 Summary](layer-5-summary-review.md) · [Day 37 →](day-37-permutations-and-combinations.md)
