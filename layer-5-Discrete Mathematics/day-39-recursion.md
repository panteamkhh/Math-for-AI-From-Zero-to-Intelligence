# Layer 5 — Day 39: Recursion

**Keywords:** Recursion

---

### 🎯 Intuition
Recursion means a function calls itself to solve a big problem by solving smaller versions of the same problem.

### 💡 ML/AI Application
Decision tree-building algorithms (recursively splitting data at each node), processing nested structures (JSON, syntax trees in NLP), and even some backpropagation implementations use recursive logic.

### 📝 Mental Exercise
Why can the Decision Tree building algorithm be described recursively as: "split this node, then do the same thing to each resulting subset"?

### 🧪 Hands-on
```python
def build_tree(data, depth=0, max_depth=3):
    indent = "  " * depth
    if depth >= max_depth or len(data) <= 1:
        print(f"{indent}Leaf: predict on {data}")
        return

    # Pretend we found the best split (in a real tree, this involves searching
    # over features/thresholds to find the split that best separates the classes)
    mid = len(data) // 2
    left, right = data[:mid], data[mid:]
    print(f"{indent}Split {data} -> {left} | {right}")

    # RECURSION: the exact same function calls itself on each smaller subset
    build_tree(left, depth + 1, max_depth)
    build_tree(right, depth + 1, max_depth)

build_tree(list(range(8)))
# Each call solves a SMALLER version of the same problem ("split and recurse"),
# until the base case (a leaf) is reached -- this is precisely how real
# Decision Tree implementations grow their branches.
```

---
⬅ [Day 38](day-38-graphs-and-trees.md) · [Layer 5 Summary](layer-5-summary-review.md)

**Next layer:** Layer 6 — Probability & Statistics →
