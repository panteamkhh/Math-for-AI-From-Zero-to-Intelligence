# Layer 5 — Day 38: Graphs & Trees

**Keywords:** Graph · Tree

---

### 🎯 Intuition
A graph is a set of nodes and the edges connecting them. A tree is a special kind of graph with no cycles and exactly one root.

### 💡 ML/AI Application
Decision Trees, social networks, knowledge graphs, and even the beam search architecture in LLMs all directly use graph/tree structures.

### 📝 Mental Exercise
Why do we call a Decision Tree a "tree" — what similarity exists between its decision structure and a real tree?

### 🧪 Hands-on
```python
from sklearn.tree import DecisionTreeClassifier, export_text
import numpy as np

# Toy data: predict "will buy" from age and income
X = np.array([[22, 20], [45, 80], [35, 40], [60, 90], [25, 25], [50, 85]])
y = np.array([0, 1, 0, 1, 0, 1])

tree = DecisionTreeClassifier(max_depth=2).fit(X, y)
print(export_text(tree, feature_names=["age", "income"]))

# Notice the structure printed above: it starts at a single ROOT decision,
# BRANCHES into two paths based on a yes/no question, and ends in LEAVES
# with a final prediction -- exactly the root/branch/leaf structure of a real tree,
# and exactly the "nodes + edges, no cycles, one root" definition of a tree graph.
```

---
⬅ [Day 37](day-37-permutations-and-combinations.md) · [Layer 5 Summary](layer-5-summary-review.md) · [Day 39 →](day-39-recursion.md)
