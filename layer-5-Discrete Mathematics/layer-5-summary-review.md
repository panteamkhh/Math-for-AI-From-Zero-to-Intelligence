# 🔢 Layer 5 — Summary Review

**Discrete Mathematics**

> A fast recap of all 7 days. Use this to refresh before an interview or right after finishing the full daily files. If a row doesn't ring a bell, go back to that day's file.

| Day | Topic | Core Idea (one line) | Where it shows up in ML/AI |
|---|---|---|---|
| [33](day-33-propositional-logic-and-truth-tables.md) | Propositional Logic & Truth Tables | Statements that are only true/false, combined with AND/OR/NOT | Pandas filters, DB queries, and boolean feature design use this directly |
| [34](day-34-logical-reasoning-and-sets.md) | Logical Reasoning & Sets (Deeper) | Certain conclusions from true premises; union/intersection/difference | inner join = intersection, outer join = union — the backbone of data prep |
| [35](day-35-relations-and-functions-discrete-view.md) | Relations & Functions (Discrete View) | A relation connects two sets; a function is a relation with exactly one output per input | Knowledge Graphs and relational DB tables are built on "relations" |
| [36](day-36-addition-and-multiplication-principles.md) | Addition & Multiplication Principles | Independent alternatives add; sequential choices multiply | Grid Search's hyperparameter combinations multiply fast — this is why |
| [37](day-37-permutations-and-combinations.md) | Permutations & Combinations | Order matters (permutation) vs. only the group matters (combination) | Feature selection = combination; layer ordering in an architecture = permutation |
| [38](day-38-graphs-and-trees.md) | Graphs & Trees | Nodes + edges; a tree is a graph with no cycles and one root | Decision Trees, social/knowledge graphs, and LLM beam search all use this structure |
| [39](day-39-recursion.md) | Recursion | A function calling itself to solve smaller versions of the same problem | Decision tree building, nested JSON/syntax-tree parsing, some backprop implementations |

---

## 🧠 60-second mental drill

Answer each in one sentence, without looking back:

1. **Day 33** — How does a boolean feature like `is_weekend AND is_holiday` connect to propositional logic?
2. **Day 34** — In "intersection vs. one set plus the intersection" language, what's the difference between an inner join and a left join?
3. **Day 35** — Why is "one email per user" a function but "products a user bought" is not?
4. **Day 36** — Why does adding one more hyperparameter value in Grid Search multiply the total runs instead of just adding to them?
5. **Day 37** — Why is choosing 3 features out of 10 a combination problem, not a permutation problem?
6. **Day 38–39** — How does a Decision Tree's construction connect both to "tree" structure and to recursion at the same time?

<details>
<summary>Show quick answers</summary>

1. `is_weekend AND is_holiday` is literally a propositional-logic AND of two true/false statements, and building its truth table shows every possible combination of the two conditions.
2. An inner join keeps only the intersection — rows present in both tables; a left join keeps everything from the left set, plus whatever intersects from the right (with gaps filled by nulls for anything that doesn't match).
3. A function requires exactly one output per input; each user has exactly one email (input→one output), but a user can have many purchased products (input→many outputs), which is a relation but not a function.
4. Because the multiplication principle applies: sequential, independent choices multiply the total count, so each new hyperparameter dimension multiplies the search space rather than just adding a few more runs.
5. Because only which features end up selected matters, not any order among them — that's the definition of a combination; a permutation would matter if the order of the chosen features changed the outcome.
6. Building a Decision Tree recursively splits a node into subsets and then applies the exact same splitting logic to each subset — this recursive "split and repeat" process is what produces the root→branches→leaves shape that makes it structurally a tree.

</details>

---

## 🗺️ Layer 5 concept map (in one paragraph)

This layer is about **structuring and counting discrete things** — the mathematics that underlies data wrangling and search, rather than continuous change. Propositional logic and sets (Days 33–34) are the language behind every filter, query, and join you'll ever write. Relations and functions viewed discretely (Day 35) explain why some data naturally fits a dictionary (one-to-one) while other data needs a full relational table (one-to-many) — the same distinction that separates knowledge graphs from simple lookups. The addition and multiplication principles (Day 36) explain *why* hyperparameter search spaces explode combinatorially, and permutations vs. combinations (Day 37) tell you exactly when order matters in a choice (layer architecture) and when it doesn't (feature subsets). Finally, graphs, trees, and recursion (Days 38–39) come together as one idea: many of the data structures ML relies on (decision trees, knowledge graphs, nested JSON, syntax trees) are graphs, and the cleanest way to build or traverse them is by having a function call itself on smaller and smaller pieces of the same problem.

---
**Full days:** [33](day-33-propositional-logic-and-truth-tables.md) · [34](day-34-logical-reasoning-and-sets.md) · [35](day-35-relations-and-functions-discrete-view.md) · [36](day-36-addition-and-multiplication-principles.md) · [37](day-37-permutations-and-combinations.md) · [38](day-38-graphs-and-trees.md) · [39](day-39-recursion.md)

**Next layer:** Layer 6 — Probability & Statistics →
