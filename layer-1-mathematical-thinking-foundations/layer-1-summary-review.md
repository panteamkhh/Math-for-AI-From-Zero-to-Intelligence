# 🧩 Layer 1 — Summary Review

**Mathematical Thinking Foundations**

> A fast recap of all 5 days. Read this before an interview, a refresher, or right after finishing the full daily files. If a row doesn't ring a bell, go back to that day's file.

| Day | Topic | Core Idea (one line) | Where it shows up in ML/AI |
|---|---|---|---|
| [1](day-01-sets-number-types-categorization.md) | Sets, Number Types, Categorization | Grouping data by "type" is the same idea as type-checking / data schemas | Deciding categorical vs. numerical vs. boolean → drives encoding (One-Hot, Label Encoding) |
| [2](day-02-absolute-value-distance-intervals.md) | Absolute Value, Distance, Intervals | Absolute value = "how far," ignoring direction | KNN, K-Means, recommenders, embedding search all reduce to some notion of distance |
| [3](day-03-exponents-exponential-growth-logarithms.md) | Exponents, Exponential Growth, Logarithms | Exponential = growth rate that itself grows; log = compresses big numbers to a comparable scale | Cross-Entropy loss uses logs; learning rate is tuned on a log scale, not linear |
| [4](day-04-patterns-sequences-abstraction.md) | Patterns, Sequences, Abstraction | A pattern is the rule behind a sequence; abstraction = dropping irrelevant detail | The whole point of ML: learn the pattern, not the noise (overfitting = memorizing noise) |
| [5](day-05-modeling-target-feature-thinking.md) | Modeling, Target, Feature Thinking | Modeling = simplifying reality into an input → output relationship | Every ML project starts with: what's the target? what features do I have? what shape is the relationship? |

---

## 🧠 60-second mental drill

Answer each in one sentence, out loud or in your head, without looking back:

1. **Day 1** — Why does knowing a column's "type" change how you preprocess it?
2. **Day 2** — Why do you need to scale features before computing distance between them?
3. **Day 3** — Why is a learning-rate search done on a log scale instead of a linear one?
4. **Day 4** — What's the difference, in your own words, between "learning a pattern" and "memorizing data"?
5. **Day 5** — What are the three questions every ML project starts with?

<details>
<summary>Show quick answers</summary>

1. It determines the encoding you use (numerical stays as-is, categorical needs One-Hot/Label Encoding, boolean often becomes 0/1) and which models can consume it directly.
2. Because raw units (e.g. age in years vs. income in dollars) live on wildly different scales — without scaling, the feature with the bigger raw range dominates the distance and silently drowns out the other.
3. Because the *effect* of learning rate on training is exponential, not linear — a linear grid wastes most of its samples in a narrow, less-informative region, while a log grid spreads samples evenly across orders of magnitude.
4. Learning a pattern means capturing the general rule that generalizes to new data; memorizing means fitting the specific noise/quirks of the training set, which fails on new data — that's overfitting.
5. What is the target? What features are available? What shape can the input→output relationship take (linear, non-linear, etc.)?

</details>

---

## 🗺️ Layer 1 concept map (in one paragraph)

Everything in this layer builds toward one skill: **turning a messy real-world problem into something you can compute with.** Sets and types (Day 1) tell you *what kind* of thing each piece of data is. Distance (Day 2) tells you *how to compare* two things once you've typed them. Exponents and logs (Day 3) tell you *how to handle scale* — when raw numbers grow too fast or too skewed to compare directly. Patterns and abstraction (Day 4) tell you *what to keep and what to throw away* when you generalize from examples. And modeling (Day 5) ties it all together: pick a target, pick features (using type-thinking from Day 1), measure relationships (using distance/scale-thinking from Days 2–3), and generalize instead of memorize (Day 4). This is the mental toolkit every later layer — algebra, calculus, linear algebra, probability — will build directly on top of.

---
**Full days:** [Day 1](day-01-sets-number-types-categorization.md) · [Day 2](day-02-absolute-value-distance-intervals.md) · [Day 3](day-03-exponents-exponential-growth-logarithms.md) · [Day 4](day-04-patterns-sequences-abstraction.md) · [Day 5](day-05-modeling-target-feature-thinking.md)

**Next layer:** Layer 2 — Algebra & Functions →
