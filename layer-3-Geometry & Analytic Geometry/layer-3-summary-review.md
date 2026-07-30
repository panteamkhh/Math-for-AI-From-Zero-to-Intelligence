# 📐 Layer 3 — Summary Review

**Geometry & Analytic Geometry**

> A fast recap of all 5 days. Use this to refresh before an interview or right after finishing the full daily files. If a row doesn't ring a bell, go back to that day's file.

| Day | Topic | Core Idea (one line) | Where it shows up in ML/AI |
|---|---|---|---|
| [21](day-21-coordinate-system-and-distance.md) | Coordinate System & Distance | A coordinate system addresses every point in space; distance generalizes Pythagoras | Any embedding is a point in a coordinate space; Euclidean distance = concept similarity |
| [22](day-22-midpoint-and-slope.md) | Midpoint & Slope | Midpoint = exact middle; slope = steepness of a line | Slope is the ancestor of "derivative" and "gradient" — the engine of gradient descent |
| [23](day-23-line-equation-parallel-perpendicular.md) | Line Equation, Parallel & Perpendicular | Same slope = parallel (never meet); 90° = perpendicular | The fitted regression line is a line equation; orthogonality is core to PCA |
| [24](day-24-circle-and-parabola.md) | Circle & Parabola | Circle = constant distance from center; parabola = bowl shape from a quadratic | Loss-function contour shape (circular vs. stretched) determines how fast/stable gradient descent converges |
| [25](day-25-vector-length-angle-projection-embedding-space.md) | Vector, Length, Angle, Projection, Embedding Space | A vector has size + direction; angle shows alignment; projection is a "shadow" | Embeddings are vectors; cosine similarity (the angle) is the standard similarity metric in search, recommenders, and LLMs |

---

## 🧠 60-second mental drill

Answer each in one sentence, without looking back:

1. **Day 21** — Why is Euclidean distance between two embeddings a meaningful measure of similarity?
2. **Day 22** — How does "slope" relate to the concept of a gradient in training a model?
3. **Day 23** — Why must PCA's principal components be orthogonal to each other?
4. **Day 24** — Why does a stretched (elliptical) loss landscape make gradient descent converge more slowly than a circular one?
5. **Day 25** — Why is cosine similarity often preferred over Euclidean distance for comparing embeddings?

<details>
<summary>Show quick answers</summary>

1. Because every embedding is just a point in a coordinate space, and Euclidean distance measures how far apart two such points are — smaller distance means the underlying concepts are more alike.
2. Slope is the rate of change of output with respect to input for a straight line; a gradient is exactly this idea generalized to functions with many inputs — it's what tells gradient descent which direction and how far to step.
3. Orthogonal components each capture a different, non-overlapping direction of variance; if they weren't orthogonal, components could describe overlapping information, wasting representational capacity.
4. With stretched contours, a step size that's safe in one direction is too large or too small in another, causing the optimizer to zig-zag and take a much longer, less stable path to the minimum.
5. Cosine similarity looks only at the angle (direction) between vectors, ignoring their magnitude — so two embeddings that mean the same thing but differ in scale/intensity still register as highly similar, unlike Euclidean distance which is thrown off by magnitude differences.

</details>

---

## 🗺️ Layer 3 concept map (in one paragraph)

This layer builds the geometric vocabulary that later reappears everywhere in ML. Coordinates and distance (Day 21) let us treat any numeric data — even a 300-dimensional embedding — as a literal point in space we can measure between. Slope (Day 22) is the earliest, simplest form of the idea that becomes "derivative" and "gradient" in later layers. Lines, parallelism, and perpendicularity (Day 23) show up the moment you fit a regression line or need orthogonal, non-redundant directions like PCA's components. Circles and parabolas (Day 24) are literally the shape of a loss function's contours, and that shape directly controls how gradient descent behaves. And vectors, angles, and projections (Day 25) are the single most important geometric idea in modern AI — because every embedding *is* a vector, and comparing vectors by angle (cosine similarity) is how models decide what's semantically similar. Geometry, in other words, is the mental map you'll keep reusing once things get "high-dimensional."

---
**Full days:** [21](day-21-coordinate-system-and-distance.md) · [22](day-22-midpoint-and-slope.md) · [23](day-23-line-equation-parallel-perpendicular.md) · [24](day-24-circle-and-parabola.md) · [25](day-25-vector-length-angle-projection-embedding-space.md)

**Next layer:** Layer 4 — Calculus & Change →
