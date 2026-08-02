# 🌊 Layer 4 — Summary Review

**Calculus & Change**

> A fast recap of all 7 days. Use this to refresh before an interview or right after finishing the full daily files. If a row doesn't ring a bell, go back to that day's file.

| Day | Topic | Core Idea (one line) | Where it shows up in ML/AI |
|---|---|---|---|
| [26](day-26-the-concept-of-a-limit.md) | The Concept of a Limit | What a function approaches as the input gets arbitrarily close to a point | The mathematical foundation of the derivative — "instantaneous rate of change" |
| [27](day-27-continuity.md) | Continuity | A function you can draw without lifting the pen — no jumps or gaps | ReLU is continuous but not smooth at 0 — why gradients need special handling there |
| [28](day-28-derivative-as-rate-of-change.md) | Derivative as Rate of Change | Slope, but at every single point of a curve | "If I nudge this weight, how much does the loss change?" — exactly the gradient |
| [29](day-29-derivative-rules.md) | Derivative Rules & Key Function Derivatives | Ready-made recipes instead of computing a limit from scratch every time | Backprop uses the chain rule to compute millions of gradients efficiently |
| [30](day-30-optimization-critical-points-max-min.md) | Optimization, Critical Points, Max/Min | Where the slope hits zero — a flat point: max, min, or saddle | Training = searching for near-zero gradient = "the model has converged" |
| [31](day-31-integral-and-area-under-curve.md) | Integral & Area Under the Curve | Summing infinitely many tiny slices to get a whole | AUC-ROC is literally an integral — the area under the ROC curve |
| [32](day-32-fundamental-theorem-of-calculus-and-gradient-intuition.md) | Fundamental Theorem of Calculus & Gradient Intuition | Derivative and integral are two sides of the same coin; gradient = derivative in many dimensions | Gradient Descent: the vector that tells every parameter which way to move to reduce loss |

---

## 🧠 60-second mental drill

Answer each in one sentence, without looking back:

1. **Day 26–27** — Why do we need a limit to define an "instantaneous" slope, and why does ReLU's kink at zero matter for gradients?
2. **Day 28–29** — What question is a derivative really answering during model training, and why can't we compute gradients via raw limits for millions of parameters?
3. **Day 30** — Why doesn't a zero gradient always mean you've found the best model?
4. **Day 31** — Why does an AUC near 1 mean a good classifier and near 0.5 mean a random one?
5. **Day 32** — Why does Gradient Descent move opposite to the gradient, not along it?

<details>
<summary>Show quick answers</summary>

1. A straight line has one constant slope everywhere, but a curve's steepness changes at every point — a limit is what lets us zoom into a single point and ask "what is the slope *right here*," even though that exact point alone has no slope of its own. ReLU has a sharp kink at 0 where the slope jumps discontinuously from 0 to 1, so there's no single well-defined derivative there — frameworks pick a convention (e.g. 0) so backprop always has something to compute with.
2. It's answering "if I change this weight slightly, how much does the loss change?" — exactly the question gradient descent needs answered for every parameter. Computing that via raw limits (tiny numeric steps) for millions of parameters one at a time would be far too slow; derivative rules like the chain rule let backprop compute all of them at once, in closed form.
3. Because gradient descent only sees the local neighborhood — a gradient of zero can mean a local minimum, a saddle point, or the true global minimum, and there's no way to tell just from the gradient being zero which one you've landed on.
4. AUC is the area under the ROC curve; a good classifier's curve bulges toward the top-left corner (high true-positive rate at low false-positive rate), pushing the area close to 1, while a random classifier's curve sits on the diagonal, giving an area of about 0.5.
5. The gradient always points in the direction of steepest *increase*; since training wants to *minimize* the loss, you have to step in the exact opposite direction — downhill, not uphill.

</details>

---

## 🗺️ Layer 4 concept map (in one paragraph)

This layer is where "change" becomes something you can compute with precisely. Limits (Day 26) and continuity (Day 27) are the quiet groundwork — they let us talk rigorously about what happens *at* a point and *near* a point, which is exactly what activation functions like ReLU force us to confront. The derivative (Day 28) turns that groundwork into the single most important question in ML — "how much does the output change if I nudge the input?" — and derivative rules (Day 29) are what make answering that question at scale (millions of parameters) computationally possible via backpropagation. Optimization and critical points (Day 30) are what training *is*: hunting for a flat spot in the loss landscape, with the caveat that not every flat spot is the best one. Integrals (Day 31) flip the direction — instead of a rate, they accumulate a total, which is exactly what AUC-ROC measures. And the Fundamental Theorem (Day 32) ties derivative and integral together while generalizing the derivative into the gradient — the vector compass that Gradient Descent follows, in reverse, all the way down to a minimum.

---
**Full days:** [26](day-26-the-concept-of-a-limit.md) · [27](day-27-continuity.md) · [28](day-28-derivative-as-rate-of-change.md) · [29](day-29-derivative-rules.md) · [30](day-30-optimization-critical-points-max-min.md) · [31](day-31-integral-and-area-under-curve.md) · [32](day-32-fundamental-theorem-of-calculus-and-gradient-intuition.md)

**Next layer:** Layer 5 — Discrete Mathematics →
